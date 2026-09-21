from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.schedule_output import ScheduleOutput


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._inflight: set[int] = set()
        if self.max_num_batched_tokens <= 0 or self.max_num_seqs <= 0:
            raise ValueError('token budget and max_num_seqs must be positive')

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # Keep the existing full-sequence allocation contract, fail impossible requests.
        if seq.num_blocks > len(self.block_manager.blocks):
            raise ValueError('request exceeds KV block capacity under full-sequence reservation')
        self.waiting.append(seq)

    def schedule(self) -> ScheduleOutput:
        if self._inflight:
            raise RuntimeError('postprocess the previous schedule before scheduling again')
        decode_seqs, prefill_seqs = [], []
        remaining = self.max_num_batched_tokens
        # Reserve decode budget and append blocks before admitting any prefill.
        while self.running and len(decode_seqs) < self.max_num_seqs and remaining:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                elif self._release_waiting_reservation():
                    # Partial prefills can hold every free block across iterations.
                    # Reclaim their KV before sacrificing the last active decoder.
                    continue
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
                remaining -= 1
        self.running.extendleft(reversed(decode_seqs))

        # Leave selected prefills WAITING until their GPU work succeeds. Do not
        # revisit them within this iteration or cross an unfinished FIFO head.
        for seq in self.waiting:
            if not remaining or len(decode_seqs) + len(prefill_seqs) >= self.max_num_seqs:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                while (num_cached_blocks == -1 and not decode_seqs and not prefill_seqs
                       and not self.running and self._release_waiting_reservation(exclude=seq)):
                    num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                self.block_manager.allocate(seq, num_cached_blocks)
            num_tokens = seq.num_tokens - seq.num_cached_tokens
            seq.is_prefill = True
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            remaining -= seq.num_scheduled_tokens
            prefill_seqs.append(seq)
            if seq.num_scheduled_tokens < num_tokens:
                break
        if not decode_seqs and not prefill_seqs:
            raise RuntimeError('no schedulable work: KV capacity cannot satisfy the pending request')
        self._inflight = {s.seq_id for s in (*decode_seqs, *prefill_seqs)}
        return ScheduleOutput(tuple(decode_seqs), tuple(prefill_seqs))

    def _release_waiting_reservation(self, exclude=None) -> bool:
        # Keep FIFO position; recompute discarded partial KV later. Selected
        # decode requests are outside this queue and can never be evicted here.
        for victim in reversed(self.waiting):
            if victim is not exclude and victim.block_table:
                self.block_manager.deallocate(victim)
                victim.num_scheduled_tokens = 0
                victim.is_prefill = True
                return True
        return False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_scheduled_tokens = 0
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence] | tuple[Sequence, ...], token_ids: list[int]):
        if len(seqs) != len(token_ids):
            raise ValueError('one sampled token is required per scheduled sequence')
        ids = [seq.seq_id for seq in seqs]
        if len(ids) != len(set(ids)) or any(i not in self._inflight for i in ids):
            raise ValueError('postprocess received unscheduled or duplicate sequences')
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            self._inflight.remove(seq.seq_id)
            if seq.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            if seq.is_prefill:
                self.waiting.remove(seq)
                seq.status = SequenceStatus.RUNNING
                seq.is_prefill = False
                self.running.append(seq)
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
