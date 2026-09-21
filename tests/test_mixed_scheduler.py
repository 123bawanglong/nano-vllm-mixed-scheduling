import pickle
import random
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


class SchedulerTests(unittest.TestCase):
    def make(self, budget=16, seqs=16, blocks=128, size=4):
        Sequence.block_size = size
        return Scheduler(SimpleNamespace(max_num_seqs=seqs, max_num_batched_tokens=budget,
            eos=99999, num_kvcache_blocks=blocks, kvcache_block_size=size))

    def seq(self, n, offset=10, max_tokens=8, ignore=True):
        return Sequence(list(range(offset, offset+n)), SamplingParams(max_tokens=max_tokens, ignore_eos=ignore))

    def schedule(self, scheduler):
        output = scheduler.schedule()
        self.assertTrue(hasattr(output, 'decode_seqs'), 'schedule() must return explicit mixed ScheduleOutput')
        ids = [s.seq_id for s in (*output.decode_seqs, *output.prefill_seqs)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(len(ids), scheduler.max_num_seqs)
        self.assertLessEqual(output.num_scheduled_tokens, scheduler.max_num_batched_tokens)
        self.assertEqual(output.num_decode_tokens, len(output.decode_seqs))
        self.assertTrue(all(s.num_scheduled_tokens == 1 for s in output.decode_seqs))
        return output

    def finish(self, scheduler, output, token=123):
        before = output.num_scheduled_tokens
        for batch in (output.decode_seqs, output.prefill_seqs):
            if batch:
                scheduler.postprocess(batch, [token] * len(batch))
        self.assertEqual(output.num_scheduled_tokens, before)  # counts are snapshots
        self.invariants(scheduler)

    def invariants(self, scheduler):
        live = [*scheduler.running, *scheduler.waiting]
        self.assertEqual(len(live), len({s.seq_id for s in live}))
        refs = Counter(b for s in live for b in s.block_table)
        bm = scheduler.block_manager
        self.assertEqual(set(bm.free_block_ids) | bm.used_block_ids, set(range(len(bm.blocks))))
        self.assertFalse(set(bm.free_block_ids) & bm.used_block_ids)
        self.assertEqual(len(bm.free_block_ids), len(set(bm.free_block_ids)))
        for block in bm.blocks:
            self.assertEqual(block.ref_count, refs[block.block_id])
        for s in live:
            self.assertEqual(s.num_scheduled_tokens, 0)
            self.assertLessEqual(s.num_cached_tokens, s.num_tokens)
        self.assertTrue(all(s.status == SequenceStatus.RUNNING for s in scheduler.running))
        self.assertTrue(all(s.status == SequenceStatus.WAITING for s in scheduler.waiting))

    def prime(self, scheduler, seqs):
        for s in seqs:
            scheduler.add(s)
        while any(s.status != SequenceStatus.RUNNING for s in seqs):
            self.finish(scheduler, self.schedule(scheduler))

    def test_prefill_only_final_state_and_decode_only(self):
        sch = self.make()
        a = self.seq(5)
        sch.add(a)
        out = self.schedule(sch)
        self.assertEqual(out.num_prefill_tokens, 5)
        self.assertFalse(out.decode_seqs)
        self.assertEqual(a.status, SequenceStatus.WAITING)  # execution hasn't succeeded yet
        self.finish(sch, out)
        self.assertEqual(a.num_cached_tokens, 5)
        self.assertEqual(a.num_tokens, 6)
        out = self.schedule(sch)
        self.assertEqual(out.decode_seqs, (a,))
        self.assertFalse(out.prefill_seqs)
        self.finish(sch, out)
        self.assertEqual(a.num_cached_tokens, 6)

    def test_three_decode_plus_1021_prefill(self):
        sch = self.make(budget=1024, blocks=2048)
        abc = [self.seq(4, offset=10+20*i) for i in range(3)]
        self.prime(sch, abc)
        d = self.seq(3000, 1000)
        sch.add(d)
        out = self.schedule(sch)
        self.assertEqual(out.decode_seqs, tuple(abc))
        self.assertEqual(out.prefill_seqs, (d,))
        self.assertEqual(d.num_scheduled_tokens, 1021)
        self.assertEqual(out.num_scheduled_tokens, 1024)
        self.finish(sch, out)
        self.assertEqual(d.num_cached_tokens, 1021)
        self.assertEqual(d.num_completion_tokens, 0)
        self.assertEqual(d.status, SequenceStatus.WAITING)

    def test_long_prompt_chunks(self):
        sch = self.make(budget=512, blocks=1024)
        a = self.seq(3000)
        sch.add(a)
        chunks = []
        while a.status == SequenceStatus.WAITING:
            out = self.schedule(sch)
            chunks.append(out.num_prefill_tokens)
            self.finish(sch, out)
        self.assertEqual(chunks, [512]*5+[440])
        self.assertEqual(a.num_completion_tokens, 1)

    def test_fifo_multiple_prefills_fill_remainder(self):
        sch = self.make(budget=1024, blocks=1024)
        reqs = [self.seq(n, i*2000) for i,n in enumerate((300,500,1000))]
        for s in reqs: sch.add(s)
        out = self.schedule(sch)
        self.assertEqual(out.prefill_seqs, tuple(reqs))
        self.assertEqual([s.num_scheduled_tokens for s in reqs], [300,500,224])
        self.finish(sch, out)
        self.assertEqual(list(sch.waiting), [reqs[-1]])

    def test_budget_and_max_seqs_apply_jointly(self):
        sch = self.make(budget=16, seqs=8)
        abc = [self.seq(2, i*10) for i in range(3)]
        self.prime(sch, abc)
        sch.add(self.seq(20,100))
        sch.max_num_batched_tokens = 2
        out = self.schedule(sch)
        self.assertEqual(len(out.decode_seqs), 2)
        self.assertFalse(out.prefill_seqs)
        self.finish(sch, out)
        sch.max_num_batched_tokens = 16
        sch.max_num_seqs = 3
        out = self.schedule(sch)
        self.assertEqual(len(out.decode_seqs), 3)
        self.assertFalse(out.prefill_seqs)
        self.finish(sch, out)

    def test_block_boundaries_and_eos_release(self):
        for length in (3,4,5,7,8,9):
            with self.subTest(length=length):
                sch = self.make(budget=3)
                a = self.seq(length,max_tokens=2)
                sch.add(a)
                for _ in range(20):
                    if sch.is_finished(): break
                    self.finish(sch,self.schedule(sch))
                self.assertTrue(a.is_finished)
                self.assertFalse(a.block_table)
                self.assertEqual(len(sch.block_manager.free_block_ids),128)
        sch = self.make()
        abc = [self.seq(2,i*20,ignore=False) for i in range(3)]
        self.prime(sch,abc)
        out = self.schedule(sch)
        sch.postprocess(out.decode_seqs,[sch.eos,100,100])
        self.assertTrue(abc[0].is_finished)
        self.assertEqual(list(sch.running),abc[1:])
        self.invariants(sch)

    def test_running_tail_preemption(self):
        sch = self.make(blocks=3)
        a,b = self.seq(4,10),self.seq(4,30)
        self.prime(sch,[a,b])
        out = self.schedule(sch)
        self.assertEqual(out.decode_seqs,(a,))
        self.assertEqual(b.status,SequenceStatus.WAITING)
        self.assertFalse(b.block_table)
        self.finish(sch,out)

    def test_partial_prefill_reservation_cannot_block_decode(self):
        sch = self.make(budget=4,blocks=3)
        a = self.seq(3)
        self.prime(sch,[a])
        d = self.seq(8,100)
        sch.add(d)
        first = self.schedule(sch)  # A uses last slot; D reserves the other two blocks
        self.finish(sch,first)
        self.assertEqual(d.num_cached_tokens,3)
        out = self.schedule(sch)  # A now needs a new block; reclaim D reservation
        self.assertEqual(out.decode_seqs,(a,))
        self.assertFalse(d.block_table)
        self.assertEqual(d.num_cached_tokens,0)
        self.finish(sch,out)

    def test_prefix_cache_reuse(self):
        sch = self.make(budget=3)
        a = self.seq(9,max_tokens=1)
        sch.add(a)
        while not sch.is_finished(): self.finish(sch,self.schedule(sch))
        b = self.seq(9,max_tokens=2)
        sch.add(b)
        out = self.schedule(sch)
        self.assertEqual(b.num_cached_tokens,8)
        self.assertEqual(out.num_prefill_tokens,1)
        self.finish(sch,out)

    def test_no_duplicate_schedule_and_bad_postprocess(self):
        sch = self.make()
        a = self.seq(3)
        sch.add(a)
        out = self.schedule(sch)
        with self.assertRaises(RuntimeError): sch.schedule()
        with self.assertRaises(ValueError): sch.postprocess(out.prefill_seqs,[])
        self.assertEqual(a.num_cached_tokens,0)
        self.finish(sch,out)

    def test_impossible_prompt_fails_explicitly(self):
        sch = self.make(blocks=2)
        with self.assertRaises(ValueError): sch.add(self.seq(9))

    def test_sequence_worker_serialization(self):
        sch = self.make(budget=3)
        a = self.seq(8)
        sch.add(a)
        out = self.schedule(sch)
        restored = pickle.loads(pickle.dumps(a))
        self.assertEqual(restored.token_ids,a.token_ids)
        self.assertEqual(restored.num_scheduled_tokens,3)
        self.finish(sch,out)
        while a.status == SequenceStatus.WAITING: self.finish(sch,self.schedule(sch))
        out = self.schedule(sch)
        restored = pickle.loads(pickle.dumps(a))
        self.assertEqual(restored.last_token,a.last_token)
        self.assertEqual(restored.num_scheduled_tokens,1)
        self.finish(sch,out)

    def test_randomized_progress_and_block_accounting(self):
        rng = random.Random(731)
        for trial in range(30):
            sch = self.make(budget=rng.randint(1,13),seqs=rng.randint(1,5),blocks=12)
            requests = []
            for step in range(500):
                if len(requests)<12 and rng.random()<0.25:
                    s=self.seq(rng.randint(1,16),offset=trial*10000+len(requests)*100,max_tokens=3)
                    requests.append(s)
                    sch.add(s)
                if not sch.is_finished(): self.finish(sch,self.schedule(sch))
                if len(requests)==12 and sch.is_finished(): break
            self.assertTrue(sch.is_finished(), f'failed to progress: trial={trial}')
            self.assertTrue(all(s.is_finished for s in requests))


if __name__ == '__main__': unittest.main(verbosity=2)
