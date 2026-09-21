import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, reset_context


class ExecutionTests(unittest.TestCase):
    def test_engine_decode_then_prefill(self):
        Sequence.block_size = 4
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = Scheduler(SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=8,
            eos=99999, num_kvcache_blocks=32, kvcache_block_size=4))
        calls = []
        def run(method, seqs, is_prefill):
            self.assertEqual(method, 'run')
            calls.append((is_prefill, [s.seq_id for s in seqs]))
            return [123] * len(seqs)
        engine.model_runner = SimpleNamespace(call=run)
        a = Sequence([1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
        engine.scheduler.add(a)
        engine.step()
        b = Sequence(list(range(20)), SamplingParams(max_tokens=2, ignore_eos=True))
        engine.scheduler.add(b)
        calls.clear()
        outputs, stats = engine.step()
        self.assertEqual(calls, [(False, [a.seq_id]), (True, [b.seq_id])])
        self.assertEqual(outputs, [(a.seq_id, [123, 123])])
        self.assertEqual((stats.num_decode_tokens, stats.num_prefill_tokens), (1, 7))
        self.assertEqual(b.num_cached_tokens, 7)
        self.assertEqual(b.num_completion_tokens, 0)
        self.assertGreaterEqual(stats.elapsed_seconds, stats.decode_seconds + stats.prefill_seconds)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires GPU for real metadata tensors')
    def test_real_runner_chunk_and_decode_metadata(self):
        Sequence.block_size = 4
        runner = ModelRunner.__new__(ModelRunner)
        runner.block_size = 4
        seq = Sequence(list(range(10, 19)))
        seq.block_table = [7, 2, 10]
        seq.num_cached_tokens = 3
        seq.num_scheduled_tokens = 4
        ids, positions = runner.prepare_prefill([seq])
        context = get_context()
        self.assertEqual(ids.tolist(), [13, 14, 15, 16])
        self.assertEqual(positions.tolist(), [3, 4, 5, 6])
        self.assertEqual(context.slot_mapping.tolist(), [31, 8, 9, 10])
        self.assertEqual(context.cu_seqlens_q.tolist(), [0, 4])
        self.assertEqual(context.cu_seqlens_k.tolist(), [0, 7])
        self.assertEqual(context.block_tables.tolist(), [[7, 2, 10]])
        reset_context()
        ids, positions = runner.prepare_decode([seq])
        context = get_context()
        self.assertEqual(ids.tolist(), [18])
        self.assertEqual(positions.tolist(), [8])
        self.assertEqual(context.slot_mapping.tolist(), [40])
        self.assertEqual(context.context_lens.tolist(), [9])
        reset_context()


if __name__ == '__main__':
    unittest.main()
