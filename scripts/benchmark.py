import argparse
import atexit
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[1]


def prompt(length, seed):
    rng = random.Random(seed)
    return [rng.randrange(100, 10000) for _ in range(length)]


def quantile(values, q):
    if not values:
        return None
    return sorted(values)[min(len(values)-1, int((len(values)-1)*q))]


class Probe:
    def __init__(self, engine):
        self.engine = engine
        self.records = {}
        self.phases = []
        self.graph_replays = 0
        self.enabled = False
        self.force = False
        self.logits = {}
        self.active = []
        self.step_pending = []
        runner = engine.model_runner
        original_run = runner.run
        original_postprocess = engine.scheduler.postprocess
        probe = self
        class GraphProxy:
            def __init__(self, graph):
                self.graph = graph
            def replay(self):
                if probe.enabled:
                    probe.graph_replays += 1
                return self.graph.replay()
        if not runner.enforce_eager:
            runner.graphs = {bs: GraphProxy(g) for bs, g in runner.graphs.items()}

        def run(seqs, is_prefill):
            self.active = [(s, s.num_completion_tokens,
                            not is_prefill or s.num_cached_tokens+s.num_scheduled_tokens == len(s)) for s in seqs]
            count = sum(s.num_scheduled_tokens for s in seqs)
            start = time.perf_counter()
            tokens = original_run(seqs, is_prefill)
            if self.enabled:
                self.phases.append({'prefill': is_prefill, 'tokens': count,
                                    'seconds': time.perf_counter()-start})
            if self.force:
                tokens = [700 + self.records[s.seq_id]['logical']*17 + pos for s, pos, _ in self.active]
            return tokens

        def postprocess(seqs, tokens, *args):
            before = {s.seq_id: s.num_completion_tokens for s in seqs}
            result = original_postprocess(seqs, tokens, *args)
            now = time.perf_counter()
            if self.enabled:
                for s in seqs:
                    if s.num_completion_tokens > before[s.seq_id]:
                        self.records[s.seq_id]['times'].append(now)
                        self.step_pending.append(s.seq_id)
                    if s.is_finished:
                        self.records[s.seq_id]['finished'] = now
            return result
        runner.run = run
        engine.scheduler.postprocess = postprocess

    def add(self, length, logical, output=16, seed=100):
        from nanovllm import SamplingParams
        self.engine.add_request(prompt(length, seed+logical), SamplingParams(temperature=0.8, max_tokens=output, ignore_eos=True))
        seq = self.engine.scheduler.waiting[-1]
        self.records[seq.seq_id] = {'logical': logical, 'seq': seq, 'arrival': time.perf_counter(),
                                    'times': [], 'step_times': [], 'finished': None, 'existing': False}
        return seq

    def step(self):
        self.step_pending.clear()
        output = self.engine.step()
        end = time.perf_counter()
        for seq_id in self.step_pending:
            self.records[seq_id]['step_times'].append(end)
        if self.enabled:
            for seq_id, _ in output[0]:
                self.records[seq_id]['step_finished'] = end
        return output

    def drain(self):
        for _ in range(10000):
            if self.engine.is_finished():
                return
            self.step()
        raise AssertionError('engine did not finish within 10000 steps')

    def capture(self, module, inputs, output):
        for row, (seq, pos, final) in zip(output, self.active):
            if final:
                key = (self.records[seq.seq_id]['logical'], pos)
                assert key not in self.logits, key
                self.logits[key] = row.detach().cpu()

    def correctness(self, destination):
        from nanovllm import SamplingParams
        torch.manual_seed(123)
        smoke = self.engine.generate([prompt(64, 333)], SamplingParams(temperature=0.8, max_tokens=8, ignore_eos=True), use_tqdm=False)
        self.force = True
        hook = self.engine.model_runner.model.lm_head.register_forward_hook(self.capture)
        first = [self.add(n, i, output=4, seed=1000) for i, n in enumerate([33, 61])]
        while any(s.num_completion_tokens == 0 for s in first):
            self.step()
        for i, n in enumerate([513, 769], 2):
            self.add(n, i, output=4, seed=1000)
        self.drain()
        hook.remove()
        self.force = False
        assert len(self.logits) == 16, self.logits.keys()
        torch.save({'logits': self.logits, 'forced_tokens': {r['logical']: r['seq'].completion_token_ids for r in self.records.values()},
                    'smoke_tokens': smoke[0]['token_ids']}, destination)
        self.records.clear()
        self.logits.clear()

    def case(self, name, rep, warm=False):
        assert self.engine.is_finished()
        self.enabled = False
        self.records.clear()
        self.phases.clear()
        self.graph_replays = 0
        torch.manual_seed(500+rep)
        case_index = ['all_decode', 'all_prefill', 'short_decode', 'long_decode', 'many_long'].index(name)
        seed = 10000 + rep*100 + case_index*1000 + (100000 if warm else 0)
        if name != 'all_prefill':
            active = [self.add(32, i, output=32, seed=seed) for i in range(4)]
            while any(s.num_completion_tokens == 0 for s in active):
                self.step()
            for r in self.records.values():
                r['existing'] = True
        torch.cuda.synchronize()
        start = time.perf_counter()
        for r in self.records.values():
            r['arrival'] = start
        self.enabled = True
        lengths = {'all_decode': [], 'all_prefill': [300, 500, 1000],
                   'short_decode': [64], 'long_decode': [3000], 'many_long': [3000]*4}[name]
        for i, length in enumerate(lengths, 4):
            self.add(length, i, output=1 if name == 'all_prefill' else 16, seed=seed)
        iterations = mixed_iterations = 0
        while not self.engine.is_finished():
            before = len(self.phases)
            self.step()
            iterations += 1
            mixed_iterations += len({p['prefill'] for p in self.phases[before:]}) == 2
        wall = time.perf_counter()-start
        self.enabled = False
        requests = []
        for r in self.records.values():
            t = r['times']
            assert t and r['finished'] is not None
            requests.append({'logical': r['logical'], 'existing': r['existing'], 'output_tokens': len(t),
                'ttft_ms': None if r['existing'] else (t[0]-r['arrival'])*1000,
                'step_ttft_ms': None if r['existing'] else (r['step_times'][0]-r['arrival'])*1000,
                'tpot_ms': (t[-1]-t[0])*1000/(len(t)-1) if len(t)>1 else None,
                'gap_ms': [(b-a)*1000 for a,b in zip(([start]+t[:-1]) if r['existing'] else t[:-1], t if r['existing'] else t[1:])],
                'step_gap_ms': [(b-a)*1000 for a,b in zip(([start]+r['step_times'][:-1]) if r['existing'] else r['step_times'][:-1], r['step_times'] if r['existing'] else r['step_times'][1:])],
                'internal_e2e_ms': (r['finished']-r['arrival'])*1000,
                'e2e_ms': (r['step_finished']-r['arrival'])*1000})
        old_gaps = [g for r in requests if r['existing'] for g in r['gap_ms']]
        phase = {}
        for is_prefill, label in ((True, 'prefill'), (False, 'decode')):
            selected = [p for p in self.phases if p['prefill'] == is_prefill]
            seconds = sum(p['seconds'] for p in selected)
            count = sum(p['tokens'] for p in selected)
            phase[label+'_tok_s'] = count/seconds if seconds else None
            phase[label+'_tokens'] = count
        return {'name': name, 'rep': rep, 'wall_ms': wall*1000,
            'output_tok_s': sum(r['output_tokens'] for r in requests)/wall,
            'new_ttft_ms': statistics.mean([r['ttft_ms'] for r in requests if r['ttft_ms'] is not None]) if lengths else None,
            'existing_gap_p95_ms': quantile(old_gaps, .95), 'existing_gap_max_ms': max(old_gaps) if old_gaps else None,
            'existing_step_gap_max_ms': max([g for r in requests if r['existing'] for g in r['step_gap_ms']], default=None),
            'iterations': iterations, 'mixed_iterations': mixed_iterations, 'graph_replays': self.graph_replays,
            **phase, 'requests': requests}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['baseline', 'mixed'], required=True)
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--graph', type=int, default=1)
    parser.add_argument('--action', choices=['correctness', 'benchmark'], default='correctness')
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    backend_root = REPO / ('src' if args.backend == 'mixed' else 'reference')
    sys.path.insert(0, str(backend_root))
    import nanovllm
    from nanovllm import LLM
    torch._dynamo.config.recompile_limit = 64
    torch.manual_seed(123)
    engine = LLM(os.environ['NANOVLLM_MODEL'], enforce_eager=not args.graph,
                 max_num_seqs=16, max_num_batched_tokens=args.budget, max_model_len=4096,
                 gpu_memory_utilization=0.6)
    probe = Probe(engine)
    dest = Path(os.environ.get('NANOVLLM_RESULTS_DIR', str(REPO/'results/latest')))
    dest.mkdir(parents=True, exist_ok=True)
    tag = f'{args.backend}_b{args.budget}_g{args.graph}' + ('_'+args.label if args.label else '')
    metadata = {**vars(args), 'import': nanovllm.__file__, 'torch': torch.__version__,
                'gpu': torch.cuda.get_device_name(), 'dtype': str(engine.model_runner.config.hf_config.dtype),
                'blocks': engine.model_runner.config.num_kvcache_blocks}
    if args.action == 'correctness':
        probe.correctness(dest/(tag+'_correctness.pt'))
        metadata['passed_local_assertions'] = True
        print('CORRECTNESS SAVED', tag, flush=True)
    else:
        cases = []
        names = ['all_decode', 'all_prefill', 'short_decode', 'long_decode', 'many_long']
        for name in names:
            probe.case(name, 99, warm=True)
        for rep in range(args.reps):
            for name in names:
                case = probe.case(name, rep)
                cases.append(case)
                print(tag, json.dumps({k:v for k,v in case.items() if k != 'requests'}), flush=True)
        metadata['cases'] = cases
    (dest/(tag+'_'+args.action+'.json')).write_text(json.dumps(metadata, indent=2))
    atexit.unregister(engine.exit)
    engine.exit()


if __name__ == '__main__':
    main()
