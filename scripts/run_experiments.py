import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
DEST = Path(os.environ.get('NANOVLLM_RESULTS_DIR', str(REPO/'results/latest')))


def run(backend, budget, graph, action, label=''):
    tag = f'{backend}_b{budget}_g{graph}' + ('_'+label if label else '')
    cmd = [sys.executable, str(REPO/'scripts/benchmark.py'), '--backend', backend,
           '--budget', str(budget), '--graph', str(graph), '--action', action, '--label', label]
    print('START', tag, action, flush=True)
    with (DEST/(tag+'_'+action+'.log')).open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((DEST/(tag+'_'+action+'.log')).read_text()[-12000:], flush=True)
        raise RuntimeError(f'{tag} {action} failed ({result.returncode})')
    print('DONE', tag, action, flush=True)


def compare(budget, graph):
    a, b = [torch.load(DEST/f'{backend}_b{budget}_g{graph}_correctness.pt', weights_only=True)
            for backend in ('baseline', 'mixed')]
    assert a['smoke_tokens'] == b['smoke_tokens'], 'unmixed sampled output mismatch'
    assert a['forced_tokens'] == b['forced_tokens']
    assert a['logits'].keys() == b['logits'].keys()
    rows = []
    for key in sorted(a['logits']):
        x, y = a['logits'][key].float(), b['logits'][key].float()
        assert x.shape == y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
        rel = ((x-y).norm()/x.norm().clamp_min(1e-12)).item()
        cosine = F.cosine_similarity(x[None], y[None]).item()
        assert rel < .03 and cosine > .999, (key, rel, cosine)
        rows.append({'key': key, 'relative_l2': rel, 'cosine': cosine,
                     'max_abs': (x-y).abs().max().item(), 'top1_equal': x.argmax().item() == y.argmax().item()})
    result = {'budget': budget, 'graph': graph, 'smoke_tokens_equal': True,
              'forced_tokens_equal': True, 'rows': rows}
    (DEST/f'comparison_b{budget}_g{graph}.json').write_text(json.dumps(result, indent=2))
    print('COMPARE PASS', budget, graph, 'max relL2', max(r['relative_l2'] for r in rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['correctness', 'benchmark', 'all'], default='all')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.action in ('correctness', 'all'):
        for budget, graph in ((128, 1), (512, 1), (1024, 1), (512, 0)):
            for backend in ('baseline', 'mixed'):
                run(backend, budget, graph, 'correctness')
            compare(budget, graph)
    if args.action in ('benchmark', 'all'):
        for budget in (128, 512, 1024):
            assert (DEST/f'comparison_b{budget}_g1.json').exists(), 'run correctness first'
            for label, backends in (('ab', ('baseline', 'mixed')), ('ba', ('mixed', 'baseline'))):
                for backend in backends:
                    run(backend, budget, 1, 'benchmark', label)


if __name__ == '__main__':
    main()
