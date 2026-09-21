"""Summarize the raw process results without changing their measurements."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    dest = args.results.resolve()
    summary = []
    raw_rows = {'baseline': [], 'mixed': []}
    for budget in (128, 512, 1024):
        for name in ('all_decode', 'all_prefill', 'short_decode', 'long_decode', 'many_long'):
            result = {'budget': budget, 'workload': name}
            for backend in raw_rows:
                cases = []
                for label in ('ab', 'ba'):
                    data = json.loads((dest / f'{backend}_b{budget}_g1_{label}_benchmark.json').read_text())
                    selected = [c for c in data['cases'] if c['name'] == name]
                    if len(selected) != 3:
                        raise ValueError(f'Expected three repetitions: {backend}/{budget}/{label}/{name}')
                    cases.extend(selected)
                    for case in selected:
                        raw_rows[backend].append(dict(budget=budget, order=label, workload=name,
                            repeat=case['rep'], max_step_gap_ms=case['existing_step_gap_max_ms'],
                            output_tok_s=case['output_tok_s'], mixed_iterations=case['mixed_iterations'],
                            graph_replays=case['graph_replays']))
                stats = {'repetitions': len(cases)}
                for key in ('existing_step_gap_max_ms', 'output_tok_s', 'mixed_iterations', 'graph_replays'):
                    values = [c[key] for c in cases if c[key] is not None]
                    stats[key] = statistics.median(values) if values else None
                ttft = [statistics.mean(r['step_ttft_ms'] for r in c['requests'] if not r['existing'])
                        for c in cases if any(not r['existing'] for r in c['requests'])]
                stats['step_ttft_ms'] = statistics.median(ttft) if ttft else None
                result[backend] = stats
            a = result['baseline']['existing_step_gap_max_ms']
            b = result['mixed']['existing_step_gap_max_ms']
            result['gap_reduction_pct'] = 100 * (1 - b/a) if a else None
            summary.append(result)
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    for backend, rows in raw_rows.items():
        with (dest / f'{backend}.tsv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0], delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
    lines = ['budget workload       baseline_gap_ms mixed_gap_ms reduction_pct baseline_tok/s mixed_tok/s']
    for item in summary:
        a, b = item['baseline'], item['mixed']
        def fmt(value):
            return 'N/A' if value is None else f'{value:.2f}'
        lines.append(f"{item['budget']:6} {item['workload']:14} "
            f"{fmt(a['existing_step_gap_max_ms']):>15} {fmt(b['existing_step_gap_max_ms']):>12} "
            f"{fmt(item['gap_reduction_pct']):>13} {a['output_tok_s']:14.2f} {b['output_tok_s']:11.2f}")
    text = '\n'.join(lines) + '\n'
    (dest / 'comparison.txt').write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
