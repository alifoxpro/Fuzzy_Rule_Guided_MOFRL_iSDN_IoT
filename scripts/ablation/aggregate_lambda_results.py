# -*- coding: utf-8 -*-
"""
============================================================================
Aggregate lambda-ablation results for paper 4.

Reads:
    experiments/results_StrategyB_lam{value}/{dataset}/results.json
for each lambda value in LAMBDAS and each dataset in DATASETS, then emits:

    paper4-Fuzzy-Augmented-Reward-Shaping/ablation/lambda_ablation.csv
    paper4-Fuzzy-Augmented-Reward-Shaping/ablation/lambda_ablation.json
    paper4-Fuzzy-Augmented-Reward-Shaping/ablation/lambda_ablation_summary.txt

The CSV has one row per (dataset, lambda, phase) combination and includes:
    MSR (mean, std), JFI, FPR, FNR, mean reward.

Usage:
    python scripts/ablation/aggregate_lambda_results.py

This script is safe to re-run: it will skip missing runs and report them
at the end so you know which ablation cells still need to be executed.
============================================================================
"""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS = PROJECT / 'experiments'
PAPER_DIR = PROJECT / 'paper4-Fuzzy-Augmented-Reward-Shaping'
OUT_DIR = PAPER_DIR / 'ablation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match the values used in run_lambda_sweep.bat. Keep the string form so
# that the folder suffixes are exact (e.g. "0.00" not "0.0").
LAMBDAS = ['0.10', '0.25', '0.50']
DATASETS = ['ciciot2023', 'diad2024']
PHASES = ['phase_b1', 'phase_b2', 'phase_b3', 'phase_b4']
PHASE_LABELS = {
    'phase_b1': 'B1-Independent',
    'phase_b2': 'B2-FedAvg',
    'phase_b3': 'B3-FedProx',
    'phase_b4': 'B4-Fuzzy',
}

# Metrics we extract from each phase dict
METRIC_KEYS = [
    ('mean_msr',       'MSR'),
    ('std_msr',        'MSR_std'),
    ('jain_fairness',  'JFI'),
    ('mean_fpr',       'FPR'),
    ('mean_fnr',       'FNR'),
    ('mean_reward',    'Reward'),
]


def load_run(lam, dataset):
    """Return the parsed results.json for a single (lambda, dataset) run,
    or None if the file does not exist.

    Tries folders in this order:
      1. results_StrategyB_lam{X.XX}/      (full production run)
      2. results_StrategyB_mini_lam{X.XX}/ (B4-only reduced-config run)
      3. results_StrategyB/                (canonical run, only if lam == 0.25)
    """
    candidates = [
        EXPERIMENTS / f'results_StrategyB_lam{lam}' / dataset / 'results.json',
        EXPERIMENTS / f'results_StrategyB_mini_lam{lam}' / dataset / 'results.json',
    ]
    if lam == '0.25':
        candidates.append(
            EXPERIMENTS / 'results_StrategyB' / dataset / 'results.json')

    for c in candidates:
        if c.exists():
            with open(c, 'r') as f:
                return json.load(f), c

    return None, candidates[0]


def main():
    rows = []          # list of dicts, one per (dataset, lambda, phase)
    missing = []       # list of (lambda, dataset) tuples that are missing
    summary_lines = []

    summary_lines.append('=' * 72)
    summary_lines.append('Lambda ablation summary')
    summary_lines.append('=' * 72)

    for dataset in DATASETS:
        summary_lines.append('')
        summary_lines.append(f'Dataset: {dataset}')
        summary_lines.append('-' * 72)
        header = f'{"lambda":>8} {"phase":>15} {"MSR":>10} {"std":>10} {"JFI":>10} {"FPR":>10}'
        summary_lines.append(header)

        for lam in LAMBDAS:
            data, path = load_run(lam, dataset)
            if data is None:
                missing.append((lam, dataset, str(path)))
                summary_lines.append(f'{lam:>8}  <missing: {path.name}>')
                continue

            for phase in PHASES:
                if phase not in data:
                    continue
                p = data[phase]
                if not isinstance(p, dict) or 'mean_msr' not in p:
                    continue

                row = {
                    'dataset': dataset,
                    'lambda': float(lam),
                    'phase': phase,
                    'phase_label': PHASE_LABELS[phase],
                }
                for json_key, col_name in METRIC_KEYS:
                    row[col_name] = p.get(json_key, None)
                rows.append(row)

                summary_lines.append(
                    f'{lam:>8} {PHASE_LABELS[phase]:>15}'
                    f' {row["MSR"]:>10.4f}'
                    f' {row["MSR_std"]:>10.4f}'
                    f' {row["JFI"]:>10.4f}'
                    f' {row["FPR"]:>10.4f}'
                )

    # === Write CSV ===
    csv_path = OUT_DIR / 'lambda_ablation.csv'
    fieldnames = ['dataset', 'lambda', 'phase', 'phase_label'] + [c for _, c in METRIC_KEYS]
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # === Write JSON (for the paper build step) ===
    json_path = OUT_DIR / 'lambda_ablation.json'
    with open(json_path, 'w') as f:
        json.dump({'rows': rows, 'missing': missing}, f, indent=2)

    # === Write summary text ===
    summary_lines.append('')
    summary_lines.append('=' * 72)
    summary_lines.append(f'Total rows collected : {len(rows)}')
    summary_lines.append(f'Missing runs         : {len(missing)}')
    if missing:
        summary_lines.append('')
        summary_lines.append('Missing runs (need to be executed):')
        for lam, ds, p in missing:
            summary_lines.append(f'  lambda={lam}, dataset={ds}')
            summary_lines.append(f'    -> expected at {p}')
    summary_lines.append('=' * 72)

    summary_path = OUT_DIR / 'lambda_ablation_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    # Print to console
    print('\n'.join(summary_lines))
    print()
    print(f'Wrote: {csv_path}')
    print(f'Wrote: {json_path}')
    print(f'Wrote: {summary_path}')

    if missing:
        print()
        print(f'WARNING: {len(missing)} run(s) are missing. The aggregator'
              ' will still produce a CSV with the available cells; re-run'
              ' the sweep for the missing ones when you have time.')
        sys.exit(0)  # Not a hard failure


if __name__ == '__main__':
    main()
