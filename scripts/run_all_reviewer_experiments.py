"""
Runs every experiment requested by the IoT-J reviewers on IoT-66079-2026.

Coverage:
  R1-2  : mu=0 ablation                          -> Table (extended)
  R1-3  : multi-seed replication (3 seeds)       -> mean-of-run stats
  R2-4  : per-component strategy ablation (5)    -> Table (ablation)
  R2-5  : same as R1-3
  R2-6  : record-level train/val/test split      -> separate script (below)
  R2-7  : rule metrics (post-hoc, ALREADY done)  -> compute_rule_metrics.py

Usage:
  python scripts/run_all_reviewer_experiments.py                  # all
  python scripts/run_all_reviewer_experiments.py --only seeds     # only seeds
  python scripts/run_all_reviewer_experiments.py --dry-run        # show cmds

Estimated wall-clock on RTX 5070 laptop (single dataset both phases):
  ~ 90 min per run x 3 seeds x 5 cells = 22.5 h  per dataset
  ~ 45 h total for both CICIoT2023 + DIAD2024
Split into batches; each cell writes its own results folder.
"""
import argparse, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / 'scripts' / 'training' / 'train_rl_multiagent_StrategyB.py'
PY = sys.executable

# Cell = (label, lambda, mu, extra flags)
CELLS = {
    'a_scalar':      dict(label='(a) Strategy-A scalar baseline (already run)', skip=True),
    'b_mosar_only':  dict(label='(b) MOSAR alone, lam=0, mu=0',      lam=0.00, mu=0.00),
    'c_mosar_link':  dict(label='(c) MOSAR+R_link, lam=0, mu=0.10',  lam=0.00, mu=0.10),
    'd_mosar_fuzzy': dict(label='(d) MOSAR+AFRAS-FL, lam=0.25, mu=0',lam=0.25, mu=0.00),
    'e_complete':    dict(label='(e) Complete R_total (canonical)',  lam=0.25, mu=0.10),
}

SEEDS = [42, 123, 2026]
DATASETS = ['ciciot2023', 'diad2024']

def cmd_for(cell_key, cell, seed, dataset):
    suffix = f'_R2-4_{cell_key}_s{seed}'
    return [
        PY, str(TRAIN),
        '--dataset', dataset,
        '--lambda-fuzzy', str(cell['lam']),
        '--mu-link',      str(cell['mu']),
        '--seed',         str(seed),
        '--suffix',       suffix,
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['seeds','ablation','all'], default='all')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    args = ap.parse_args()

    plan = []
    for cell_key, cell in CELLS.items():
        if cell.get('skip'): continue
        if args.only == 'seeds' and cell_key != 'e_complete': continue
        if args.only == 'ablation' and cell_key == 'a_scalar': continue
        for seed in SEEDS:
            for ds in args.datasets:
                plan.append((cell_key, cell['label'], seed, ds))

    print(f'Planned runs: {len(plan)}')
    for i, (ck, lbl, s, ds) in enumerate(plan, 1):
        print(f'  {i:2d}. cell={ck:14s} seed={s:5d} dataset={ds:10s}  {lbl}')
    if args.dry_run:
        print('\nDRY RUN — nothing executed.')
        return

    for i, (ck, lbl, seed, ds) in enumerate(plan, 1):
        t0 = time.time()
        cmd = cmd_for(ck, CELLS[ck], seed, ds)
        print(f'\n[{i}/{len(plan)}] launching: {" ".join(cmd)}')
        rc = subprocess.call(cmd)
        dt = (time.time()-t0)/60
        print(f'  -> rc={rc}   wall={dt:.1f} min')
        if rc != 0:
            print(f'  !!! FAILURE — continuing to next cell')

if __name__ == '__main__':
    main()
