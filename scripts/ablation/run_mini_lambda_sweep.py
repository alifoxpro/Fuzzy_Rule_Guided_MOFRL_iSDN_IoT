# -*- coding: utf-8 -*-
"""
============================================================================
Mini lambda sweep for the Strategy B pipeline.

This is a smoke-test / fast-iteration version of run_lambda_sweep.bat.
It monkey-patches the Strategy B training script to use a tiny dataset
and a tiny number of episodes / FL rounds so that the entire ablation
(3 lambda values, 1 dataset) fits in the time budget of a single
command.

Key reductions compared to the full sweep:
    MAX_ROWS               : 3,000,000  ->  30,000   (100x smaller)
    INDEPENDENT_EPISODES   : 100        ->  5        (20x smaller)
    FL_ROUNDS              : 30         ->  3        (10x smaller)
    LOCAL_EPISODES_PER_ROUND : 5        ->  2        (2.5x smaller)
    dataset                : both       ->  ciciot2023 only

The resulting MSR/JFI/FPR numbers are NOT publication-quality --- they
are a proof-of-plumbing that the argparse overrides, the RESULTS_BASE
suffix, and the LAMBDA_FUZZY global all propagate correctly through
the pipeline. For publication numbers, use run_lambda_sweep.bat with
the full hyperparameters and a real GPU.

This file lives in scripts/ablation/ but is invoked from the project
root like the rest of the ablation tooling:

    python scripts/ablation/run_mini_lambda_sweep.py

It writes results into:
    experiments/results_StrategyB_mini_lam{X.XX}/ciciot2023/
============================================================================
"""
import importlib.util
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
SCRIPT = PROJECT / 'scripts' / 'training' / 'train_rl_multiagent_StrategyB.py'

# Lambda values to sweep. Skipping 0.25 would require merging the canonical
# run separately; because the mini sweep uses a very different training
# config, we run 0.25 here too so that all four cells use the same config.
LAMBDAS = [0.00, 0.10, 0.25, 0.50]

# Reduced-config overrides
MINI_CONFIG = {
    'MAX_ROWS': 30_000,
    'INDEPENDENT_EPISODES': 5,
    'FL_ROUNDS': 3,
    'LOCAL_EPISODES_PER_ROUND': 2,
}


def import_strategy_b_fresh():
    """Import (or re-import) the Strategy B module so that the
    module-level globals start from a clean state each time. This is
    necessary because RESULTS_BASE is set at import time from the
    RESULTS_SUFFIX env var."""
    # Drop any cached version
    for key in list(sys.modules):
        if key.startswith('strategy_b'):
            del sys.modules[key]
    spec = importlib.util.spec_from_file_location('strategy_b', str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_one(lam, dataset):
    suffix = f'_mini_lam{lam:.2f}'
    os.environ['RESULTS_SUFFIX'] = suffix
    os.environ['LAMBDA_FUZZY'] = str(lam)
    os.environ['MU_LINK'] = '0.10'  # same as the canonical run

    print('=' * 70)
    print(f'  MINI SWEEP  |  lambda = {lam:.2f}  |  dataset = {dataset}')
    print(f'  suffix      = {suffix}')
    print('=' * 70)

    mod = import_strategy_b_fresh()

    # Apply the reduced-config overrides
    for k, v in MINI_CONFIG.items():
        setattr(mod, k, v)
    mod.LAMBDA_FUZZY = float(lam)
    mod.MU_LINK = 0.10

    # Sanity print
    print(f'  After patch: LAMBDA_FUZZY={mod.LAMBDA_FUZZY}, '
          f'MAX_ROWS={mod.MAX_ROWS}, '
          f'INDEPENDENT_EPISODES={mod.INDEPENDENT_EPISODES}, '
          f'FL_ROUNDS={mod.FL_ROUNDS}, '
          f'LOCAL_EPISODES_PER_ROUND={mod.LOCAL_EPISODES_PER_ROUND}')
    print(f'  RESULTS_BASE = {mod.RESULTS_BASE}')

    mod.train_dataset(dataset)
    print(f'  DONE lambda={lam:.2f}')


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ciciot2023',
                   choices=['ciciot2023', 'diad2024'])
    p.add_argument('--lambdas', nargs='+', type=float, default=None,
                   help='Lambda values to run. Defaults to LAMBDAS.')
    args = p.parse_args()

    lambdas = args.lambdas if args.lambdas is not None else LAMBDAS
    for lam in lambdas:
        run_one(lam, args.dataset)

    print()
    print('=' * 70)
    print(f'  MINI SWEEP COMPLETE  |  {len(lambdas)} runs on {args.dataset}')
    print('=' * 70)


if __name__ == '__main__':
    main()
