# -*- coding: utf-8 -*-
"""
============================================================================
Smart lambda-sweep: only re-runs Phase B4 (Fuzzy-Augmented).

Because Phase B4 is the only phase in which R_total depends on
LAMBDA_FUZZY, the other phases (B0 rule-mining, B1 independent MOSAR,
B2 FedAvg, B3 FedProx) give the same numerical result for every
lambda value we test. This script exploits that fact by:

  1. Copying the canonical results_StrategyB/{dataset}/ contents into
     a fresh results_StrategyB_mini_lamX.XX/{dataset}/ folder.
  2. Removing the B4 entry from the progress file so the script
     re-runs B4 (and only B4) from the B3 checkpoints.
  3. Monkey-patching FL_ROUNDS, LOCAL_EPISODES_PER_ROUND, MAX_ROWS,
     and LAMBDA_FUZZY before calling train_dataset(dataset_key).
  4. The data-loading step still happens (to instantiate envs) but
     subsamples to MAX_ROWS rows per link, which is fast.

Wall-clock target on CPU: a few minutes per lambda value.

Usage:
    python scripts/ablation/run_b4_only_sweep.py [--dataset ciciot2023]
                                                 [--lambdas 0.00 0.10 0.50]
============================================================================
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
SCRIPT = PROJECT / 'scripts' / 'training' / 'train_rl_multiagent_StrategyB.py'
CANONICAL = PROJECT / 'experiments' / 'results_StrategyB'

# Reduced training config for the B4-only re-run. These only affect B4
# because B0-B3 are loaded from the canonical progress file.
MINI_CONFIG = {
    'MAX_ROWS': 30_000,        # faster data loading
    'FL_ROUNDS': 3,            # 3 FL rounds for B4 (canonical uses 30)
    'LOCAL_EPISODES_PER_ROUND': 2,
}


def is_run_complete(dst, dataset):
    """Return True if a (lambda, dataset) run has already finished.
    A run is complete iff its progress_state.json marks B4 as
    completed and a results.json file with a 'phase_b4' entry
    exists."""
    progress_file = dst / 'progress_state.json'
    results_file = dst / 'results.json'
    if not progress_file.exists() or not results_file.exists():
        return False
    try:
        with open(progress_file) as f:
            progress = json.load(f)
        if f'{dataset}::phase_b4' not in progress.get('completed', []):
            return False
        with open(results_file) as f:
            results = json.load(f)
        return 'phase_b4' in results and 'mean_msr' in results['phase_b4']
    except Exception:
        return False


def copy_canonical_to_mini(lam, dataset, target_windows, force=False):
    """Prepare the results_StrategyB_mini_lamX.XX/{dataset}/ folder for
    a B4-only re-run.

    Behaviour:
      - If the destination folder does not exist, copy the canonical
        B0-B3 artefacts and truncate the mamba cache.
      - If the destination folder exists but B4 is not yet completed,
        LEAVE IT ALONE so the training script can resume. This covers
        the case of a crashed run mid-B4.
      - If the destination folder exists and B4 IS completed, and
        force=False, skip the run entirely.
      - If force=True, wipe the destination and rebuild from scratch.

    Returns one of: 'built', 'resume', 'skip'.
    """
    suffix = f'_mini_lam{lam:.2f}'
    dst = PROJECT / 'experiments' / f'results_StrategyB{suffix}' / dataset
    src = CANONICAL / dataset

    if not src.exists():
        raise RuntimeError(f'Canonical source does not exist: {src}')

    if dst.exists():
        if is_run_complete(dst, dataset) and not force:
            print(f'  SKIP: {dst.relative_to(PROJECT)} already complete')
            return 'skip'
        if force:
            print(f'  FORCE: wiping {dst.relative_to(PROJECT)}')
            shutil.rmtree(dst)
        else:
            print(f'  RESUME: {dst.relative_to(PROJECT)} exists, '
                  f'keeping existing state')
            return 'resume'

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f'  BUILD: copying canonical -> {dst.relative_to(PROJECT)}')

    # We need the mamba cache, master agent results, and the B3
    # checkpoints. We do NOT need the fuzzy_retrain folder because
    # B4 will be re-computed.
    keep = ['mosar_independent', 'mosar_fedavg', 'mosar_fedprox',
            'master_agent', 'afras_fl',
            'progress_state.json', 'results.json']
    dst.mkdir(parents=True, exist_ok=True)
    for item in keep:
        s = src / item
        d = dst / item
        if s.is_dir():
            shutil.copytree(s, d)
        elif s.is_file():
            shutil.copy2(s, d)

    # Truncate the mamba cache to the expected size
    cache_src = src / 'mamba_cache'
    cache_dst = dst / 'mamba_cache'
    cache_dst.mkdir(exist_ok=True)
    if cache_src.exists():
        print(f'  Truncating mamba cache to {target_windows} windows/link')
        for f in sorted(cache_src.iterdir()):
            if f.suffix != '.npy':
                continue
            arr = np.load(f)
            if arr.shape[0] > target_windows:
                arr = arr[:target_windows]
            np.save(cache_dst / f.name, arr)
            print(f'    {f.name}: {arr.shape}')

    # Drop the B4/B5 entries so the training script re-runs them
    progress_file = dst / 'progress_state.json'
    with open(progress_file) as f:
        progress = json.load(f)
    progress['completed'] = [k for k in progress['completed']
                             if not (k.endswith('::phase_b4')
                                     or k.endswith('::phase_b5'))]
    progress['results'] = {k: v for k, v in progress['results'].items()
                           if not (k.endswith('::phase_b4')
                                   or k.endswith('::phase_b5'))}
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    print(f'  Progress file reset: B4 and B5 removed')
    return 'built'


def import_strategy_b_fresh():
    """Re-import the strategy B module so that RESULTS_BASE picks up
    the current RESULTS_SUFFIX env var."""
    for key in list(sys.modules):
        if key.startswith('strategy_b'):
            del sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        'strategy_b', str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_one(lam, dataset, force=False):
    suffix = f'_mini_lam{lam:.2f}'

    print('=' * 72)
    print(f'  B4-only run  |  lambda = {lam:.2f}  |  dataset = {dataset}')
    print('=' * 72)

    # Expected windows-per-link: MAX_ROWS / STRIDE. STRIDE == 20 in
    # the current config, so 30,000 / 20 = 1500. We use that as the
    # truncation length for the mamba cache arrays.
    target_windows = MINI_CONFIG['MAX_ROWS'] // 20

    # Step 1: prepare / resume / skip the run folder
    status = copy_canonical_to_mini(lam, dataset, target_windows, force=force)
    if status == 'skip':
        print(f'  DONE lambda={lam:.2f} (already complete, nothing to do)')
        return True

    # Step 2: set env vars BEFORE importing the module
    os.environ['RESULTS_SUFFIX'] = suffix
    os.environ['LAMBDA_FUZZY'] = f'{lam:.2f}'
    os.environ['MU_LINK'] = '0.10'

    # Step 3: fresh import
    mod = import_strategy_b_fresh()

    # Step 4: apply reduced-config overrides
    for k, v in MINI_CONFIG.items():
        setattr(mod, k, v)
    mod.LAMBDA_FUZZY = float(lam)

    print(f'  Patched: LAMBDA_FUZZY={mod.LAMBDA_FUZZY}  '
          f'MAX_ROWS={mod.MAX_ROWS}  '
          f'FL_ROUNDS={mod.FL_ROUNDS}  '
          f'LOCAL_EPISODES_PER_ROUND={mod.LOCAL_EPISODES_PER_ROUND}')
    print(f'  RESULTS_BASE = {mod.RESULTS_BASE}')

    # Step 5: run train_dataset (it will skip B0-B3 and only run B4/B5)
    try:
        mod.train_dataset(dataset)
        print(f'  DONE lambda={lam:.2f}')
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'  FAILED lambda={lam:.2f}: {e}')
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ciciot2023',
                   choices=['ciciot2023', 'diad2024'])
    p.add_argument('--lambdas', nargs='+', type=float,
                   default=[0.00, 0.10, 0.50])
    p.add_argument('--force', action='store_true',
                   help='Wipe and rebuild even if a completed run '
                        'already exists. Default is to resume incomplete '
                        'runs and skip completed ones.')
    args = p.parse_args()

    successes = []
    failures = []
    for lam in args.lambdas:
        if run_one(lam, args.dataset, force=args.force):
            successes.append(lam)
        else:
            failures.append(lam)

    print()
    print('=' * 72)
    print(f'  MINI SWEEP SUMMARY')
    print(f'  success: {successes}')
    print(f'  failed : {failures}')
    print('=' * 72)


if __name__ == '__main__':
    main()
