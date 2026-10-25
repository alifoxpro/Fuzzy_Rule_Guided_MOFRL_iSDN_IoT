# -*- coding: utf-8 -*-
"""Re-run FL training for an additional A5 seed. Keeps SEED=42 (identical data
subsample / windows / Mamba cache -> cache reused, no recompute) but offsets
seed_everything so DQN init, exploration and episode sampling differ, yielding
an independent training trajectory for multi-seed A5.

Usage: python _run_a5_seed.py <dataset> <effective_seed>
       (effective_seed must differ from 42; offset = effective_seed - 42)
"""
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_rl_multiagent as M

dataset = sys.argv[1]
eff_seed = int(sys.argv[2])
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else M.FL_ROUNDS
offset = eff_seed - M.SEED
assert offset != 0, 'effective_seed must differ from base SEED=42'

# reduced-budget multi-seed A5 supplement
M.FL_ROUNDS = rounds
print(f'FL_ROUNDS set to {M.FL_ROUNDS}', flush=True)

# offset ONLY the training RNG; data subsample (train_test_split random_state=SEED)
# and PCA stay on the constant SEED, so windows and the Mamba cache are identical.
_orig = M.seed_everything
def _patched(s=M.SEED):
    _orig(s + offset)
M.seed_everything = _patched

PROJ = Path(__file__).resolve().parent.parent.parent
src_base = PROJ / 'experiments' / 'results_A5run'
dst_base = PROJ / 'experiments' / f'results_A5run_s{eff_seed}_r{rounds}'
M.RESULTS_BASE = dst_base

# reuse the existing Mamba cache (identical windows) to skip recompute
src_cache = src_base / dataset / 'mamba_cache'
dst_cache = dst_base / dataset / 'mamba_cache'
if src_cache.exists() and not dst_cache.exists():
    dst_cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_cache, dst_cache)
    print(f'copied cache -> {dst_cache}', flush=True)

print(f'A5 seed run | dataset={dataset} effective_seed={eff_seed} '
      f'offset={offset} | RESULTS_BASE={dst_base}', flush=True)
M.train_dataset(dataset)
print('A5_SEED_RUN_DONE', flush=True)
