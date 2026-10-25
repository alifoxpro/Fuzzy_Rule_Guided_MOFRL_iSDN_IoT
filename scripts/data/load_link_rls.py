# -*- coding: utf-8 -*-
"""
============================================================================
Record-Level Split (RLS) loader — drop-in replacement for load_link_csv
============================================================================
Loads per-link features from the leakage-free splits produced by
record_level_split.py. The returned (X, y) is:
  - already stratified into TRAIN / VAL / TEST at the RECORD level,
  - already scaled by a StandardScaler fitted on TRAIN ONLY,
  - restricted to the caller-supplied feature_order (via feature_cols
    stored in the split manifest).

Usage from a training script:

    from load_link_rls import load_link_rls
    X, y = load_link_rls(dataset='ciciot2023',
                         link_name='Link_1_Core',
                         feature_order=feature_order,
                         split='train',
                         rls_root=DATA_DIR / 'unified_14features_5links_rls')

X is (N, len(feature_order)) float32, y is (N,) int64. When
feature_order is a subset or a re-ordering of the split's feature_cols,
this function slices/reorders the scaled columns accordingly so the
downstream pipeline sees exactly the features it asked for.
============================================================================
"""

from pathlib import Path
from typing import Sequence, Tuple, Union

import json
import numpy as np

VALID_SPLITS = ('train', 'val', 'test')


def load_link_rls(dataset: str,
                  link_name: str,
                  feature_order: Sequence[str],
                  split: str,
                  rls_root: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load one link's split-scaled arrays and return (X, y).

    Parameters
    ----------
    dataset : 'ciciot2023' | 'diad2024' | ...
    link_name : e.g. 'Link_1_Core'
    feature_order : list of feature names the caller wants, in order.
                    Must be a subset of the split's feature_cols.
    split : 'train' | 'val' | 'test'
    rls_root : root folder that contains {dataset}_5links/ sub-folders
               produced by record_level_split.py.

    Returns
    -------
    X : (N, len(feature_order)) float32
    y : (N,) int64
    """
    if split not in VALID_SPLITS:
        raise ValueError(f'split must be one of {VALID_SPLITS}, got {split!r}')

    rls_root = Path(rls_root)
    ds_dir = rls_root / f'{dataset}_5links'
    manifest_file = ds_dir / 'split_manifest.json'
    if not manifest_file.exists():
        raise FileNotFoundError(
            f'RLS manifest not found: {manifest_file}. '
            f'Run scripts/data/record_level_split.py first.')

    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    if link_name not in manifest['links']:
        raise KeyError(f'{link_name} missing from manifest at {manifest_file}')

    stored_feature_cols = manifest['links'][link_name]['feature_cols']

    # Verify every requested feature is present
    missing = [c for c in feature_order if c not in stored_feature_cols]
    if missing:
        raise KeyError(
            f'{link_name}: features requested by caller but not present in '
            f'the RLS split: {missing}. Present in split: {stored_feature_cols}')

    npz_path = ds_dir / f'{link_name}_{split}.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f'RLS split file not found: {npz_path}')

    data = np.load(npz_path)
    X_full = data['X']
    y = data['y']

    # Column-select and reorder to match feature_order
    col_idx = np.array([stored_feature_cols.index(c) for c in feature_order],
                       dtype=np.int64)
    X = X_full[:, col_idx].astype(np.float32, copy=True)
    # y is already int64 in the NPZ, but be defensive
    y = np.asarray(y, dtype=np.int64)

    return X, y


def load_link_rls_stats(dataset: str,
                        link_name: str,
                        rls_root: Union[str, Path]) -> dict:
    """Return the manifest entry for a single link (row counts, scaler stats)."""
    rls_root = Path(rls_root)
    with open(rls_root / f'{dataset}_5links' / 'split_manifest.json') as f:
        manifest = json.load(f)
    return manifest['links'][link_name]
