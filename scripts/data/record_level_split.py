# -*- coding: utf-8 -*-
"""
============================================================================
Record-Level Train / Val / Test Split with Leakage-Free Preprocessing
============================================================================
Addresses reviewer comment R2-6 (IoT-J IoT-72156-2026): the flow-level
preprocessing in unify_features.py + train_rl_multiagent_StrategyB.py
fits the StandardScaler on the concatenated data BEFORE splitting into
train/val/test, which is a textbook source of data leakage.

This script:
  1. Loads each per-link unified CSV.
  2. Splits records into TRAIN / VAL / TEST at the RECORD level, stratified
     by severity, with a fixed seed.
  3. Fits a StandardScaler on TRAIN ONLY, then applies its transform to
     VAL and TEST.
  4. Writes six files per link:
        Link_X_train.csv  Link_X_val.csv  Link_X_test.csv
        Link_X_train.npz  Link_X_val.npz  Link_X_test.npz
     Plus one scaler:
        scaler_Link_X.pkl
  5. Writes a JSON manifest that pins the exact split sizes, seed,
     stratification keys, and per-feature scaler statistics for
     reproducibility.

Output tree:
    data/unified_processed_rls/{dataset}_5links/
        Link_X_{train,val,test}.{csv,npz}
        scaler_Link_X.pkl
        split_manifest.json

Usage:
    python scripts/data/record_level_split.py
    python scripts/data/record_level_split.py --dataset ciciot2023
    python scripts/data/record_level_split.py --seed 123
============================================================================
"""

import argparse
import gc
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Paths and constants
# ============================================================================
PROJECT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT / 'data' / 'unified_processed'
OUTPUT_DIR = PROJECT / 'data' / 'unified_processed_rls'

DATASETS = ['ciciot2023', 'diad2024']

# When --input-layout=14feat, the per-dataset folder is
# <input>/{dataset}/Link_X.csv (no "_5links" suffix), matching
# data/unified_14features_5links/{dataset}/. Otherwise the layout is
# <input>/{dataset}_5links/Link_X.csv, matching unified_processed/.

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

DEFAULT_SEED = 42
CHUNK_SIZE = 500_000

SEV_MAP = {'High': 0, 'Medium': 1, 'Low': 2}
FEATURE_COLS_EXCLUDED = {'severity', 'link_id'}


def log(msg=''):
    print(msg, flush=True)


# ============================================================================
# Core split logic (one link)
# ============================================================================
def split_one_link(csv_path: Path, out_dir: Path, seed: int) -> dict:
    """Load one link CSV, stratified split, fit scaler on TRAIN, dump splits."""
    link_name = csv_path.stem
    log(f'    [{link_name}]')

    # ---- Load full CSV in chunks (memory-safe on 6M-row files) --------------
    chunks = []
    total = 0
    for ch in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=True):
        chunks.append(ch)
        total += len(ch)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    feature_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDED]
    n_feat = len(feature_cols)

    # ---- Clean numeric features --------------------------------------------
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # ---- Map severity to integer labels ------------------------------------
    df['label'] = df['severity'].map(SEV_MAP).fillna(2).astype(np.int64)

    # ---- Class counts (before split) ---------------------------------------
    class_before = df['label'].value_counts().to_dict()

    # ---- Stratified split: TRAIN vs TEMP (val+test) ------------------------
    df_train, df_temp = train_test_split(
        df,
        train_size=TRAIN_FRAC,
        random_state=seed,
        stratify=df['label'],
        shuffle=True,
    )

    # ---- Stratified split: VAL vs TEST inside TEMP -------------------------
    relative_val = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    df_val, df_test = train_test_split(
        df_temp,
        train_size=relative_val,
        random_state=seed,
        stratify=df_temp['label'],
        shuffle=True,
    )
    del df, df_temp
    gc.collect()

    log(f'      Rows total={total:,} train={len(df_train):,} '
        f'val={len(df_val):,} test={len(df_test):,}')

    # ---- Fit scaler on TRAIN ONLY ------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values.astype(np.float32))
    X_val = scaler.transform(df_val[feature_cols].values.astype(np.float32))
    X_test = scaler.transform(df_test[feature_cols].values.astype(np.float32))

    y_train = df_train['label'].values.astype(np.int64)
    y_val = df_val['label'].values.astype(np.int64)
    y_test = df_test['label'].values.astype(np.int64)

    # ---- Save CSVs (record-level splits, unscaled features + label) --------
    #     CSVs keep raw features so downstream code can re-fit its own
    #     scaler if it wants to; NPZs hold the scaled arrays used by the
    #     training pipeline directly.
    for split_name, df_split in [
        ('train', df_train), ('val', df_val), ('test', df_test)]:
        out_csv = out_dir / f'{link_name}_{split_name}.csv'
        df_split.drop(columns=['label']).to_csv(out_csv, index=False)

    # ---- Save NPZs (pre-scaled arrays for the training pipeline) ----------
    for split_name, X, y in [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test)]:
        out_npz = out_dir / f'{link_name}_{split_name}.npz'
        np.savez_compressed(out_npz, X=X, y=y)

    # ---- Save scaler --------------------------------------------------------
    out_pkl = out_dir / f'scaler_{link_name}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'feature_cols': feature_cols,
            'seed': seed,
        }, f)

    # ---- Per-link manifest entry -------------------------------------------
    return {
        'link_name': link_name,
        'total_rows': int(total),
        'n_features': n_feat,
        'feature_cols': feature_cols,
        'class_counts_before_split': {int(k): int(v) for k, v in class_before.items()},
        'train_size': int(len(df_train)),
        'val_size': int(len(df_val)),
        'test_size': int(len(df_test)),
        'train_class_counts': {int(k): int(v) for k, v in
                               pd.Series(y_train).value_counts().to_dict().items()},
        'val_class_counts': {int(k): int(v) for k, v in
                             pd.Series(y_val).value_counts().to_dict().items()},
        'test_class_counts': {int(k): int(v) for k, v in
                              pd.Series(y_test).value_counts().to_dict().items()},
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
    }


# ============================================================================
# Driver
# ============================================================================
def process_dataset(dataset: str, seed: int,
                    input_root: Path = INPUT_DIR,
                    output_root: Path = OUTPUT_DIR,
                    layout: str = 'unified_processed') -> dict:
    if layout == '14feat':
        src_dir = input_root / dataset
    else:
        src_dir = input_root / f'{dataset}_5links'
    out_dir = output_root / f'{dataset}_5links'
    os.makedirs(out_dir, exist_ok=True)

    link_files = sorted(src_dir.glob('Link_*.csv'))
    log(f'\n  [{dataset.upper()}] {len(link_files)} links -> {out_dir}')

    ds_manifest = {'dataset': dataset,
                   'seed': seed,
                   'train_frac': TRAIN_FRAC,
                   'val_frac': VAL_FRAC,
                   'test_frac': TEST_FRAC,
                   'links': {}}

    for lf in link_files:
        entry = split_one_link(lf, out_dir, seed)
        ds_manifest['links'][entry['link_name']] = entry

    with open(out_dir / 'split_manifest.json', 'w') as f:
        json.dump(ds_manifest, f, indent=2)
    log(f'  Manifest -> {out_dir / "split_manifest.json"}')
    return ds_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=DATASETS + ['all'], default='all')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--input-root', default=str(INPUT_DIR),
                    help='Root folder containing the per-dataset link CSVs.')
    ap.add_argument('--output-root', default=str(OUTPUT_DIR),
                    help='Root folder to write the split artefacts into.')
    ap.add_argument('--layout', choices=['unified_processed', '14feat'],
                    default='unified_processed',
                    help='Directory layout of --input-root: unified_processed = '
                         '<root>/{ds}_5links/Link_X.csv; 14feat = '
                         '<root>/{ds}/Link_X.csv.')
    args = ap.parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    t0 = time.time()
    log('=' * 72)
    log('  Record-Level Split (leakage-free preprocessing)')
    log(f'  Splits: train={TRAIN_FRAC:.0%}  val={VAL_FRAC:.0%}  test={TEST_FRAC:.0%}')
    log(f'  Seed:   {args.seed}')
    log('=' * 72)

    targets = DATASETS if args.dataset == 'all' else [args.dataset]
    all_manifests = {}
    for ds in targets:
        all_manifests[ds] = process_dataset(
            ds, args.seed,
            input_root=input_root,
            output_root=output_root,
            layout=args.layout,
        )

    log(f'\n  Total time: {time.time() - t0:.1f}s')
    log('=' * 72)
    log('  Record-Level Split Complete')
    log('=' * 72)


if __name__ == '__main__':
    main()
