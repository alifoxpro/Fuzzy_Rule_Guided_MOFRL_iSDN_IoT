# -*- coding: utf-8 -*-
"""
============================================================================
FL-RLMAMS Stage 3: Unified Feature Namespace
============================================================================
Reads processed 5-link CSVs from both CICIoT2023 and DIAD2024, renames
columns to a shared namespace (~20 semantically matched features), and
writes unified CSVs for downstream XAI feature selection.

Source: data/processed/{dataset}_5links/
Output: data/unified_processed/{dataset}_5links/
============================================================================
"""

import os, sys, io, json, time
from pathlib import Path

import pandas as pd
import numpy as np

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except AttributeError:
    pass

# ============================================================================
# Paths
# ============================================================================
PROJECT = Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = Path(os.environ.get('FLRLMAMS_PROCESSED', PROJECT / 'data' / 'processed'))
OUTPUT_DIR = PROJECT / 'data' / 'unified_processed'
REPORT_FILE = OUTPUT_DIR / 'unification_report.json'

DATASETS = {
    'ciciot2023': SOURCE_DIR / 'ciciot2023_5links',
    'diad2024':   SOURCE_DIR / 'diad2024_5links',
}

CHUNK_SIZE = 100_000

# ============================================================================
# Feature Mapping: unified_name -> {ciciot2023: col, diad2024: col}
# ============================================================================
FEATURE_MAP = {
    'flow_duration':      {'ciciot2023': 'flow_duration',      'diad2024': 'Flow Duration'},
    'protocol_type':      {'ciciot2023': 'Protocol Type',      'diad2024': 'Protocol'},
    'flow_packets_per_s': {'ciciot2023': 'Rate',               'diad2024': 'Flow Packets/s'},
    'fwd_packets_per_s':  {'ciciot2023': 'Srate',              'diad2024': 'Fwd Packets/s'},
    'bwd_packets_per_s':  {'ciciot2023': 'Drate',              'diad2024': 'Bwd Packets/s'},
    'fin_flag_count':     {'ciciot2023': 'fin_flag_number',    'diad2024': 'FIN Flag Count'},
    'syn_flag_count':     {'ciciot2023': 'syn_flag_number',    'diad2024': 'SYN Flag Count'},
    'rst_flag_count':     {'ciciot2023': 'rst_flag_number',    'diad2024': 'RST Flag Count'},
    'psh_flag_count':     {'ciciot2023': 'psh_flag_number',    'diad2024': 'PSH Flag Count'},
    'ack_flag_count':     {'ciciot2023': 'ack_flag_number',    'diad2024': 'ACK Flag Count'},
    'ece_flag_count':     {'ciciot2023': 'ece_flag_number',    'diad2024': 'ECE Flag Count'},
    'cwr_flag_count':     {'ciciot2023': 'cwr_flag_number',    'diad2024': 'CWR Flag Count'},
    'urg_flag_count':     {'ciciot2023': 'urg_count',          'diad2024': 'URG Flag Count'},
    'flow_iat_mean':      {'ciciot2023': 'IAT',                'diad2024': 'Flow IAT Mean'},
    'pkt_len_min':        {'ciciot2023': 'Min',                'diad2024': 'Packet Length Min'},
    'pkt_len_max':        {'ciciot2023': 'Max',                'diad2024': 'Packet Length Max'},
    'avg_pkt_size':       {'ciciot2023': 'AVG',                'diad2024': 'Average Packet Size'},
    'pkt_len_std':        {'ciciot2023': 'Std',                'diad2024': 'Packet Length Std'},
    'pkt_len_variance':   {'ciciot2023': 'Variance',           'diad2024': 'Packet Length Variance'},
    'fwd_header_length':  {'ciciot2023': 'Header_Length',      'diad2024': 'Fwd Header Length'},
}

UNIFIED_FEATURES = sorted(FEATURE_MAP.keys())
META_COLS = ['severity', 'link_id']
OUTPUT_COLS = UNIFIED_FEATURES + META_COLS


# ============================================================================
# Helpers
# ============================================================================
def log(msg=''):
    print(msg, flush=True)


def process_dataset(dataset_name, input_dir, output_dir):
    """Process all link CSVs for one dataset."""
    link_files = sorted(input_dir.glob('Link_*.csv'))
    log(f'\n  [{dataset_name.upper()}] Found {len(link_files)} link files')

    # Build rename mapping: original_col -> unified_name
    rename_map = {}
    for unified_name, mapping in FEATURE_MAP.items():
        original_col = mapping[dataset_name]
        rename_map[original_col] = unified_name

    os.makedirs(output_dir, exist_ok=True)
    dataset_stats = {}

    for lf in link_files:
        link_name = lf.stem
        log(f'    Processing {link_name}...')
        out_path = output_dir / lf.name

        # Read header to check which columns exist
        header_df = pd.read_csv(lf, nrows=0)
        available_cols = set(header_df.columns)

        # Check which mapped features are available
        missing = []
        present_originals = []
        for unified_name, mapping in FEATURE_MAP.items():
            orig_col = mapping[dataset_name]
            if orig_col in available_cols:
                present_originals.append(orig_col)
            else:
                missing.append(f'{unified_name} (expected: {orig_col})')

        if missing:
            log(f'      WARNING: Missing columns: {missing}')

        # Columns to read: mapped features + metadata
        cols_to_read = present_originals + [c for c in META_COLS if c in available_cols]

        # Stream process
        total_rows = 0
        first_chunk = True
        stats = {'rows': 0, 'columns': [], 'missing_features': missing}

        for chunk in pd.read_csv(lf, usecols=cols_to_read, chunksize=CHUNK_SIZE,
                                 low_memory=False):
            # Rename columns
            chunk = chunk.rename(columns=rename_map)

            # Ensure all unified features exist (fill missing with 0)
            for uf in UNIFIED_FEATURES:
                if uf not in chunk.columns:
                    chunk[uf] = 0.0

            # Reorder columns
            chunk = chunk[OUTPUT_COLS]

            # Clean numeric data
            for col in UNIFIED_FEATURES:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0.0)
                chunk[col] = chunk[col].replace([np.inf, -np.inf], 0.0)

            # Write
            chunk.to_csv(out_path, mode='w' if first_chunk else 'a',
                        header=first_chunk, index=False)
            first_chunk = False
            total_rows += len(chunk)

        stats['rows'] = total_rows
        stats['columns'] = OUTPUT_COLS
        dataset_stats[link_name] = stats
        log(f'      Wrote {total_rows:,} rows -> {out_path.name}')

    return dataset_stats


def validate_outputs():
    """Validate that both datasets have identical column structure."""
    log('\n  [Validation]')
    all_good = True

    for ds_name in ['ciciot2023', 'diad2024']:
        ds_dir = OUTPUT_DIR / f'{ds_name}_5links'
        link_files = sorted(ds_dir.glob('Link_*.csv'))
        for lf in link_files:
            header = pd.read_csv(lf, nrows=0)
            cols = list(header.columns)
            if cols != OUTPUT_COLS:
                log(f'    MISMATCH in {ds_name}/{lf.name}: {cols}')
                all_good = False

    if all_good:
        log('    All files have identical column structure')
    return all_good


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    log('=' * 70)
    log('  FL-RLMAMS Stage 3: Unified Feature Namespace')
    log(f'  Mapping {len(FEATURE_MAP)} shared features across datasets')
    log('=' * 70)

    report = {'feature_map': {k: v for k, v in FEATURE_MAP.items()},
              'unified_features': UNIFIED_FEATURES,
              'datasets': {}}

    for ds_name, input_dir in DATASETS.items():
        output_dir = OUTPUT_DIR / f'{ds_name}_5links'
        stats = process_dataset(ds_name, input_dir, output_dir)
        report['datasets'][ds_name] = stats

    # Validate
    valid = validate_outputs()
    report['validation_passed'] = valid

    # Save report
    os.makedirs(REPORT_FILE.parent, exist_ok=True)
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
    log(f'\n  Report saved: {REPORT_FILE.name}')

    elapsed = time.time() - t0
    log(f'\n  Total time: {elapsed:.1f}s')
    log('=' * 70)
    log('  Unification Complete!')
    log('=' * 70)


if __name__ == '__main__':
    main()
