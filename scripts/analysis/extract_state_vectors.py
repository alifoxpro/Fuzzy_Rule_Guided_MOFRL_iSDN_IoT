#!/usr/bin/env python
"""
Extract real state vectors (28D) from each link and save as CSV.
Uses cached Mamba outputs — no retraining needed.
"""
import os, sys, json, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# ── Constants ──
SEED = 42
WINDOW_SIZE = 20
STRIDE = 20
STATE_DIM = 28
N_SAMPLES = None  # None = all windows
MAX_SUBSAMPLE = 3_000_000

LINK_NAMES = [
    'Link_1_Core', 'Link_2_Aggregation_L', 'Link_3_Aggregation_R',
    'Link_4_Edge_L', 'Link_5_Edge_R'
]
LINK_SHORT = ['L1_Core', 'L2_AggrL', 'L3_AggrR', 'L4_EdgeL', 'L5_EdgeR']

STATE_COLUMNS = [
    'traffic_load', 'latency_estimate', 'entropy', 'packet_rate',
    'attack_detected', 'severity_pred', 'mamba_confidence', 'severity_ratio',
    'mamba_pca_0', 'mamba_pca_1', 'mamba_pca_2', 'mamba_pca_3',
    'mamba_pca_4', 'mamba_pca_5', 'mamba_pca_6', 'mamba_pca_7',
    'mamba_pca_8', 'mamba_pca_9', 'mamba_pca_10', 'mamba_pca_11',
    'mamba_pca_12', 'mamba_pca_13', 'mamba_pca_14', 'mamba_pca_15',
    'neighbor_left_severity', 'neighbor_right_severity',
    'neighbor_alert_count', 'neighbor_avg_severity'
]

SEVERITY_MAP = {0: 'High', 1: 'Medium', 2: 'Low'}
ACTION_MAP = {0: 'ALLOW', 1: 'RATE_LIMIT', 2: 'BLOCK'}
OPTIMAL_MAP = {0: 2, 1: 1, 2: 0}  # High→BLOCK, Med→LIMIT, Low→ALLOW

BASE = Path(__file__).resolve().parent.parent.parent

DS_CONFIG = {
    'ciciot2023': {
        'data_dir': BASE / 'data' / 'unified_14features_5links' / 'ciciot2023',
        'json_file': BASE / 'data' / 'unified_selected_14.json',
        'mamba_dir': BASE / 'experiments' / 'results_StrategyA' / 'ciciot2023' / 'mamba_cache',
    },
    'diad2024': {
        'data_dir': BASE / 'data' / 'unified_14features_5links' / 'diad2024',
        'json_file': BASE / 'data' / 'unified_selected_14.json',
        'mamba_dir': BASE / 'experiments' / 'results_StrategyA' / 'diad2024' / 'mamba_cache',
    }
}


def load_link_csv(csv_path, feature_order, max_rows=MAX_SUBSAMPLE):
    """Load CSV with chunked reading for large files."""
    chunks = []
    total = 0
    for chunk in pd.read_csv(csv_path, chunksize=500_000):
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_rows:
            break
    df = pd.concat(chunks, ignore_index=True)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED).reset_index(drop=True)

    available = [f for f in feature_order if f in df.columns]
    X = df[available].values.astype(np.float32)

    if 'label' in df.columns:
        y = df['label'].values
    elif 'Label' in df.columns:
        y = df['Label'].values
    else:
        y = np.zeros(len(df), dtype=int)
    return X, y, available


def create_windows(X, y, window_size, stride):
    n = len(X)
    n_windows = (n - window_size) // stride + 1
    X_w = np.zeros((n_windows, window_size, X.shape[1]), dtype=np.float32)
    y_w = np.zeros(n_windows, dtype=np.int64)
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        X_w[i] = X[start:end]
        counts = Counter(y[start:end])
        y_w[i] = counts.most_common(1)[0][0]
    return X_w, y_w


def compute_link_metrics(X_windows, feature_order):
    """Pre-compute traffic_load, latency, packet_rate from features.
    Uses same mapping as train_rl_multiagent_StrategyB.py:
      traffic_load = flow_packets_per_s (normalized)
      latency_est  = flow_duration (normalized)
      packet_rate  = avg_pkt_size (normalized)
    """
    X_mean = X_windows.mean(axis=1)  # (N, F)
    n = len(X_mean)

    def _find_idx(name):
        try:
            return feature_order.index(name)
        except ValueError:
            return -1

    def _normalize(col):
        return (col - col.min()) / (col.max() - col.min() + 1e-8)

    # traffic_load: flow_packets_per_s
    idx = _find_idx('flow_packets_per_s')
    if idx >= 0 and idx < X_mean.shape[1]:
        traffic_load = _normalize(X_mean[:, idx])
    else:
        traffic_load = np.zeros(n, dtype=np.float32)

    # latency_estimate: flow_duration
    idx = _find_idx('flow_duration')
    if idx >= 0 and idx < X_mean.shape[1]:
        latency_est = _normalize(X_mean[:, idx])
    else:
        latency_est = np.zeros(n, dtype=np.float32)

    # packet_rate: avg_pkt_size
    idx = _find_idx('avg_pkt_size')
    if idx >= 0 and idx < X_mean.shape[1]:
        packet_rate = _normalize(X_mean[:, idx])
    else:
        packet_rate = np.zeros(n, dtype=np.float32)

    return traffic_load, latency_est, packet_rate


def compute_entropy(recent_severities):
    if len(recent_severities) < 2:
        return 0.5
    counts = Counter(recent_severities)
    total = len(recent_severities)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy / math.log2(3)


def severity_ratio(recent_severities):
    if len(recent_severities) == 0:
        return 0.0
    return sum(1 for s in recent_severities if s == 0) / len(recent_severities)


def extract_dataset(ds_key):
    cfg = DS_CONFIG[ds_key]
    data_dir = cfg['data_dir']
    mamba_dir = cfg['mamba_dir']

    # Load feature order
    with open(cfg['json_file'], 'r') as f:
        feature_data = json.load(f)
        if isinstance(feature_data, dict):
            feature_order = feature_data.get('selected_features', list(feature_data.keys()))
        else:
            feature_order = feature_data
    print(f"\n{'='*60}")
    print(f"  Dataset: {ds_key.upper()}")
    print(f"  Features ({len(feature_order)}): {feature_order}")
    print(f"{'='*60}")

    output_dir = BASE / 'experiments' / f'state_vectors_{ds_key}'
    os.makedirs(output_dir, exist_ok=True)

    # Load all links first (for neighbor computation)
    all_mamba_softmax = []
    all_mamba_embed = []
    all_traffic = []
    all_latency = []
    all_pktrate = []
    all_y = []
    all_n_windows = []

    from sklearn.decomposition import PCA

    for i, link_name in enumerate(LINK_NAMES):
        print(f"\n  [{i+1}/5] {link_name}")
        csv_path = data_dir / f'{link_name}.csv'
        X, y, available = load_link_csv(csv_path, feature_order)
        print(f"    Rows: {len(X):,}, Features: {len(available)}")

        X_w, y_w = create_windows(X, y, WINDOW_SIZE, STRIDE)
        print(f"    Windows: {len(X_w):,}")

        # Load Mamba cache
        softmax_path = mamba_dir / f'{link_name}_softmax.npy'
        embed_path = mamba_dir / f'{link_name}_embed.npy'

        if softmax_path.exists() and embed_path.exists():
            softmax = np.load(softmax_path)
            embed_raw = np.load(embed_path)
            print(f"    Mamba cache loaded: softmax {softmax.shape}, embed {embed_raw.shape}")
        else:
            print(f"    WARNING: Mamba cache not found, using dummy")
            softmax = np.zeros((len(y_w), 3), dtype=np.float32)
            softmax[:, 2] = 1.0  # default Low
            embed_raw = np.zeros((len(y_w), 128), dtype=np.float32)

        # PCA
        pca = PCA(n_components=16, random_state=SEED)
        embed_pca = pca.fit_transform(embed_raw)
        print(f"    PCA: {embed_raw.shape[1]}D → 16D (explained: {pca.explained_variance_ratio_.sum():.3f})")

        # Link metrics
        traffic_load, latency_est, packet_rate = compute_link_metrics(X_w, available)

        all_mamba_softmax.append(softmax)
        all_mamba_embed.append(embed_pca)
        all_traffic.append(traffic_load)
        all_latency.append(latency_est)
        all_pktrate.append(packet_rate)
        all_y.append(y_w)
        all_n_windows.append(len(y_w))

        del X, y, X_w, embed_raw

    # Now extract state vectors with neighbor info
    np.random.seed(SEED)

    for i, link_name in enumerate(LINK_NAMES):
        n_w = all_n_windows[i]
        if N_SAMPLES is None:
            sample_idx = np.arange(n_w)
        else:
            sample_idx = np.random.choice(n_w, size=min(N_SAMPLES, n_w), replace=False)
            sample_idx.sort()

        softmax = all_mamba_softmax[i]
        embed_pca = all_mamba_embed[i]
        traffic = all_traffic[i]
        latency = all_latency[i]
        pktrate = all_pktrate[i]
        y_w = all_y[i]

        rows = []
        recent_sevs = []

        for step, idx in enumerate(sample_idx):
            state = np.zeros(STATE_DIM, dtype=np.float32)

            # [0-3] Link metrics
            state[0] = traffic[idx]
            state[1] = latency[idx]
            state[2] = compute_entropy(recent_sevs)
            state[3] = pktrate[idx]

            # [4-7] Mamba detection
            sm = softmax[idx]
            pred = sm.argmax()
            state[4] = float(pred != 2)          # attack_detected
            state[5] = float(pred) / 2.0          # severity_pred normalized
            state[6] = float(sm.max())             # mamba_confidence
            state[7] = severity_ratio(recent_sevs) # severity_ratio

            # [8-23] Mamba PCA embedding
            state[8:24] = embed_pca[idx]

            # [24-27] Neighbor info
            # Left neighbor
            if i > 0:
                n_left = all_n_windows[i-1]
                left_idx = min(idx, n_left - 1)
                left_pred = all_mamba_softmax[i-1][left_idx].argmax()
                state[24] = float(left_pred) / 2.0
            # Right neighbor
            if i < 4:
                n_right = all_n_windows[i+1]
                right_idx = min(idx, n_right - 1)
                right_pred = all_mamba_softmax[i+1][right_idx].argmax()
                state[25] = float(right_pred) / 2.0

            # Alert count from all neighbors
            alert = 0
            sev_sum = 0
            n_neighbors = 0
            for j in range(5):
                if j != i:
                    n_j = all_n_windows[j]
                    j_idx = min(idx, n_j - 1)
                    j_pred = all_mamba_softmax[j][j_idx].argmax()
                    if j_pred == 0:  # High
                        alert += 1
                    sev_sum += j_pred
                    n_neighbors += 1
            state[26] = alert / max(n_neighbors, 1)
            state[27] = sev_sum / (max(n_neighbors, 1) * 2.0)

            recent_sevs.append(pred)
            if len(recent_sevs) > 20:
                recent_sevs = recent_sevs[-20:]

            # Extra info columns
            true_sev = int(y_w[idx])
            optimal_action = OPTIMAL_MAP[true_sev]

            row = list(state)
            row.extend([
                idx,                            # window_index
                true_sev,                       # true_severity (0=H, 1=M, 2=L)
                SEVERITY_MAP[true_sev],         # true_severity_name
                ACTION_MAP[optimal_action],     # optimal_action
                float(sm[0]),                   # P(High)
                float(sm[1]),                   # P(Medium)
                float(sm[2]),                   # P(Low)
            ])
            rows.append(row)

        # Create DataFrame
        extra_cols = [
            'window_index', 'true_severity', 'true_severity_name',
            'optimal_action', 'P_High', 'P_Medium', 'P_Low'
        ]
        columns = STATE_COLUMNS + extra_cols
        df = pd.DataFrame(rows, columns=columns)

        csv_out = output_dir / f'{LINK_SHORT[i]}_state_vectors.csv'
        df.to_csv(csv_out, index=False, float_format='%.6f')
        print(f"\n  Saved: {csv_out}")
        print(f"    Samples: {len(df)}")
        print(f"    Columns: {len(df.columns)}")
        print(f"    Severity dist: H={sum(df.true_severity==0)}, M={sum(df.true_severity==1)}, L={sum(df.true_severity==2)}")

        # Print first 3 rows as sample
        print(f"\n    Sample (first 3 rows):")
        print(f"    {'─'*80}")
        for r in range(min(3, len(df))):
            row = df.iloc[r]
            print(f"    Window {int(row.window_index):>6d} | "
                  f"Severity: {row.true_severity_name:<6s} | "
                  f"Optimal: {row.optimal_action:<10s} | "
                  f"traffic={row.traffic_load:.3f} latency={row.latency_estimate:.3f} "
                  f"attack={int(row.attack_detected)} conf={row.mamba_confidence:.3f} "
                  f"P(H)={row.P_High:.3f} P(M)={row.P_Medium:.3f} P(L)={row.P_Low:.3f}")

    # Summary CSV: all links combined
    all_dfs = []
    for i in range(5):
        csv_path = output_dir / f'{LINK_SHORT[i]}_state_vectors.csv'
        df = pd.read_csv(csv_path)
        df.insert(0, 'link', LINK_SHORT[i])
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_dir / 'all_links_state_vectors.csv', index=False, float_format='%.6f')
    print(f"\n  Combined CSV: {output_dir / 'all_links_state_vectors.csv'}")
    print(f"    Total rows: {len(combined)}")
    print(f"\n  Done! Files in: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ciciot2023', choices=['ciciot2023', 'diad2024', 'all'])
    args = parser.parse_args()

    if args.dataset == 'all':
        for ds in ['ciciot2023', 'diad2024']:
            extract_dataset(ds)
    else:
        extract_dataset(args.dataset)
