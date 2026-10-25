# -*- coding: utf-8 -*-
"""
============================================================================
FL-RLMAMS Phase 4: State Vector Analysis & IEEE Charts
============================================================================
Analyzes the 28D state vector components per link and generates
publication-quality IEEE IEEEtran charts.

Usage:
  python analyze_state_vector.py --dataset ciciot2023
  python analyze_state_vector.py --dataset diad2024
  python analyze_state_vector.py --dataset all
============================================================================
"""

import os, json, sys, argparse, gc, math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# ============================================================================
# IEEE Style
# ============================================================================
COL1 = 3.5
COL2 = 7.16

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

# ============================================================================
# Config
# ============================================================================
PROJECT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT / 'data'
RESULTS_BASE = PROJECT / 'experiments' / 'results' / 'phase4_rl'

SEED = 42
WINDOW_SIZE = 20
STRIDE = 20
MAX_ROWS = 3_000_000
SEV_MAP = {'High': 0, 'Medium': 1, 'Low': 2}
SEV_NAMES = ['High', 'Medium', 'Low']
SEV_COLORS = ['#E74C3C', '#F39C12', '#27AE60']

LINK_NAMES = [
    'Link_1_Core', 'Link_2_Aggregation_L', 'Link_3_Aggregation_R',
    'Link_4_Edge_L', 'Link_5_Edge_R'
]
LINK_SHORT = ['L1-Core', 'L2-AggrL', 'L3-AggrR', 'L4-EdgeL', 'L5-EdgeR']
LINK_COLORS = ['#2166AC', '#B2182B', '#4DAF4A', '#984EA3', '#FF7F00']

DATASETS = {
    'ciciot2023': DATA_DIR / 'unified_14features_5links' / 'ciciot2023',
    'diad2024': DATA_DIR / 'unified_14features_5links' / 'diad2024',
}
DS_LABELS = {'ciciot2023': 'CICIoT2023', 'diad2024': 'DIAD2024'}

STATE_COMPONENTS = {
    'Link Metrics': ['traffic_load', 'latency_est', 'entropy', 'packet_rate'],
    'LFA State': ['attack_detected', 'severity_pred', 'confidence', 'severity_ratio'],
    'Mamba Embed': [f'embed_{i}' for i in range(16)],
    'Neighbor Flags': ['left_sev', 'right_sev', 'alert_count', 'coord_signal'],
}

STATE_NAMES_FULL = (STATE_COMPONENTS['Link Metrics'] +
                    STATE_COMPONENTS['LFA State'] +
                    STATE_COMPONENTS['Mamba Embed'] +
                    STATE_COMPONENTS['Neighbor Flags'])


def log(msg=''):
    print(msg)


# ============================================================================
# Build State Vectors from cached Mamba outputs + raw data
# ============================================================================
def build_state_vectors(dataset_key):
    """Build state vectors for all 5 links from cached data."""
    log(f'\n  Building State Vectors for {DS_LABELS[dataset_key]}...')

    with open(DATA_DIR / 'unified_selected_14.json', 'r') as f:
        feat_info = json.load(f)
    feature_order = feat_info['selected_features']

    data_dir = DATASETS[dataset_key]
    cache_dir = RESULTS_BASE / dataset_key / 'mamba_cache'

    all_states = {}
    all_severities = {}
    all_stats = {}

    for li, link_name in enumerate(LINK_NAMES):
        log(f'    [{li+1}/5] {link_name}')

        # Load cached Mamba outputs
        cache_softmax = cache_dir / f'{link_name}_softmax.npy'
        cache_embed = cache_dir / f'{link_name}_embed.npy'

        if not cache_softmax.exists():
            log(f'      WARNING: No cache for {link_name}, skipping')
            continue

        softmax = np.load(cache_softmax)
        embeddings_128 = np.load(cache_embed)

        # PCA 128→16
        pca = PCA(n_components=16, random_state=SEED)
        embed_16 = pca.fit_transform(embeddings_128[:min(50000, len(embeddings_128))])
        if len(embeddings_128) > 50000:
            embed_16 = pca.transform(embeddings_128)
        else:
            embed_16 = pca.transform(embeddings_128)

        # Load raw CSV for link metrics
        csv_path = data_dir / f'{link_name}.csv'
        needed_cols = list(feature_order) + ['severity']
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
        usecols = [c for c in needed_cols if c in header]
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=True)
        if len(df) > MAX_ROWS:
            df, _ = train_test_split(df, train_size=MAX_ROWS, random_state=SEED,
                                      stratify=df['severity'])
            df = df.reset_index(drop=True)

        available = [f for f in feature_order if f in df.columns]
        X = df[available].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_str = df['severity'].values
        y = np.array([SEV_MAP.get(s, 2) for s in y_str], dtype=np.int64)
        del df
        gc.collect()

        # Create windows
        n_samples = X.shape[0]
        n_windows = (n_samples - WINDOW_SIZE) // STRIDE + 1
        # Truncate to match cache
        n_windows = min(n_windows, len(softmax))

        # Compute window-level features
        X_mean = np.zeros((n_windows, len(available)), dtype=np.float32)
        y_win = np.zeros(n_windows, dtype=np.int64)
        for i in range(n_windows):
            start = i * STRIDE
            end = start + WINDOW_SIZE
            X_mean[i] = X[start:end].mean(axis=0)
            y_win[i] = y[end - 1]

        # Feature indices
        def _idx(name):
            try:
                return available.index(name)
            except ValueError:
                return -1

        idx_packets = _idx('flow_packets_per_s')
        idx_duration = _idx('flow_duration')
        idx_pktsize = _idx('avg_pkt_size')

        # Build state vectors (28D)
        states = np.zeros((n_windows, 28), dtype=np.float32)

        # 1. Link Metrics (4D)
        if idx_packets >= 0:
            v = X_mean[:, idx_packets]
            states[:, 0] = (v - v.min()) / (v.max() - v.min() + 1e-8)
        if idx_duration >= 0:
            v = X_mean[:, idx_duration]
            states[:, 1] = (v - v.min()) / (v.max() - v.min() + 1e-8)
        # entropy — computed per-window from severity distribution in neighborhood
        for i in range(n_windows):
            start_w = max(0, i - 10)
            window_sevs = softmax[start_w:i+1].argmax(axis=1)
            counts = Counter(window_sevs.tolist())
            total = len(window_sevs)
            ent = 0.0
            for c in counts.values():
                p = c / total
                if p > 0:
                    ent -= p * math.log2(p)
            states[i, 2] = ent / math.log2(3) if total > 1 else 0.5
        if idx_pktsize >= 0:
            v = X_mean[:, idx_pktsize]
            states[:, 3] = (v - v.min()) / (v.max() - v.min() + 1e-8)

        # 2. LFA State (4D)
        preds = softmax[:n_windows].argmax(axis=1)
        states[:, 4] = (preds != 2).astype(np.float32)  # attack_detected
        states[:, 5] = preds / 2.0  # severity_pred normalized
        states[:, 6] = softmax[:n_windows].max(axis=1)  # confidence
        # severity_ratio: fraction of High in last 10
        for i in range(n_windows):
            start_w = max(0, i - 10)
            recent = preds[start_w:i+1]
            states[i, 7] = np.mean(recent == 0)

        # 3. Mamba Embedding (16D)
        states[:, 8:24] = embed_16[:n_windows]

        # 4. Neighbor Flags (4D) — zeros for analysis (no runtime neighbors)
        # states[:, 24:28] = 0 (already zeros)

        all_states[link_name] = states
        all_severities[link_name] = y_win

        # Compute statistics
        stats = {
            'n_windows': int(n_windows),
            'severity_dist': {
                'High': int(np.sum(y_win == 0)),
                'Medium': int(np.sum(y_win == 1)),
                'Low': int(np.sum(y_win == 2)),
            },
            'mamba_accuracy': float(np.mean(preds == y_win)),
            'state_means': states.mean(axis=0).tolist(),
            'state_stds': states.std(axis=0).tolist(),
        }
        all_stats[link_name] = stats
        log(f'      Windows: {n_windows:,}, Mamba Acc: {stats["mamba_accuracy"]:.4f}')

        del X, y, X_mean, embed_16, embeddings_128
        gc.collect()

    return all_states, all_severities, all_stats


# ============================================================================
# Chart 1: State Vector Distribution per Link (box plots)
# ============================================================================
def plot_state_distribution(all_states, charts_dir, ds_label):
    """Box plots of key state components across 5 links."""
    fig, axes = plt.subplots(2, 4, figsize=(COL2, 4.5))

    key_dims = [
        (0, 'Traffic Load'),
        (1, 'Latency Est.'),
        (2, 'Entropy'),
        (3, 'Packet Rate'),
        (4, 'Attack Detected'),
        (5, 'Severity Pred.'),
        (6, 'Confidence'),
        (7, 'Severity Ratio'),
    ]

    for ax_idx, (dim, name) in enumerate(key_dims):
        ax = axes[ax_idx // 4, ax_idx % 4]
        data_per_link = []
        for ln in LINK_NAMES:
            if ln in all_states:
                # Subsample for box plot
                vals = all_states[ln][:5000, dim]
                data_per_link.append(vals)
            else:
                data_per_link.append(np.zeros(100))

        bp = ax.boxplot(data_per_link, labels=LINK_SHORT, patch_artist=True,
                        widths=0.6, showfliers=False,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], LINK_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(name, fontsize=8)
        ax.tick_params(axis='x', labelsize=5.5, rotation=30)
        if ax_idx % 4 == 0:
            ax.set_ylabel('Value')

    plt.suptitle(f'Fig. S1. State Vector Distribution per Link — {ds_label}',
                 fontsize=9, fontweight='bold', y=1.0)
    plt.tight_layout()
    out = charts_dir / 'FigS1_state_distribution.png'
    plt.savefig(out); plt.close()
    log(f'  Saved: {out.name}')


# ============================================================================
# Chart 2: Severity Distribution per Link (stacked bars)
# ============================================================================
def plot_severity_distribution(all_severities, charts_dir, ds_label):
    """Stacked bar chart of severity classes per link."""
    fig, ax = plt.subplots(figsize=(COL1, 2.8))

    x = np.arange(len(LINK_NAMES))
    width = 0.6
    bottoms = np.zeros(len(LINK_NAMES))

    for sev_idx, (sev_name, sev_color) in enumerate(zip(SEV_NAMES, SEV_COLORS)):
        counts = []
        for ln in LINK_NAMES:
            if ln in all_severities:
                total = len(all_severities[ln])
                c = np.sum(all_severities[ln] == sev_idx) / total
                counts.append(c)
            else:
                counts.append(0)
        counts = np.array(counts)
        ax.bar(x, counts, width, bottom=bottoms, label=sev_name,
               color=sev_color, alpha=0.85, edgecolor='black', linewidth=0.5)
        # Labels
        for i, (c, b) in enumerate(zip(counts, bottoms)):
            if c > 0.05:
                ax.text(x[i], b + c/2, f'{c:.0%}', ha='center', va='center',
                        fontsize=5.5, fontweight='bold', color='white')
        bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels(LINK_SHORT, fontsize=7)
    ax.set_ylabel('Proportion')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title(f'Fig. S2. Severity Distribution — {ds_label}', fontsize=9)
    plt.tight_layout()
    out = charts_dir / 'FigS2_severity_distribution.png'
    plt.savefig(out); plt.close()
    log(f'  Saved: {out.name}')


# ============================================================================
# Chart 3: Mamba Detection Accuracy per Link
# ============================================================================
def plot_mamba_accuracy(all_stats, charts_dir, ds_label):
    """Bar chart of Mamba-Only detection accuracy per link."""
    fig, ax = plt.subplots(figsize=(COL1, 2.5))

    x = np.arange(len(LINK_NAMES))
    accs = [all_stats.get(ln, {}).get('mamba_accuracy', 0) for ln in LINK_NAMES]

    bars = ax.bar(x, accs, color=LINK_COLORS, alpha=0.85,
                  edgecolor='black', linewidth=0.5, width=0.6)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(LINK_SHORT, fontsize=7)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.5, 1.05)
    ax.set_title(f'Fig. S3. Mamba-Only Detection Accuracy — {ds_label}', fontsize=9)
    plt.tight_layout()
    out = charts_dir / 'FigS3_mamba_accuracy.png'
    plt.savefig(out); plt.close()
    log(f'  Saved: {out.name}')


# ============================================================================
# Chart 4: State Vector Correlation Heatmap
# ============================================================================
def plot_state_correlation(all_states, charts_dir, ds_label):
    """Correlation heatmap of 28 state dimensions (using Link 1 data)."""
    # Use first link with enough data
    for ln in LINK_NAMES:
        if ln in all_states and len(all_states[ln]) > 1000:
            data = all_states[ln][:10000]
            break
    else:
        return

    # Use short names for the 8 key dims + 4 embed samples + 4 neighbor
    short_names = ['traf_load', 'latency', 'entropy', 'pkt_rate',
                   'atk_det', 'sev_pred', 'conf', 'sev_ratio',
                   'emb_0', 'emb_1', 'emb_2', 'emb_3',
                   'emb_4', 'emb_5', 'emb_6', 'emb_7',
                   'emb_8', 'emb_9', 'emb_10', 'emb_11',
                   'emb_12', 'emb_13', 'emb_14', 'emb_15',
                   'n_left', 'n_right', 'n_alert', 'n_coord']

    corr = np.corrcoef(data.T)

    fig, ax = plt.subplots(figsize=(COL2, 5.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                xticklabels=short_names, yticklabels=short_names,
                ax=ax, linewidths=0.3, vmin=-1, vmax=1,
                cbar_kws={'shrink': 0.7, 'label': 'Correlation'})
    ax.tick_params(axis='both', labelsize=5.5)
    ax.set_title(f'Fig. S4. State Vector Correlation — {ds_label}', fontsize=9)
    plt.tight_layout()
    out = charts_dir / 'FigS4_state_correlation.png'
    plt.savefig(out); plt.close()
    log(f'  Saved: {out.name}')


# ============================================================================
# Chart 5: Per-Link State Comparison (radar)
# ============================================================================
def plot_state_radar(all_stats, charts_dir, ds_label):
    """Radar chart comparing 5 links on key state metrics."""
    categories = ['Traffic\nLoad', 'Latency', 'Entropy', 'Pkt Rate',
                   'Atk Det.', 'Sev Pred.', 'Confidence', 'Sev Ratio']
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(COL1 + 0.5, COL1 + 0.5), subplot_kw=dict(polar=True))

    for li, ln in enumerate(LINK_NAMES):
        if ln not in all_stats:
            continue
        means = all_stats[ln]['state_means'][:8]
        values = list(means) + [means[0]]
        ax.plot(angles, values, '-', color=LINK_COLORS[li], linewidth=1.5,
                label=LINK_SHORT[li], marker='o', markersize=3)
        ax.fill(angles, values, alpha=0.05, color=LINK_COLORS[li])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=5)
    ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.35, 1.1))
    ax.set_title(f'Fig. S5. Per-Link State Profile — {ds_label}', fontsize=9, y=1.12)
    plt.tight_layout()
    out = charts_dir / 'FigS5_state_radar.png'
    plt.savefig(out); plt.close()
    log(f'  Saved: {out.name}')


# ============================================================================
# Chart 6: State by Severity Class (violin plots)
# ============================================================================
def plot_state_by_severity(all_states, all_severities, charts_dir, ds_label):
    """Violin plots: key state dims grouped by true severity."""
    # Combine all links
    all_s = []
    all_y = []
    for ln in LINK_NAMES:
        if ln in all_states:
            all_s.append(all_states[ln][:5000])
            all_y.append(all_severities[ln][:5000])
    if not all_s:
        return
    states = np.concatenate(all_s, axis=0)
    severities = np.concatenate(all_y, axis=0)

    key_dims = [(0, 'Traffic Load'), (1, 'Latency'), (6, 'Confidence'), (7, 'Sev. Ratio')]

    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.8))
    for ax_idx, (dim, name) in enumerate(key_dims):
        ax = axes[ax_idx]
        data = []
        labels = []
        for sev_idx, sev_name in enumerate(SEV_NAMES):
            mask = severities == sev_idx
            if mask.sum() > 0:
                data.append(states[mask, dim])
                labels.append(sev_name)

        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(SEV_COLORS[i])
            pc.set_alpha(0.7)
        parts['cmeans'].set_color('black')
        parts['cmedians'].set_color('red')

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(name, fontsize=8)
        if ax_idx == 0:
            ax.set_ylabel('Value')

    plt.suptitle(f'Fig. S6. State Features by Severity — {ds_label}',
                 fontsize=9, fontweight='bold', y=1.0)
    plt.tight_layout()
    out = charts_dir / 'FigS6_state_by_severity.png'
    plt.savefig(out); plt.close()
    log(f'  Saved: {out.name}')


# ============================================================================
# Chart 7: Mamba Embedding t-SNE / PCA Visualization
# ============================================================================
def plot_embedding_pca(all_states, all_severities, charts_dir, ds_label):
    """PCA 2D projection of 16D Mamba embeddings, colored by severity."""
    fig, axes = plt.subplots(1, 5, figsize=(COL2, 2.5))

    for li, ln in enumerate(LINK_NAMES):
        ax = axes[li]
        if ln not in all_states:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            continue

        embeds = all_states[ln][:5000, 8:24]  # 16D mamba embed
        sevs = all_severities[ln][:5000]

        pca = PCA(n_components=2, random_state=SEED)
        proj = pca.fit_transform(embeds)

        for sev_idx in range(3):
            mask = sevs == sev_idx
            if mask.sum() > 0:
                ax.scatter(proj[mask, 0], proj[mask, 1], c=SEV_COLORS[sev_idx],
                          s=3, alpha=0.3, label=SEV_NAMES[sev_idx] if li == 4 else '')

        ax.set_title(LINK_SHORT[li], fontsize=7, fontweight='bold')
        ax.tick_params(labelsize=5)
        if li == 0:
            ax.set_ylabel('PC2', fontsize=7)
        ax.set_xlabel('PC1', fontsize=6)

    axes[4].legend(fontsize=5, loc='lower right', markerscale=3)
    plt.suptitle(f'Fig. S7. Mamba Embedding PCA — {ds_label}',
                 fontsize=9, fontweight='bold', y=1.0)
    plt.tight_layout()
    out = charts_dir / 'FigS7_embedding_pca.png'
    plt.savefig(out); plt.close()
    log(f'  Saved: {out.name}')


# ============================================================================
# Chart 8: Cross-Dataset Comparison (if both available)
# ============================================================================
def plot_cross_dataset_comparison(stats_ciciot, stats_diad, charts_dir):
    """Compare state vector statistics between CICIoT2023 and DIAD2024."""
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.8))

    # 8a: Mamba accuracy per link
    ax = axes[0]
    x = np.arange(5)
    width = 0.35
    acc_c = [stats_ciciot.get(ln, {}).get('mamba_accuracy', 0) for ln in LINK_NAMES]
    acc_d = [stats_diad.get(ln, {}).get('mamba_accuracy', 0) for ln in LINK_NAMES]
    ax.bar(x - width/2, acc_c, width, label='CICIoT2023', color='#2166AC',
           alpha=0.85, hatch='///', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, acc_d, width, label='DIAD2024', color='#B2182B',
           alpha=0.85, hatch='...', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LINK_SHORT, fontsize=5.5, rotation=30)
    ax.set_ylabel('Mamba Accuracy')
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=6)
    ax.set_title('(a) Detection Accuracy', fontsize=8)

    # 8b: Severity proportions comparison
    ax = axes[1]
    for di, (stats, ds_key) in enumerate([(stats_ciciot, 'ciciot2023'), (stats_diad, 'diad2024')]):
        proportions = np.zeros(3)
        total = 0
        for ln in LINK_NAMES:
            if ln in stats:
                sd = stats[ln]['severity_dist']
                proportions[0] += sd['High']
                proportions[1] += sd['Medium']
                proportions[2] += sd['Low']
                total += stats[ln]['n_windows']
        if total > 0:
            proportions /= total
        x_pos = np.arange(3) + di * 0.35 - 0.175
        bars = ax.bar(x_pos, proportions, 0.3, label=DS_LABELS[ds_key],
                      color=['#2166AC', '#B2182B'][di], alpha=0.85,
                      hatch=['///', '...'][di], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, proportions):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.0%}', ha='center', fontsize=5.5)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(SEV_NAMES, fontsize=7)
    ax.set_ylabel('Proportion')
    ax.legend(fontsize=6)
    ax.set_title('(b) Severity Distribution', fontsize=8)

    # 8c: Mean state values (first 8 dims)
    ax = axes[2]
    dim_names = ['TLoad', 'Lat', 'Ent', 'PRate', 'AtkD', 'SPred', 'Conf', 'SRat']
    for di, (stats, ds_key) in enumerate([(stats_ciciot, 'ciciot2023'), (stats_diad, 'diad2024')]):
        means = np.zeros(8)
        count = 0
        for ln in LINK_NAMES:
            if ln in stats:
                means += np.array(stats[ln]['state_means'][:8])
                count += 1
        if count > 0:
            means /= count
        x_pos = np.arange(8) + di * 0.35 - 0.175
        ax.bar(x_pos, means, 0.3, label=DS_LABELS[ds_key],
               color=['#2166AC', '#B2182B'][di], alpha=0.85,
               hatch=['///', '...'][di], edgecolor='black', linewidth=0.5)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(dim_names, fontsize=5.5, rotation=30)
    ax.set_ylabel('Mean Value')
    ax.legend(fontsize=6)
    ax.set_title('(c) State Component Means', fontsize=8)

    plt.suptitle('Fig. S8. Cross-Dataset State Vector Comparison',
                 fontsize=9, fontweight='bold', y=1.0)
    plt.tight_layout()
    out = charts_dir / 'FigS8_cross_dataset_comparison.png'
    plt.savefig(out); plt.close()
    log(f'  Saved: {out.name}')


# ============================================================================
# LaTeX Table: State Vector Statistics
# ============================================================================
def generate_state_table(all_stats_dict, charts_dir):
    """Generate LaTeX table of state vector statistics."""
    lines = []
    lines.append('% Table: State Vector Statistics per Link')
    lines.append('\\begin{table*}[!t]')
    lines.append('\\centering')
    lines.append('\\caption{State Vector Statistics Across SDN Links}')
    lines.append('\\label{tab:state_vector}')
    lines.append('\\renewcommand{\\arraystretch}{1.2}')
    lines.append('\\begin{tabular}{l|l|ccccc}')
    lines.append('\\hline')
    lines.append('\\textbf{Dataset} & \\textbf{Metric} & \\textbf{L1-Core} & '
                 '\\textbf{L2-AggrL} & \\textbf{L3-AggrR} & '
                 '\\textbf{L4-EdgeL} & \\textbf{L5-EdgeR} \\\\')
    lines.append('\\hline')

    for ds_key, ds_label in [('ciciot2023', 'CICIoT2023'), ('diad2024', 'DIAD2024')]:
        stats = all_stats_dict.get(ds_key, {})
        if not stats:
            continue

        # Windows
        vals = [str(stats.get(ln, {}).get('n_windows', 0)) for ln in LINK_NAMES]
        lines.append(f'\\multirow{{4}}{{*}}{{{ds_label}}} & Windows & {" & ".join(vals)} \\\\')

        # Mamba Accuracy
        vals = [f'{stats.get(ln, {}).get("mamba_accuracy", 0):.4f}' for ln in LINK_NAMES]
        lines.append(f' & Mamba Acc & {" & ".join(vals)} \\\\')

        # High %
        vals = []
        for ln in LINK_NAMES:
            sd = stats.get(ln, {}).get('severity_dist', {})
            total = stats.get(ln, {}).get('n_windows', 1)
            vals.append(f'{sd.get("High", 0)/total:.1%}')
        lines.append(f' & High \\% & {" & ".join(vals)} \\\\')

        # Confidence mean
        vals = [f'{stats.get(ln, {}).get("state_means", [0]*7)[6]:.3f}' for ln in LINK_NAMES]
        lines.append(f' & Conf. Mean & {" & ".join(vals)} \\\\')

        lines.append('\\hline')

    lines.append('\\end{tabular}')
    lines.append('\\end{table*}')

    out = charts_dir / 'state_vector_table.tex'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log(f'  Saved: {out.name}')


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='State Vector Analysis')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['ciciot2023', 'diad2024', 'all'])
    args = parser.parse_args()

    charts_dir = RESULTS_BASE / 'state_vector_charts'
    os.makedirs(charts_dir, exist_ok=True)

    log('=' * 60)
    log('  State Vector Analysis — IEEE Charts')
    log('=' * 60)

    all_stats_dict = {}

    datasets = ['ciciot2023', 'diad2024'] if args.dataset == 'all' else [args.dataset]

    for ds_key in datasets:
        ds_label = DS_LABELS[ds_key]
        log(f'\n{"="*50}')
        log(f'  Dataset: {ds_label}')
        log(f'{"="*50}')

        # Check if mamba cache exists
        cache_dir = RESULTS_BASE / ds_key / 'mamba_cache'
        if not cache_dir.exists():
            log(f'  WARNING: No mamba cache at {cache_dir}. Run train_rl_multiagent.py first.')
            continue

        states, severities, stats = build_state_vectors(ds_key)
        all_stats_dict[ds_key] = stats

        log(f'\n  Generating Charts for {ds_label}...')
        plot_state_distribution(states, charts_dir, ds_label)
        plot_severity_distribution(severities, charts_dir, ds_label)
        plot_mamba_accuracy(stats, charts_dir, ds_label)
        plot_state_correlation(states, charts_dir, ds_label)
        plot_state_radar(stats, charts_dir, ds_label)
        plot_state_by_severity(states, severities, charts_dir, ds_label)
        plot_embedding_pca(states, severities, charts_dir, ds_label)

        # Save stats JSON
        with open(charts_dir / f'{ds_key}_state_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        del states, severities
        gc.collect()

    # Cross-dataset comparison
    if len(all_stats_dict) == 2:
        log(f'\n  Cross-Dataset Comparison...')
        plot_cross_dataset_comparison(all_stats_dict['ciciot2023'],
                                      all_stats_dict['diad2024'], charts_dir)

    # LaTeX table
    generate_state_table(all_stats_dict, charts_dir)

    log(f'\n{"="*60}')
    log(f'  Output: {charts_dir}')
    log('=' * 60)


if __name__ == '__main__':
    main()
