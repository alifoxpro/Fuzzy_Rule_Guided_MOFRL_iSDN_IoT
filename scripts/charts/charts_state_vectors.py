#!/usr/bin/env python
"""
IEEE-Quality Academic Charts for State Vectors (28D).
Reads from extracted CSV files — no retraining needed.

Categories:
  1. Feature Distributions (KDE per severity)
  2. Correlation Heatmap (28×28)
  3. Per-Link Comparison (box/violin)
  4. t-SNE / PCA Scatter (28D → 2D)
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

warnings.filterwarnings('ignore')

# ── IEEE IEEEtran Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.3',
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'mathtext.fontset': 'stix',
})

FIG_W, FIG_H = 7.16, 4.5  # IEEE double-column

BASE = Path(__file__).resolve().parent.parent.parent

LINK_SHORT = ['L1_Core', 'L2_AggrL', 'L3_AggrR', 'L4_EdgeL', 'L5_EdgeR']
LINK_LABELS = ['L1-Core', 'L2-AggrL', 'L3-AggrR', 'L4-EdgeL', 'L5-EdgeR']

SEVERITY_COLORS = {'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}
LINK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
ACTION_COLORS = {'ALLOW': '#2ca02c', 'RATE_LIMIT': '#ff7f0e', 'BLOCK': '#d62728'}

STATE_NAMES_SHORT = [
    'traffic', 'latency', 'entropy', 'pkt_rate',
    'atk_det', 'sev_pred', 'mamba_conf', 'sev_ratio',
    'pca_0', 'pca_1', 'pca_2', 'pca_3',
    'pca_4', 'pca_5', 'pca_6', 'pca_7',
    'pca_8', 'pca_9', 'pca_10', 'pca_11',
    'pca_12', 'pca_13', 'pca_14', 'pca_15',
    'nbr_left', 'nbr_right', 'nbr_alert', 'nbr_avg_sev'
]

STATE_GROUPS = {
    'Link Metrics': [0, 1, 2, 3],
    'Mamba Detection': [4, 5, 6, 7],
    'Mamba PCA': list(range(8, 24)),
    'Neighbor Info': [24, 25, 26, 27],
}

DS_LABELS = {'ciciot2023': 'CICIoT2023', 'diad2024': 'DIAD2024'}

# Exact 28 state column names (matching extract_state_vectors.py)
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


def load_data(ds_key):
    """Load all link CSVs for a dataset."""
    sv_dir = BASE / 'experiments' / f'state_vectors_{ds_key}'
    link_dfs = {}
    for i, short in enumerate(LINK_SHORT):
        csv_path = sv_dir / f'{short}_state_vectors.csv'
        if csv_path.exists():
            link_dfs[short] = pd.read_csv(csv_path)
    combined_path = sv_dir / 'all_links_state_vectors.csv'
    if combined_path.exists():
        combined = pd.read_csv(combined_path)
    else:
        combined = pd.concat(link_dfs.values(), ignore_index=True)
    return link_dfs, combined


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"    Saved: {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════
#  Category 1: Feature Distributions
# ═══════════════════════════════════════════════════════════

def fig01_link_metrics_kde(combined, charts_dir, ds_label):
    """KDE of 4 link metrics split by severity class."""
    cols = ['traffic_load', 'latency_estimate', 'entropy', 'packet_rate']
    titles = ['Traffic Load', 'Latency Estimate', 'Entropy', 'Packet Rate']

    for ci, col in enumerate(cols):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        for sev_name, color in SEVERITY_COLORS.items():
            subset = combined[combined['true_severity_name'] == sev_name][col].dropna()
            if len(subset) > 0:
                subset_sample = subset.sample(n=min(50000, len(subset)), random_state=42)
                subset_sample.plot.kde(ax=ax, color=color, label=sev_name, linewidth=2, alpha=0.85)
        ax.set_xlabel(titles[ci])
        ax.set_ylabel('Probability Density')
        ax.set_title(f'State Feature Distribution: {titles[ci]} by Severity\n{ds_label}',
                      fontweight='bold')
        ax.legend(title='Severity', frameon=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
        save_fig(fig, charts_dir / f'fig01{chr(97+ci)}_kde_{col}.png')


def fig01e_mamba_features_kde(combined, charts_dir, ds_label):
    """KDE of 4 Mamba detection features split by severity."""
    cols = ['attack_detected', 'severity_pred', 'mamba_confidence', 'severity_ratio']
    titles = ['Attack Detected', 'Severity Prediction', 'Mamba Confidence', 'Severity Ratio']

    for ci, col in enumerate(cols):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        for sev_name, color in SEVERITY_COLORS.items():
            subset = combined[combined['true_severity_name'] == sev_name][col].dropna()
            if len(subset) > 0:
                if col == 'attack_detected':
                    # Binary — bar chart instead of KDE
                    vals = subset.value_counts(normalize=True).sort_index()
                    positions = [v + (list(SEVERITY_COLORS.keys()).index(sev_name) - 1) * 0.25 for v in vals.index]
                    ax.bar(positions, vals.values, width=0.2, color=color, alpha=0.8, label=sev_name)
                else:
                    subset_sample = subset.sample(n=min(50000, len(subset)), random_state=42)
                    subset_sample.plot.kde(ax=ax, color=color, label=sev_name, linewidth=2, alpha=0.85)
        ax.set_xlabel(titles[ci])
        ax.set_ylabel('Probability Density' if col != 'attack_detected' else 'Proportion')
        ax.set_title(f'State Feature Distribution: {titles[ci]} by Severity\n{ds_label}',
                      fontweight='bold')
        ax.legend(title='Severity', frameon=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
        save_fig(fig, charts_dir / f'fig01{chr(101+ci)}_kde_{col}.png')


def fig01i_softmax_distribution(combined, charts_dir, ds_label):
    """Distribution of Mamba softmax probabilities P(H), P(M), P(L) by true severity."""
    for sev_name, color in SEVERITY_COLORS.items():
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        subset = combined[combined['true_severity_name'] == sev_name]
        if len(subset) == 0:
            plt.close(fig)
            continue
        sample = subset.sample(n=min(50000, len(subset)), random_state=42)
        for pcol, pcolor, plabel in [('P_High', '#d62728', 'P(High)'),
                                      ('P_Medium', '#ff7f0e', 'P(Medium)'),
                                      ('P_Low', '#2ca02c', 'P(Low)')]:
            sample[pcol].plot.kde(ax=ax, color=pcolor, label=plabel, linewidth=2, alpha=0.85)
        ax.set_xlabel('Softmax Probability')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Mamba Softmax Distribution — True Severity: {sev_name}\n{ds_label}',
                      fontweight='bold')
        ax.set_xlim(-0.1, 1.1)
        ax.legend(frameon=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
        save_fig(fig, charts_dir / f'fig01_softmax_{sev_name.lower()}.png')


# ═══════════════════════════════════════════════════════════
#  Category 2: Correlation Heatmap
# ═══════════════════════════════════════════════════════════

def fig02a_correlation_heatmap(combined, charts_dir, ds_label):
    """28×28 correlation heatmap of state features."""
    state_cols = STATE_COLUMNS
    corr = combined[state_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson Correlation')

    ax.set_xticks(range(28))
    ax.set_xticklabels(STATE_NAMES_SHORT, rotation=90, fontsize=6)
    ax.set_yticks(range(28))
    ax.set_yticklabels(STATE_NAMES_SHORT, fontsize=6)

    # Draw group boundaries
    boundaries = [0, 4, 8, 24, 28]
    group_names = ['Link\nMetrics', 'Mamba\nDetect', 'Mamba PCA (16D)', 'Neighbor']
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color='black', linewidth=1.5)
        ax.axvline(b - 0.5, color='black', linewidth=1.5)

    # Group labels on top
    for gi, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mid = (start + end) / 2 - 0.5
        ax.text(mid, -1.5, group_names[gi], ha='center', va='bottom', fontsize=7,
                fontweight='bold', color='navy')

    ax.set_title(f'State Vector Correlation Matrix (28D)\n{ds_label}', fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(True)
    save_fig(fig, charts_dir / 'fig02a_correlation_heatmap.png')


def fig02b_correlation_per_link(link_dfs, charts_dir, ds_label):
    """Correlation heatmap for key features (non-PCA) per link."""
    key_indices = [0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27]
    key_names = [STATE_NAMES_SHORT[i] for i in key_indices]

    for li, (short, df) in enumerate(link_dfs.items()):
        state_cols = STATE_COLUMNS
        key_cols = [state_cols[i] for i in key_indices]
        corr = df[key_cols].corr()

        fig, ax = plt.subplots(figsize=(FIG_W, 5))
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson Correlation')

        ax.set_xticks(range(len(key_names)))
        ax.set_xticklabels(key_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(key_names)))
        ax.set_yticklabels(key_names, fontsize=8)

        # Annotate values
        for yi in range(len(key_names)):
            for xi in range(len(key_names)):
                val = corr.values[yi, xi]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(xi, yi, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)

        ax.set_title(f'Feature Correlation — {LINK_LABELS[li]}\n{ds_label}',
                      fontweight='bold')
        for spine in ax.spines.values():
            spine.set_visible(True)
        save_fig(fig, charts_dir / f'fig02b_corr_{short}.png')


# ═══════════════════════════════════════════════════════════
#  Category 3: Per-Link Comparison
# ═══════════════════════════════════════════════════════════

def fig03a_box_link_metrics(link_dfs, charts_dir, ds_label):
    """Box plots comparing link metrics across 5 links."""
    cols = ['traffic_load', 'latency_estimate', 'entropy', 'packet_rate']
    titles = ['Traffic Load', 'Latency Estimate', 'Entropy', 'Packet Rate']

    for ci, col in enumerate(cols):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        data_list = []
        labels = []
        for li, (short, df) in enumerate(link_dfs.items()):
            # Sample for performance
            sample = df[col].dropna().sample(n=min(10000, len(df)), random_state=42)
            data_list.append(sample.values)
            labels.append(LINK_LABELS[li])

        bp = ax.boxplot(data_list, labels=labels, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        flierprops=dict(marker='.', markersize=2, alpha=0.3))
        for patch, color in zip(bp['boxes'], LINK_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel('SDN Link')
        ax.set_ylabel(titles[ci])
        ax.set_title(f'Per-Link Distribution: {titles[ci]}\n{ds_label}', fontweight='bold')
        for spine in ax.spines.values():
            spine.set_visible(True)
        save_fig(fig, charts_dir / f'fig03a_box_{col}.png')


def fig03b_violin_mamba(link_dfs, charts_dir, ds_label):
    """Violin plots of Mamba confidence and severity ratio across links."""
    cols = ['mamba_confidence', 'severity_ratio']
    titles = ['Mamba Confidence', 'Severity Ratio (High %)']

    for ci, col in enumerate(cols):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        data_list = []
        for short, df in link_dfs.items():
            sample = df[col].dropna().sample(n=min(10000, len(df)), random_state=42)
            data_list.append(sample.values)

        vp = ax.violinplot(data_list, positions=range(len(link_dfs)),
                           showmeans=True, showmedians=True)
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(LINK_COLORS[i])
            body.set_alpha(0.7)
        vp['cmeans'].set_color('red')
        vp['cmedians'].set_color('black')

        ax.set_xticks(range(len(link_dfs)))
        ax.set_xticklabels(LINK_LABELS)
        ax.set_xlabel('SDN Link')
        ax.set_ylabel(titles[ci])
        ax.set_title(f'Per-Link Distribution: {titles[ci]}\n{ds_label}', fontweight='bold')

        # Add legend for mean/median
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='red', label='Mean'),
                          Line2D([0], [0], color='black', label='Median')]
        ax.legend(handles=legend_elements, frameon=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
        save_fig(fig, charts_dir / f'fig03b_violin_{col}.png')


def fig03c_severity_distribution(link_dfs, charts_dir, ds_label):
    """Stacked bar: severity class distribution per link."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(link_dfs))
    width = 0.6

    high_pcts, med_pcts, low_pcts = [], [], []
    for short, df in link_dfs.items():
        total = len(df)
        high_pcts.append(sum(df['true_severity'] == 0) / total * 100)
        med_pcts.append(sum(df['true_severity'] == 1) / total * 100)
        low_pcts.append(sum(df['true_severity'] == 2) / total * 100)

    bars_h = ax.bar(x, high_pcts, width, label='High', color=SEVERITY_COLORS['High'], alpha=0.85)
    bars_m = ax.bar(x, med_pcts, width, bottom=high_pcts, label='Medium',
                    color=SEVERITY_COLORS['Medium'], alpha=0.85)
    bottoms = [h + m for h, m in zip(high_pcts, med_pcts)]
    bars_l = ax.bar(x, low_pcts, width, bottom=bottoms, label='Low',
                    color=SEVERITY_COLORS['Low'], alpha=0.85)

    # Annotate percentages
    for i in range(len(link_dfs)):
        if high_pcts[i] > 5:
            ax.text(i, high_pcts[i] / 2, f'{high_pcts[i]:.1f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
        if med_pcts[i] > 5:
            ax.text(i, high_pcts[i] + med_pcts[i] / 2, f'{med_pcts[i]:.1f}%', ha='center',
                    va='center', fontsize=8, fontweight='bold', color='white')
        if low_pcts[i] > 5:
            ax.text(i, bottoms[i] + low_pcts[i] / 2, f'{low_pcts[i]:.1f}%', ha='center',
                    va='center', fontsize=8, fontweight='bold', color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(LINK_LABELS)
    ax.set_xlabel('SDN Link')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Severity Class Distribution per Link\n{ds_label}', fontweight='bold')
    ax.legend(title='Severity', frameon=True)
    ax.set_ylim(0, 105)
    for spine in ax.spines.values():
        spine.set_visible(True)
    save_fig(fig, charts_dir / 'fig03c_severity_dist.png')


def fig03d_neighbor_comparison(link_dfs, charts_dir, ds_label):
    """Grouped bar: neighbor features across links."""
    cols = ['neighbor_left_severity', 'neighbor_right_severity',
            'neighbor_alert_count', 'neighbor_avg_severity']
    labels = ['Left Sev.', 'Right Sev.', 'Alert Count', 'Avg Sev.']

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(link_dfs))
    n_cols = len(cols)
    width = 0.18

    for ci, (col, lbl) in enumerate(zip(cols, labels)):
        means = []
        stds = []
        for short, df in link_dfs.items():
            means.append(df[col].mean())
            stds.append(df[col].std())
        offset = (ci - n_cols / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=lbl,
                      capsize=3, alpha=0.85, error_kw={'linewidth': 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(LINK_LABELS)
    ax.set_xlabel('SDN Link')
    ax.set_ylabel('Feature Value (mean ± std)')
    ax.set_title(f'Neighbor Communication Features per Link\n{ds_label}', fontweight='bold')
    ax.legend(frameon=True, ncol=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
    save_fig(fig, charts_dir / 'fig03d_neighbor_comparison.png')


def fig03e_action_distribution(link_dfs, charts_dir, ds_label):
    """Optimal action distribution per link."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(link_dfs))
    width = 0.25

    for ai, (action, color) in enumerate(ACTION_COLORS.items()):
        pcts = []
        for short, df in link_dfs.items():
            total = len(df)
            pcts.append(sum(df['optimal_action'] == action) / total * 100)
        offset = (ai - 1) * width
        bars = ax.bar(x + offset, pcts, width, label=action, color=color, alpha=0.85)
        for xi, pct in enumerate(pcts):
            if pct > 3:
                ax.text(xi + offset, pct + 1, f'{pct:.1f}%', ha='center', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(LINK_LABELS)
    ax.set_xlabel('SDN Link')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Optimal Action Distribution per Link\n{ds_label}', fontweight='bold')
    ax.legend(title='Action', frameon=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
    save_fig(fig, charts_dir / 'fig03e_action_dist.png')


# ═══════════════════════════════════════════════════════════
#  Category 4: t-SNE / PCA Scatter
# ═══════════════════════════════════════════════════════════

def fig04a_pca_scatter_severity(combined, charts_dir, ds_label):
    """PCA 28D → 2D scatter colored by severity."""
    from sklearn.decomposition import PCA

    state_cols = STATE_COLUMNS
    X = combined[state_cols].values.astype(np.float32)

    # Sample for performance
    n_sample = min(20000, len(X))
    idx = np.random.RandomState(42).choice(len(X), n_sample, replace=False)
    X_sample = X[idx]
    labels = combined['true_severity_name'].values[idx]

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for sev_name, color in SEVERITY_COLORS.items():
        mask = labels == sev_name
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=sev_name,
                   s=8, alpha=0.4, edgecolors='none')

    ax.set_xlabel(f'PC-1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC-2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.set_title(f'PCA Projection of State Vectors (28D → 2D) — by Severity\n{ds_label}',
                  fontweight='bold')
    ax.legend(title='Severity', frameon=True, markerscale=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
    save_fig(fig, charts_dir / 'fig04a_pca_severity.png')


def fig04b_pca_scatter_link(combined, charts_dir, ds_label):
    """PCA 28D → 2D scatter colored by link."""
    from sklearn.decomposition import PCA

    state_cols = STATE_COLUMNS
    X = combined[state_cols].values.astype(np.float32)

    n_sample = min(20000, len(X))
    idx = np.random.RandomState(42).choice(len(X), n_sample, replace=False)
    X_sample = X[idx]
    links = combined['link'].values[idx]

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for li, short in enumerate(LINK_SHORT):
        mask = links == short
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=LINK_COLORS[li],
                   label=LINK_LABELS[li], s=8, alpha=0.4, edgecolors='none')

    ax.set_xlabel(f'PC-1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC-2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.set_title(f'PCA Projection of State Vectors (28D → 2D) — by Link\n{ds_label}',
                  fontweight='bold')
    ax.legend(title='Link', frameon=True, markerscale=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
    save_fig(fig, charts_dir / 'fig04b_pca_link.png')


def fig04c_tsne_severity(combined, charts_dir, ds_label):
    """t-SNE 28D → 2D scatter colored by severity."""
    from sklearn.manifold import TSNE

    state_cols = STATE_COLUMNS
    X = combined[state_cols].values.astype(np.float32)

    # t-SNE needs smaller sample
    n_sample = min(10000, len(X))
    idx = np.random.RandomState(42).choice(len(X), n_sample, replace=False)
    X_sample = X[idx]
    labels = combined['true_severity_name'].values[idx]

    print("    Computing t-SNE (may take 1-2 minutes)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for sev_name, color in SEVERITY_COLORS.items():
        mask = labels == sev_name
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=sev_name,
                   s=8, alpha=0.5, edgecolors='none')

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(f't-SNE Projection of State Vectors (28D → 2D) — by Severity\n{ds_label}',
                  fontweight='bold')
    ax.legend(title='Severity', frameon=True, markerscale=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
    save_fig(fig, charts_dir / 'fig04c_tsne_severity.png')


def fig04d_tsne_link(combined, charts_dir, ds_label):
    """t-SNE 28D → 2D scatter colored by link."""
    from sklearn.manifold import TSNE

    state_cols = STATE_COLUMNS
    X = combined[state_cols].values.astype(np.float32)

    n_sample = min(10000, len(X))
    idx = np.random.RandomState(42).choice(len(X), n_sample, replace=False)
    X_sample = X[idx]
    links = combined['link'].values[idx]

    print("    Computing t-SNE by link...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for li, short in enumerate(LINK_SHORT):
        mask = links == short
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=LINK_COLORS[li],
                   label=LINK_LABELS[li], s=8, alpha=0.5, edgecolors='none')

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(f't-SNE Projection of State Vectors (28D → 2D) — by Link\n{ds_label}',
                  fontweight='bold')
    ax.legend(title='Link', frameon=True, markerscale=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
    save_fig(fig, charts_dir / 'fig04d_tsne_link.png')


def fig04e_pca_explained_variance(combined, charts_dir, ds_label):
    """PCA explained variance (scree plot) showing how many components capture the information."""
    from sklearn.decomposition import PCA

    state_cols = STATE_COLUMNS
    X = combined[state_cols].values.astype(np.float32)
    n_sample = min(50000, len(X))
    X_sample = X[np.random.RandomState(42).choice(len(X), n_sample, replace=False)]

    pca = PCA(n_components=28, random_state=42)
    pca.fit(X_sample)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
    individual = pca.explained_variance_ratio_ * 100

    ax.bar(range(1, 29), individual, color='#1f77b4', alpha=0.7, label='Individual')
    ax.plot(range(1, 29), cumulative, 'ro-', linewidth=2, markersize=5, label='Cumulative')

    # Mark 95% threshold
    ax.axhline(95, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='95% Threshold')
    n95 = np.argmax(cumulative >= 95) + 1
    ax.axvline(n95, color='green', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(n95 + 0.5, 50, f'n={n95}', fontsize=9, color='green', fontweight='bold')

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title(f'PCA Scree Plot — State Vector Dimensionality Analysis\n{ds_label}',
                  fontweight='bold')
    ax.legend(frameon=True)
    ax.set_xlim(0.5, 28.5)
    ax.set_ylim(0, 105)
    for spine in ax.spines.values():
        spine.set_visible(True)
    save_fig(fig, charts_dir / 'fig04e_pca_scree.png')


# ═══════════════════════════════════════════════════════════
#  Cross-Dataset Comparison
# ═══════════════════════════════════════════════════════════

def fig05_cross_dataset(charts_dir):
    """Compare state vector statistics across CICIoT2023 and DIAD2024."""
    datasets = {}
    for ds_key in ['ciciot2023', 'diad2024']:
        sv_dir = BASE / 'experiments' / f'state_vectors_{ds_key}'
        combined_path = sv_dir / 'all_links_state_vectors.csv'
        if combined_path.exists():
            datasets[ds_key] = pd.read_csv(combined_path)

    if len(datasets) < 2:
        print("    Skipping cross-dataset: need both datasets")
        return

    # fig05a: Mean feature values comparison
    key_cols = ['traffic_load', 'latency_estimate', 'entropy', 'packet_rate',
                'attack_detected', 'mamba_confidence', 'severity_ratio']
    key_labels = ['Traffic', 'Latency', 'Entropy', 'Pkt Rate',
                  'Atk Det.', 'Mamba Conf.', 'Sev Ratio']

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(key_cols))
    width = 0.35

    for di, (ds_key, df) in enumerate(datasets.items()):
        means = [df[col].mean() for col in key_cols]
        stds = [df[col].std() for col in key_cols]
        offset = (di - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=DS_LABELS[ds_key],
               capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(key_labels, rotation=30, ha='right')
    ax.set_ylabel('Feature Value (mean ± std)')
    ax.set_title('Cross-Dataset State Vector Comparison\nCICIoT2023 vs DIAD2024',
                  fontweight='bold')
    ax.legend(frameon=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
    save_fig(fig, charts_dir / 'fig05a_cross_feature_means.png')

    # fig05b: Severity distribution comparison
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(3)
    width = 0.35

    for di, (ds_key, df) in enumerate(datasets.items()):
        total = len(df)
        pcts = [sum(df['true_severity'] == s) / total * 100 for s in [0, 1, 2]]
        offset = (di - 0.5) * width
        bars = ax.bar(x + offset, pcts, width, label=DS_LABELS[ds_key], alpha=0.85)
        for xi, pct in enumerate(pcts):
            ax.text(xi + offset, pct + 1, f'{pct:.1f}%', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(['High (Attack)', 'Medium (Suspicious)', 'Low (Benign)'])
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Severity Class Distribution — Cross-Dataset Comparison',
                  fontweight='bold')
    ax.legend(frameon=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
    save_fig(fig, charts_dir / 'fig05b_cross_severity.png')


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def generate_dataset_charts(ds_key):
    ds_label = DS_LABELS[ds_key]
    charts_dir = BASE / 'experiments' / f'state_vectors_{ds_key}' / 'charts'
    os.makedirs(charts_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  State Vector Charts — {ds_label}")
    print(f"  Output: {charts_dir}")
    print(f"{'='*60}")

    link_dfs, combined = load_data(ds_key)
    if len(link_dfs) == 0:
        print(f"  ERROR: No state vector CSVs found for {ds_key}")
        print(f"  Run: python scripts/extract_state_vectors.py --dataset {ds_key}")
        return

    print(f"  Loaded: {len(combined):,} rows, {len(link_dfs)} links")

    # Cat 1: Feature Distributions
    print("\n  --- Category 1: Feature Distributions ---")
    fig01_link_metrics_kde(combined, charts_dir, ds_label)
    fig01e_mamba_features_kde(combined, charts_dir, ds_label)
    fig01i_softmax_distribution(combined, charts_dir, ds_label)

    # Cat 2: Correlation
    print("\n  --- Category 2: Correlation Heatmaps ---")
    fig02a_correlation_heatmap(combined, charts_dir, ds_label)
    fig02b_correlation_per_link(link_dfs, charts_dir, ds_label)

    # Cat 3: Per-Link Comparison
    print("\n  --- Category 3: Per-Link Comparison ---")
    fig03a_box_link_metrics(link_dfs, charts_dir, ds_label)
    fig03b_violin_mamba(link_dfs, charts_dir, ds_label)
    fig03c_severity_distribution(link_dfs, charts_dir, ds_label)
    fig03d_neighbor_comparison(link_dfs, charts_dir, ds_label)
    fig03e_action_distribution(link_dfs, charts_dir, ds_label)

    # Cat 4: Dimensionality Reduction
    print("\n  --- Category 4: t-SNE / PCA ---")
    fig04a_pca_scatter_severity(combined, charts_dir, ds_label)
    fig04b_pca_scatter_link(combined, charts_dir, ds_label)
    fig04c_tsne_severity(combined, charts_dir, ds_label)
    fig04d_tsne_link(combined, charts_dir, ds_label)
    fig04e_pca_explained_variance(combined, charts_dir, ds_label)

    print(f"\n  Done! All charts in: {charts_dir}")


def main():
    parser = argparse.ArgumentParser(description='IEEE Charts for State Vectors')
    parser.add_argument('--dataset', default='all', choices=['ciciot2023', 'diad2024', 'all'])
    args = parser.parse_args()

    ds_list = ['ciciot2023', 'diad2024'] if args.dataset == 'all' else [args.dataset]
    for ds in ds_list:
        generate_dataset_charts(ds)

    # Cross-dataset
    if len(ds_list) == 2:
        print("\n  --- Cross-Dataset Comparison ---")
        cross_dir = BASE / 'experiments' / 'state_vectors_cross'
        os.makedirs(cross_dir, exist_ok=True)
        fig05_cross_dataset(cross_dir)
        print(f"  Cross charts in: {cross_dir}")

    print("\n  All done!")


if __name__ == '__main__':
    main()
