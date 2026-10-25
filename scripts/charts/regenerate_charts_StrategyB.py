#!/usr/bin/env python3
"""
FL-RLMAMS Strategy B — IEEE Academic Chart Generator
=====================================================
Reads cached results from Strategy B training and generates
38+ IEEE-quality standalone figures (NO subplots).

Usage:
    python scripts/regenerate_charts_StrategyB.py --dataset ciciot2023
    python scripts/regenerate_charts_StrategyB.py --dataset diad2024
    python scripts/regenerate_charts_StrategyB.py --dataset all
"""
import os, sys, json, argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde

# ── IEEE IEEEtran Style ────────────────────────────────────
# Matches IEEE Transactions column width (3.5in single, 7.16in double)
# Font: Times New Roman, sizes match IEEEtran.cls defaults
plt.rcParams.update({
    # ── Typography (IEEEtran) ──
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',  # STIX = Times-compatible math
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
    # ── Figure ──
    'figure.dpi': 150,
    'savefig.dpi': 600,           # IEEE requires 600 DPI minimum
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    # ── Axes ──
    'axes.grid': True,
    'axes.grid.which': 'major',
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    # ── Lines ──
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    # ── Ticks ──
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
})

# IEEE double-column figure: 7.16 × 4.5 inches
FIG_W, FIG_H = 7.16, 4.5
PHASE_COLORS = {
    'B1': '#2196F3', 'B2': '#4CAF50', 'B3': '#FF9800', 'B4': '#9C27B0', 'B5': '#F44336',
    'A_ind': '#78909C', 'A_fedavg': '#26A69A', 'A_fedprox': '#EF5350',
}
LINK_COLORS = ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A', '#C62828']
LINK_NAMES = ['Link_1_Core', 'Link_2_Aggregation_L', 'Link_3_Aggregation_R',
              'Link_4_Edge_L', 'Link_5_Edge_R']
LINK_SHORT = ['L1-Core', 'L2-AggrL', 'L3-AggrR', 'L4-EdgeL', 'L5-EdgeR']
ACTION_NAMES = ['ALLOW', 'RATE_LIMIT', 'BLOCK']
ACTION_COLORS = ['#4CAF50', '#FF9800', '#F44336']
SEVERITY_NAMES = ['High', 'Medium', 'Low']

_PROJECT = Path(__file__).resolve().parent.parent.parent
RESULTS_BASE = _PROJECT / 'experiments' / 'results_StrategyB'
STRATEGY_A_BASE = _PROJECT / 'experiments' / 'results_StrategyB'
DATA_DIR = _PROJECT / 'data'
WINDOW_SIZE = 20

# Data paths for loading raw CSV samples (needed for before/after charts)
DATA_CONFIG = {
    'ciciot2023': {
        'data_dir': DATA_DIR / 'unified_14features_5links' / 'ciciot2023',
        'json_file': DATA_DIR / 'unified_selected_14.json',
    },
    'diad2024': {
        'data_dir': DATA_DIR / 'unified_14features_5links' / 'diad2024',
        'json_file': DATA_DIR / 'unified_selected_14.json',
    },
}

# CSV Link filenames
LINK_CSV_NAMES = [
    'Link_1_Core', 'Link_2_Aggregation_L', 'Link_3_Aggregation_R',
    'Link_4_Edge_L', 'Link_5_Edge_R'
]


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def _load_raw_sample(ds_key, link_name, feature_names, n_rows=20000):
    """Load a small CSV sample and compute window means for 'before' view.
    Returns X_raw_mean (N_windows, 2) or None if CSV not available."""
    cfg = DATA_CONFIG.get(ds_key)
    if cfg is None:
        return None
    csv_path = cfg['data_dir'] / f'{link_name}.csv'
    if not csv_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=n_rows)
        available = [f for f in feature_names if f in df.columns]
        if len(available) < 2:
            return None
        X = df[available[:2]].values.astype(np.float32)
        # Create windows
        n_win = len(X) // WINDOW_SIZE
        if n_win < 10:
            return None
        X_trim = X[:n_win * WINDOW_SIZE].reshape(n_win, WINDOW_SIZE, 2)
        X_mean = X_trim.mean(axis=1)  # (n_win, 2) — raw before denoising
        return X_mean
    except Exception:
        return None


def save_fig(fig, path, chart_name):
    fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close(fig)
    print(f'    [OK] {chart_name}')


def get_feature_names(afras_dir, link_name):
    """Get actual top-2 feature names for a link from membership params."""
    p = afras_dir / link_name / 'phase5_membership_params.json'
    if p.exists():
        params = load_json(p)
        return list(params.keys())
    return ['feature_0', 'feature_1']


# ═══════════════════════════════════════════════════════════
#  SECTION 1: AFRAS-FL CHARTS (10 figures)
# ═══════════════════════════════════════════════════════════

def fig01_denoising(afras_dir, charts_dir, ds_label, link_name='Link_1_Core',
                    ds_key='ciciot2023'):
    """P1: EKF + Wavelet Denoising — Before vs After signal comparison.
    Academic value: proves noise reduction while preserving class-separable structure.
    Shows: (a) Raw signal time-series, (b) Denoised signal, (c) Noise reduction stats."""
    npz_path = afras_dir / link_name / 'phase1_denoised.npz'
    if not npz_path.exists():
        print(f'    [SKIP] fig01 — no phase1 data for {link_name}')
        return
    data = np.load(npz_path)
    X_clean = data['X_denoised']  # (N, 2)
    feat_names = get_feature_names(afras_dir, link_name)
    short = LINK_SHORT[LINK_NAMES.index(link_name)]

    # Try to load raw data for before/after comparison
    X_raw = _load_raw_sample(ds_key, link_name, feat_names, n_rows=20000)

    for f_idx in range(min(2, X_clean.shape[1])):
        fname = feat_names[f_idx] if f_idx < len(feat_names) else f'feature_{f_idx}'
        safe_ln = short.replace('-', '_')

        if X_raw is not None and f_idx < X_raw.shape[1]:
            # ── Chart A: Before vs After Signal (time-series style) ──
            n_show = min(200, len(X_raw), len(X_clean))
            raw_sig = X_raw[:n_show, f_idx]
            clean_sig = X_clean[:n_show, f_idx]
            t = np.arange(n_show)

            # Compute noise stats
            # Approximate: raw variance vs clean variance
            raw_std = np.std(X_raw[:, f_idx])
            clean_std = np.std(X_clean[:, f_idx])
            noise_reduction = max(0, (1 - clean_std / raw_std) * 100) if raw_std > 0 else 0

            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
            ax.plot(t, raw_sig, color='#F44336', alpha=0.6, linewidth=0.8,
                    label=f'Raw Signal ($\\sigma$={raw_std:.3f})')
            ax.plot(t, clean_sig, color='#1565C0', linewidth=1.5,
                    label=f'After EKF+Wavelet ($\\sigma$={clean_std:.3f})')

            # Highlight noise region
            ax.fill_between(t, raw_sig, clean_sig, alpha=0.15, color='#FF9800',
                            label='Removed Noise')

            ax.set_xlabel('Window Index (time steps)')
            ax.set_ylabel(f'{fname} (amplitude)')
            ax.set_title(f'Phase 1: Signal Enhancement — {fname} — {short} — {ds_label}\n'
                         f'Noise Reduction: {noise_reduction:.1f}% | '
                         f'N={len(X_clean):,} windows')
            ax.legend(loc='upper right', fontsize=8)
            save_fig(fig, charts_dir / f'fig01a_signal_{fname}_{safe_ln}.png',
                     f'fig01a_signal_{fname}_{safe_ln}')

        # ── Chart B: Distribution Before vs After (KDE) ──
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        clean_vals = X_clean[:, f_idx]

        # Denoised distribution
        try:
            kde_clean = gaussian_kde(clean_vals)
            x_range = np.linspace(clean_vals.min(), clean_vals.max(), 300)
            ax.fill_between(x_range, kde_clean(x_range), alpha=0.3, color='#1565C0')
            ax.plot(x_range, kde_clean(x_range), color='#1565C0', linewidth=2,
                    label='After EKF + Wavelet')
        except Exception:
            ax.hist(clean_vals, bins=100, density=True, alpha=0.5, color='#1565C0',
                    label='After EKF + Wavelet')

        # Raw distribution (if available)
        if X_raw is not None and f_idx < X_raw.shape[1]:
            raw_vals = X_raw[:, f_idx]
            try:
                kde_raw = gaussian_kde(raw_vals)
                x_range_raw = np.linspace(raw_vals.min(), raw_vals.max(), 300)
                ax.plot(x_range_raw, kde_raw(x_range_raw), color='#F44336',
                        linewidth=1.5, linestyle='--', alpha=0.7,
                        label='Raw (before denoising)')
            except Exception:
                pass

        ax.set_xlabel(f'{fname}')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Phase 1: Feature Distribution Before vs After — {fname} — {short} — {ds_label}\n'
                     f'Clean multimodal peaks indicate preserved class structure')
        ax.legend(fontsize=8)
        save_fig(fig, charts_dir / f'fig01b_dist_{fname}_{safe_ln}.png',
                 f'fig01b_dist_{fname}_{safe_ln}')


def fig02_nurbs_smote(afras_dir, charts_dir, ds_label, link_name='Link_1_Core'):
    """P2: NURBS-SMOTE — before/after balance + SMOTE vs NURBS-SMOTE density comparison.
    Academic value: proves geometric oversampling is superior to linear interpolation."""
    # ── Chart A: Class Balance Before vs After (all links) ──
    link_data = []
    for ln in LINK_NAMES:
        p1 = afras_dir / ln / 'phase1_denoised.npz'
        p2 = afras_dir / ln / 'phase2_balanced.npz'
        n_before = len(np.load(p1)['X_denoised']) if p1.exists() else 0
        entry = {'before': n_before, 'after_total': 0, 'classes': {}, 'n_classes': 0, 'ir': 1.0}
        if p2.exists():
            y = np.load(p2)['y_balanced']
            entry['after_total'] = len(y)
            unique, counts = np.unique(y, return_counts=True)
            entry['classes'] = dict(zip(unique.tolist(), counts.tolist()))
            entry['n_classes'] = len(unique)
            entry['ir'] = max(counts) / min(counts) if len(counts) > 1 else 1.0
        link_data.append(entry)

    fig, ax = plt.subplots(figsize=(10, FIG_H))
    x = np.arange(5)
    w = 0.30
    before_vals = [d['before'] for d in link_data]
    after_vals = [d['after_total'] for d in link_data]
    ax.bar(x - w/2, before_vals, w, label='Before NURBS-SMOTE (unbalanced)',
           color='#BDBDBD', edgecolor='black', linewidth=0.5, alpha=0.9, hatch='//')
    ax.bar(x + w/2, after_vals, w, label='After NURBS-SMOTE (balanced)',
           color='#42A5F5', edgecolor='black', linewidth=0.5, alpha=0.85)
    for i, d in enumerate(link_data):
        ax.text(x[i] - w/2, before_vals[i], f'{before_vals[i]:,}',
                ha='center', va='bottom', fontsize=7, color='#616161')
        growth = ((d['after_total'] - d['before']) / d['before'] * 100) if d['before'] > 0 else 0
        ax.text(x[i] + w/2, after_vals[i], f'{after_vals[i]:,}\n+{growth:.0f}%',
                ha='center', va='bottom', fontsize=7, fontweight='bold', color='#0D47A1')
        ir = d['ir']
        badge_color = '#2E7D32' if ir <= 1.01 else '#E65100'
        ax.text(x[i], max(before_vals[i], after_vals[i]) * 1.12,
                f'IR={ir:.2f} | {d["n_classes"]} classes',
                ha='center', va='bottom', fontsize=8, fontweight='bold', color=badge_color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9' if ir <= 1.01 else '#FFF3E0',
                          alpha=0.8, edgecolor=badge_color, linewidth=0.5))
    ax.set_xlabel('SDN Link (Zone)')
    ax.set_ylabel('Sample Count')
    ax.set_title(f'Phase 2: NURBS-SMOTE Geometric Oversampling — {ds_label}\n'
                 f'IR = Imbalance Ratio (1.00 = perfectly balanced)')
    ax.set_xticks(x)
    ax.set_xticklabels(LINK_SHORT)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, max(max(before_vals), max(after_vals)) * 1.25)
    save_fig(fig, charts_dir / 'fig02a_class_balance.png', 'fig02a_class_balance')

    # ── Chart B: SMOTE (linear) vs NURBS-SMOTE (geometric) density comparison ──
    # For each link: generate traditional SMOTE, compare density with NURBS-SMOTE
    for li, ln in enumerate(LINK_NAMES):
        p1 = afras_dir / ln / 'phase1_denoised.npz'
        p2 = afras_dir / ln / 'phase2_balanced.npz'
        if not (p1.exists() and p2.exists()):
            continue

        X_orig = np.load(p1)['X_denoised']
        nurbs_data = np.load(p2)
        X_nurbs = nurbs_data['X_balanced']
        y_nurbs = nurbs_data['y_balanced']
        feat_names = get_feature_names(afras_dir, ln)
        short = LINK_SHORT[li]

        # Identify synthetic samples (those beyond original count)
        n_orig = len(X_orig)
        if len(X_nurbs) <= n_orig:
            continue

        # Generate traditional SMOTE for comparison
        try:
            from sklearn.neighbors import NearestNeighbors
            # Get labels for original data
            # Reconstruct original y from the first n_orig entries of y_nurbs
            # (NURBS-SMOTE appends synthetic after original)
            y_orig = y_nurbs[:n_orig] if n_orig <= len(y_nurbs) else y_nurbs[:len(X_orig)]

            # Simple linear SMOTE: for each minority sample, interpolate with k-NN
            rng = np.random.RandomState(42)
            unique_cls, cls_counts = np.unique(y_orig, return_counts=True)
            max_count = cls_counts.max()
            X_smote_list = [X_orig.copy()]
            y_smote_list = [y_orig.copy()]

            for c, cnt in zip(unique_cls, cls_counts):
                if cnt >= max_count:
                    continue
                n_need = max_count - cnt
                X_c = X_orig[y_orig == c]
                if len(X_c) < 2:
                    continue
                nn = NearestNeighbors(n_neighbors=min(6, len(X_c))).fit(X_c)
                _, indices = nn.kneighbors(X_c)

                synth = []
                for _ in range(n_need):
                    idx = rng.randint(len(X_c))
                    nn_idx = rng.choice(indices[idx, 1:])
                    lam = rng.random()
                    # Linear interpolation (traditional SMOTE)
                    new_pt = X_c[idx] + lam * (X_c[nn_idx] - X_c[idx])
                    synth.append(new_pt)
                X_smote_list.append(np.array(synth, dtype=np.float32))
                y_smote_list.append(np.full(n_need, c, dtype=np.int64))

            X_smote = np.concatenate(X_smote_list)

            # Density comparison chart
            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

            # Only synthetic points for density
            X_nurbs_synth = X_nurbs[n_orig:]
            X_smote_synth = X_smote[n_orig:]

            n_sample = min(5000, len(X_nurbs_synth), len(X_smote_synth))
            if n_sample < 10:
                plt.close(fig)
                continue

            rng2 = np.random.RandomState(42)
            ns_idx = rng2.choice(len(X_nurbs_synth), n_sample, replace=False)
            ss_idx = rng2.choice(len(X_smote_synth), n_sample, replace=len(X_smote_synth) < n_sample)

            ax.scatter(X_smote_synth[ss_idx, 0], X_smote_synth[ss_idx, 1],
                       alpha=0.3, s=8, c='#FF9800', marker='s',
                       label=f'Traditional SMOTE (linear, N={len(X_smote_synth):,})')
            ax.scatter(X_nurbs_synth[ns_idx, 0], X_nurbs_synth[ns_idx, 1],
                       alpha=0.4, s=8, c='#4CAF50', marker='o',
                       label=f'NURBS-SMOTE (geometric, N={len(X_nurbs_synth):,})')

            # Original data as reference
            n_orig_show = min(2000, n_orig)
            orig_idx = rng2.choice(n_orig, n_orig_show, replace=False)
            ax.scatter(X_orig[orig_idx, 0], X_orig[orig_idx, 1],
                       alpha=0.15, s=4, c='#616161', marker='.',
                       label=f'Original (N={n_orig:,})')

            # Distance to nearest real sample (quality metric)
            nn_real = NearestNeighbors(n_neighbors=1).fit(X_orig)
            d_nurbs, _ = nn_real.kneighbors(X_nurbs_synth[:n_sample])
            d_smote, _ = nn_real.kneighbors(X_smote_synth[:n_sample])
            avg_d_nurbs = d_nurbs.mean()
            avg_d_smote = d_smote.mean()

            ax.text(0.02, 0.98,
                    f'Avg distance to nearest real sample:\n'
                    f'  SMOTE (linear): {avg_d_smote:.4f}\n'
                    f'  NURBS-SMOTE (geometric): {avg_d_nurbs:.4f}\n'
                    f'  {"NURBS closer (better)" if avg_d_nurbs < avg_d_smote else "SMOTE closer"}',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            ax.set_xlabel(feat_names[0] if feat_names else 'Feature 0')
            ax.set_ylabel(feat_names[1] if len(feat_names) > 1 else 'Feature 1')
            ax.set_title(f'Phase 2: SMOTE vs NURBS-SMOTE Synthetic Sample Density — {short} — {ds_label}\n'
                         f'Orange=Linear Interpolation | Green=Geometric B-Spline Curve')
            ax.legend(loc='lower right', fontsize=7, markerscale=2)

            safe_ln = short.replace('-', '_')
            save_fig(fig, charts_dir / f'fig02b_density_{safe_ln}.png', f'fig02b_density_{safe_ln}')

        except Exception as e:
            print(f'    [WARN] fig02b density for {ln}: {e}')


def fig03_boundary_detection(afras_dir, charts_dir, ds_label, link_name='Link_1_Core'):
    """P3: K-D Tree + R-Tree boundary detection — boundary vs interior + distance map.
    Academic value: proves spatial indexing identifies decision boundary regions."""
    npz_path = afras_dir / link_name / 'phase3_boundary.npz'
    bal_path = afras_dir / link_name / 'phase2_balanced.npz'
    if not npz_path.exists():
        print(f'    [SKIP] fig03 — no phase3 data')
        return
    data = np.load(npz_path)
    bp = data['boundary_points']  # (N_boundary, 2)
    bl = data['boundary_labels']  # (N_boundary,)
    feat_names = get_feature_names(afras_dir, link_name)
    short = LINK_SHORT[LINK_NAMES.index(link_name)]
    safe_ln = short.replace('-', '_')

    X_all, y_all = None, None
    if bal_path.exists():
        bal = np.load(bal_path)
        X_all = bal['X_balanced']
        y_all = bal['y_balanced']

    # ── Chart A: Boundary vs Interior Points (scatter) ──
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    rng = np.random.RandomState(42)

    if X_all is not None:
        class_colors = ['#F44336', '#FF9800', '#4CAF50']
        n_bg = min(5000, len(X_all))
        bg_idx = rng.choice(len(X_all), n_bg, replace=False)
        for c, name, color in zip([0, 1, 2], SEVERITY_NAMES, class_colors):
            mask = y_all[bg_idx] == c
            if mask.any():
                ax.scatter(X_all[bg_idx[mask], 0], X_all[bg_idx[mask], 1],
                           alpha=0.08, s=4, c=color, zorder=1)

    # Boundary points (highlighted with black edge)
    n = min(3000, len(bp))
    idx = rng.choice(len(bp), n, replace=False)
    for c, name, color in zip([0, 1, 2], SEVERITY_NAMES, ['#F44336', '#FF9800', '#4CAF50']):
        mask = bl[idx] == c
        if mask.any():
            ax.scatter(bp[idx[mask], 0], bp[idx[mask], 1], alpha=0.7, s=25,
                       c=color, label=f'Boundary: {name}',
                       edgecolors='black', linewidths=0.5, zorder=3)

    n_total = len(X_all) if X_all is not None else 0
    pct = (len(bp) / n_total * 100) if n_total > 0 else 0
    ax.text(0.02, 0.98,
            f'Boundary points: {len(bp):,} ({pct:.1f}% of total)\n'
            f'Method: K-D Tree neighbors + R-Tree spatial index\n'
            f'Boundary Ratio: {pct:.1f}% (lower = cleaner separation)',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel(feat_names[0])
    ax.set_ylabel(feat_names[1] if len(feat_names) > 1 else 'feature_1')
    ax.set_title(f'Phase 3: Boundary Detection (K-D Tree + R-Tree) — {short} — {ds_label}')
    ax.legend(markerscale=2, fontsize=8)
    save_fig(fig, charts_dir / f'fig03a_boundary_{safe_ln}.png', f'fig03a_boundary_{safe_ln}')

    # ── Chart B: Distance Map (heatmap) ──
    if X_all is not None and len(X_all) > 100:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

        # Build grid for distance map
        n_sample = min(50000, len(X_all))
        X_sub = X_all[rng.choice(len(X_all), n_sample, replace=False)]

        x_min, x_max = X_sub[:, 0].min(), X_sub[:, 0].max()
        y_min, y_max = X_sub[:, 1].min(), X_sub[:, 1].max()
        pad_x = (x_max - x_min) * 0.05
        pad_y = (y_max - y_min) * 0.05

        grid_res = 100
        xx, yy = np.meshgrid(
            np.linspace(x_min - pad_x, x_max + pad_x, grid_res),
            np.linspace(y_min - pad_y, y_max + pad_y, grid_res))
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

        # For each grid point, find distance to nearest point of each class
        # Distance map = min inter-class distance (low = boundary area)
        try:
            trees = {}
            for c in np.unique(y_all[:n_sample]):
                c_mask = y_all[:n_sample] == c
                if c_mask.sum() > 0:
                    trees[c] = KDTree(X_sub[c_mask[:n_sample] if len(c_mask) >= n_sample else c_mask[:len(X_sub)]])

            if len(trees) >= 2:
                # Use subset of X_all that matches X_sub
                y_sub = y_all[rng.choice(len(y_all), n_sample, replace=False)]
                trees_clean = {}
                for c in np.unique(y_sub):
                    c_mask = y_sub == c
                    if c_mask.sum() > 0:
                        trees_clean[c] = KDTree(X_sub[c_mask])

                # Distance difference between two nearest classes
                class_keys = sorted(trees_clean.keys())
                dist_vals = np.zeros(len(grid_pts))
                for i, pt in enumerate(grid_pts):
                    dists = []
                    for c in class_keys:
                        d, _ = trees_clean[c].query(pt.reshape(1, -1))
                        dists.append(d[0])
                    dists.sort()
                    # Small difference = near boundary
                    dist_vals[i] = abs(dists[1] - dists[0]) if len(dists) >= 2 else dists[0]

                Z = dist_vals.reshape(xx.shape)
                cmap_dist = LinearSegmentedColormap.from_list('dist', ['#FF6F00', '#FFF8E1', '#0D47A1'])
                im = ax.contourf(xx, yy, Z, levels=30, cmap=cmap_dist, alpha=0.8)
                fig.colorbar(im, ax=ax, label='Inter-class Distance (low = boundary region)')

                # Overlay boundary points
                n_bp = min(1000, len(bp))
                bp_idx = rng.choice(len(bp), n_bp, replace=False)
                ax.scatter(bp[bp_idx, 0], bp[bp_idx, 1], s=10, c='black',
                           alpha=0.5, marker='x', linewidths=0.5, label=f'Boundary ({len(bp):,})')

                ax.set_xlabel(feat_names[0])
                ax.set_ylabel(feat_names[1] if len(feat_names) > 1 else 'feature_1')
                ax.set_title(f'Phase 3: Distance Map Analysis — {short} — {ds_label}\n'
                             f'Warm regions = decision boundaries between classes')
                ax.legend(fontsize=8)
                save_fig(fig, charts_dir / f'fig03b_distmap_{safe_ln}.png',
                         f'fig03b_distmap_{safe_ln}')
            else:
                plt.close(fig)
        except Exception as e:
            print(f'    [WARN] fig03b distance map: {e}')
            plt.close(fig)

    # ── Chart C: Decision Regions (k-NN classification) ──
    if X_all is not None and len(X_all) > 100:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        try:
            n_fit = min(20000, len(X_all))
            fit_idx = rng.choice(len(X_all), n_fit, replace=False)
            X_fit = X_all[fit_idx]
            y_fit = y_all[fit_idx]

            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=5).fit(X_fit, y_fit)

            x_min, x_max = X_fit[:, 0].min(), X_fit[:, 0].max()
            y_min, y_max = X_fit[:, 1].min(), X_fit[:, 1].max()
            pad_x = (x_max - x_min) * 0.05
            pad_y = (y_max - y_min) * 0.05

            grid_res = 150
            xx, yy = np.meshgrid(
                np.linspace(x_min - pad_x, x_max + pad_x, grid_res),
                np.linspace(y_min - pad_y, y_max + pad_y, grid_res))
            Z = knn.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

            # Color map for 3 classes
            from matplotlib.colors import ListedColormap
            region_cmap = ListedColormap(['#FFCDD2', '#FFE0B2', '#C8E6C9'])  # light: red, orange, green
            ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], cmap=region_cmap, alpha=0.6)

            # Data points
            n_show = min(3000, len(X_fit))
            show_idx = rng.choice(len(X_fit), n_show, replace=False)
            class_colors = ['#D32F2F', '#E65100', '#2E7D32']
            for c, name, color in zip([0, 1, 2], SEVERITY_NAMES, class_colors):
                mask = y_fit[show_idx] == c
                if mask.any():
                    ax.scatter(X_fit[show_idx[mask], 0], X_fit[show_idx[mask], 1],
                               s=6, c=color, alpha=0.5, label=f'{name}')

            # Boundary overlay
            n_bp = min(500, len(bp))
            bp_idx = rng.choice(len(bp), n_bp, replace=False)
            ax.scatter(bp[bp_idx, 0], bp[bp_idx, 1], s=20, facecolors='none',
                       edgecolors='black', linewidths=0.8, label='Boundary', zorder=4)

            ax.set_xlabel(feat_names[0])
            ax.set_ylabel(feat_names[1] if len(feat_names) > 1 else 'feature_1')
            ax.set_title(f'Phase 3: Decision Regions (k-NN) — {short} — {ds_label}\n'
                         f'Red=High | Orange=Medium | Green=Low | Circles=Boundary')
            ax.legend(fontsize=8, markerscale=2)
            save_fig(fig, charts_dir / f'fig03c_regions_{safe_ln}.png',
                     f'fig03c_regions_{safe_ln}')
        except Exception as e:
            print(f'    [WARN] fig03c decision regions: {e}')
            plt.close(fig)


def fig04_adaptive_grid(afras_dir, charts_dir, ds_label, link_name='Link_1_Core'):
    """P4: Adaptive grid — cells with class coloring + actual data points overlay.
    Academic value: proves adaptive refinement focuses on ambiguous boundary regions."""
    grid_path = afras_dir / link_name / 'phase4_grid.json'
    bal_path = afras_dir / link_name / 'phase2_balanced.npz'
    if not grid_path.exists():
        print(f'    [SKIP] fig04 — no grid data')
        return
    cells = load_json(grid_path)
    feat_names = get_feature_names(afras_dir, link_name)
    short = LINK_SHORT[LINK_NAMES.index(link_name)]
    safe_ln = short.replace('-', '_')
    rng = np.random.RandomState(42)

    # Load actual data points
    X_all, y_all = None, None
    if bal_path.exists():
        bal = np.load(bal_path)
        X_all = bal['X_balanced']
        y_all = bal['y_balanced']

    # Class colors (translucent for cell fill, solid for points)
    class_fill = {0: '#FFCDD2', 1: '#FFE0B2', 2: '#C8E6C9'}  # light red, orange, green
    class_point = {0: '#D32F2F', 1: '#E65100', 2: '#2E7D32'}  # dark red, orange, green

    # ── Chart A: Grid with Data Points (like plots/04) ──
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    n_fine = 0
    n_coarse = 0

    for cell in cells:
        x0, x1 = cell['x_range']
        y0, y1 = cell['y_range']
        mc = cell['majority_class']
        p = cell['purity']
        is_fine = cell.get('is_fine', False)

        # Cell fill: class color with alpha based on purity
        fill_color = class_fill.get(mc, '#E0E0E0')
        alpha = 0.3 + 0.5 * p  # higher purity = more opaque

        ec = '#D32F2F' if is_fine else '#9E9E9E'
        lw = 1.0 if is_fine else 0.3

        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              facecolor=fill_color, edgecolor=ec,
                              linewidth=lw, alpha=alpha)
        ax.add_patch(rect)

        if is_fine:
            n_fine += 1
        else:
            n_coarse += 1

    # Overlay actual data points
    if X_all is not None:
        n_show = min(5000, len(X_all))
        show_idx = rng.choice(len(X_all), n_show, replace=False)
        for c, name, color in zip([0, 1, 2], SEVERITY_NAMES, ['#D32F2F', '#E65100', '#2E7D32']):
            mask = y_all[show_idx] == c
            if mask.any():
                ax.scatter(X_all[show_idx[mask], 0], X_all[show_idx[mask], 1],
                           s=3, c=color, alpha=0.4, label=f'{name} ({mask.sum():,})',
                           zorder=2)

    # Stats
    purities = [c['purity'] for c in cells]
    n_pts_total = sum(c['n_points'] for c in cells)
    ax.text(0.02, 0.98,
            f'Grid: {len(cells)} cells (coarse={n_coarse}, fine={n_fine})\n'
            f'Mean purity: {np.mean(purities):.3f}\n'
            f'Total classified: {n_pts_total:,}\n'
            f'Fine cells = boundary refinement zones',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel(feat_names[0])
    ax.set_ylabel(feat_names[1] if len(feat_names) > 1 else 'feature_1')
    ax.set_title(f'Phase 4: Adaptive Grid Classification — {short} — {ds_label}\n'
                 f'Red=High (BLOCK) | Orange=Medium (RATE_LIMIT) | Green=Low (ALLOW)')
    ax.legend(fontsize=8, markerscale=3, loc='lower right')
    ax.autoscale()
    save_fig(fig, charts_dir / f'fig04a_grid_{safe_ln}.png', f'fig04a_grid_{safe_ln}')

    # ── Chart B: Purity Heatmap (continuous) ──
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    purity_cmap = plt.cm.RdYlGn

    for cell in cells:
        x0, x1 = cell['x_range']
        y0, y1 = cell['y_range']
        p = cell['purity']
        is_fine = cell.get('is_fine', False)
        ec = '#D32F2F' if is_fine else 'gray'
        lw = 0.8 if is_fine else 0.3

        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              facecolor=purity_cmap(p), edgecolor=ec,
                              linewidth=lw, alpha=0.8)
        ax.add_patch(rect)

        # Label cells with enough points
        width = x1 - x0
        if cell['n_points'] > 50 and width > (cells[0]['x_range'][1] - cells[0]['x_range'][0]) * 0.3:
            mc = cell['majority_class']
            label = {0: 'H', 1: 'M', 2: 'L'}.get(mc, '?')
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, f'{label}\n{p:.0%}',
                    ha='center', va='center', fontsize=6, fontweight='bold', color='black')

    sm = plt.cm.ScalarMappable(cmap=purity_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Cell Purity (1.0 = single class)')

    legend_elements = [
        mpatches.Patch(edgecolor='gray', facecolor='none', linestyle='--',
                       linewidth=0.5, label=f'Coarse cell ({n_coarse})'),
        mpatches.Patch(edgecolor='#D32F2F', facecolor='none', linestyle='-',
                       linewidth=1.0, label=f'Fine cell ({n_fine})'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    ax.set_xlabel(feat_names[0])
    ax.set_ylabel(feat_names[1] if len(feat_names) > 1 else 'feature_1')
    ax.set_title(f'Phase 4: Cell Purity Heatmap — {short} — {ds_label}\n'
                 f'Green=pure (one class) | Red=mixed (boundary)')
    ax.autoscale()
    save_fig(fig, charts_dir / f'fig04b_purity_{safe_ln}.png', f'fig04b_purity_{safe_ln}')


def fig05_membership_functions(afras_dir, charts_dir, ds_label, link_name='Link_1_Core'):
    """P5: Gaussian membership functions WITH data histogram overlay.
    Academic value: proves membership functions align with actual data distribution."""
    params_path = afras_dir / link_name / 'phase5_membership_params.json'
    bal_path = afras_dir / link_name / 'phase2_balanced.npz'
    if not params_path.exists():
        print(f'    [SKIP] fig05 — no membership params')
        return
    params = load_json(params_path)
    short = LINK_SHORT[LINK_NAMES.index(link_name)]
    colors_mf = {'Low': '#4CAF50', 'Medium': '#FF9800', 'High': '#F44336'}

    # Load data for histogram overlay
    X_data = None
    if bal_path.exists():
        X_data = np.load(bal_path)['X_balanced']

    for f_idx, (feat_name, feat_params) in enumerate(params.items()):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        lo, hi = feat_params['range']
        x = np.linspace(lo - 1, hi + 1, 500)

        # Data histogram (background)
        if X_data is not None and f_idx < X_data.shape[1]:
            ax2 = ax.twinx()
            ax2.hist(X_data[:, f_idx], bins=80, density=True, alpha=0.15,
                     color='#616161', edgecolor='none', label='Data distribution')
            ax2.set_ylabel('Data Density', color='#9E9E9E', fontsize=9)
            ax2.tick_params(axis='y', labelcolor='#9E9E9E')
            ax2.set_ylim(bottom=0)

        # Membership functions (foreground)
        for level in ['Low', 'Medium', 'High']:
            c = feat_params[level]['center']
            w = feat_params[level]['width']
            sigma = w / (2 * np.sqrt(2 * np.log(2)))
            y = np.exp(-0.5 * ((x - c) / max(sigma, 1e-8)) ** 2)
            ax.plot(x, y, color=colors_mf[level], linewidth=2.5, label=f'{level} (c={c:.2f})')
            ax.fill_between(x, y, alpha=0.12, color=colors_mf[level])
            ax.axvline(c, color=colors_mf[level], linestyle='--', alpha=0.3, linewidth=0.8)

        ax.set_xlabel(feat_name)
        ax.set_ylabel('Membership Degree μ(x)')
        ax.set_title(f'Phase 5: Fuzzy Membership Functions — {feat_name} — {short} — {ds_label}')
        ax.set_ylim(-0.05, 1.15)
        ax.legend(loc='upper right', fontsize=8)
        safe_name = feat_name.replace('/', '_')
        safe_ln = short.replace('-', '_')
        save_fig(fig, charts_dir / f'fig05_membership_{safe_name}_{safe_ln}.png',
                 f'fig05_mf_{safe_name}_{safe_ln}')


def fig06_rule_confidence_support(afras_dir, charts_dir, ds_label):
    """P5b: Rule confidence × support bubble chart across all links."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    all_plotted = False
    for i, link_name in enumerate(LINK_NAMES):
        rules_path = afras_dir / link_name / 'phase5_rules.json'
        if not rules_path.exists():
            continue
        rules = load_json(rules_path)
        for r in rules:
            size = max(10, min(300, r['support'] / 500))
            color = ACTION_COLORS[r['consequent_action']]
            ax.scatter(r['confidence'], r['support'], s=size, c=color,
                       alpha=0.6, edgecolors='black', linewidths=0.5)
            all_plotted = True

    if not all_plotted:
        print('    [SKIP] fig06 — no rules')
        plt.close(fig)
        return

    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=c, markersize=10, label=n)
                       for c, n in zip(ACTION_COLORS, ACTION_NAMES)]
    ax.legend(handles=legend_elements)
    ax.set_xlabel('Rule Confidence')
    ax.set_ylabel('Rule Support (samples)')
    ax.set_title(f'Phase 5: Fuzzy Rule Confidence vs Support — {ds_label}')
    save_fig(fig, charts_dir / 'fig06_rule_confidence_support.png', 'fig06_rules')


def fig07_rule_priority(afras_dir, charts_dir, ds_label, link_name='Link_1_Core'):
    """P6: Organized rules with priority, confidence, and support annotations.
    Academic value: proves rule ranking balances confidence and coverage."""
    org_path = afras_dir / link_name / 'phase6_organized.json'
    if not org_path.exists():
        print(f'    [SKIP] fig07 — no organized rules')
        return
    rules = load_json(org_path)
    short = LINK_SHORT[LINK_NAMES.index(link_name)]

    fig, ax = plt.subplots(figsize=(10, max(4, len(rules) * 0.6 + 1)))
    labels = []
    priorities = []
    colors = []
    for r in rules:
        ant_str = ' ∧ '.join([f'{a[0]}={a[1]}' for a in r['antecedents']])
        labels.append(f"R{r['rank']}: IF {ant_str} → {r['consequent']}")
        priorities.append(r['priority'])
        colors.append(ACTION_COLORS[r['consequent_action']])

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, priorities, color=colors, edgecolor='black',
                   linewidth=0.3, alpha=0.8, height=0.6)

    # Annotate confidence and support on each bar
    for i, (r, b) in enumerate(zip(rules, bars)):
        ax.text(b.get_width() + 0.1, i,
                f"conf={r['confidence']:.3f} | sup={r['support']:,}",
                va='center', fontsize=7, color='#424242')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Priority Score = log(support) × confidence')
    ax.set_title(f'Phase 6: Rule Organization & Priority — {short} — {ds_label}\n'
                 f'{len(rules)} rules extracted | Higher priority = more reliable & frequent')
    ax.invert_yaxis()
    legend_elements = [mpatches.Patch(facecolor=c, label=n)
                       for c, n in zip(ACTION_COLORS, ACTION_NAMES)]
    ax.legend(handles=legend_elements, loc='lower right')
    safe_ln = short.replace('-', '_')
    save_fig(fig, charts_dir / f'fig07_priority_{safe_ln}.png', f'fig07_priority_{safe_ln}')


def fig08_rules_per_link_heatmap(main_results, charts_dir, ds_label):
    """P7: Rules per link — action type heatmap."""
    b0 = main_results.get('phase_b0', {})
    if not b0:
        print('    [SKIP] fig08 — no AFRAS data')
        return

    data_matrix = np.zeros((5, 3))
    for i, ln in enumerate(LINK_NAMES):
        if ln in b0:
            data_matrix[i, 0] = b0[ln].get('allow_rules', 0)
            data_matrix[i, 1] = b0[ln].get('limit_rules', 0)
            data_matrix[i, 2] = b0[ln].get('block_rules', 0)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_yticks(range(5))
    ax.set_yticklabels(LINK_SHORT)
    for i in range(5):
        for j in range(3):
            val = int(data_matrix[i, j])
            ax.text(j, i, str(val), ha='center', va='center',
                    color='white' if val > data_matrix.max() / 2 else 'black', fontsize=11)
    fig.colorbar(im, ax=ax, label='Number of Rules')
    ax.set_title(f'AFRAS-FL: Rules per Link by Action Type — {ds_label}')
    save_fig(fig, charts_dir / 'fig08_rules_heatmap.png', 'fig08_rules_heatmap')


def fig08b_xai_decision_surface(afras_dir, charts_dir, ds_label, link_name='Link_1_Core'):
    """XAI: Feature-Action Decision Surface — shows how 2 features determine actions.
    This is the core interpretability visualization for IEEE papers."""
    params_path = afras_dir / link_name / 'phase5_membership_params.json'
    rules_path = afras_dir / link_name / 'phase6_organized.json'
    if not params_path.exists() or not rules_path.exists():
        print(f'    [SKIP] fig08b — no data for {link_name}')
        return

    params = load_json(params_path)
    rules = load_json(rules_path)
    feat_names = list(params.keys())
    if len(feat_names) < 2:
        print(f'    [SKIP] fig08b — need 2 features')
        return

    f1_name, f2_name = feat_names[0], feat_names[1]
    f1_range = params[f1_name]['range']
    f2_range = params[f2_name]['range']

    # Create grid
    res = 200
    x1 = np.linspace(f1_range[0], f1_range[1], res)
    x2 = np.linspace(f2_range[0], f2_range[1], res)
    X1, X2 = np.meshgrid(x1, x2)

    def gaussian_mf(x, center, width):
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-0.5 * ((x - center) / max(sigma, 1e-8)) ** 2)

    # For each grid point, find the best-matching rule
    decision_map = np.full((res, res), -1, dtype=int)
    confidence_map = np.zeros((res, res))

    for r in rules:
        ant_dict = {a[0]: a[1] for a in r['antecedents']}
        level1 = ant_dict.get(f1_name)
        level2 = ant_dict.get(f2_name)
        if level1 is None or level2 is None:
            continue

        c1 = params[f1_name][level1]['center']
        w1 = params[f1_name][level1]['width']
        c2 = params[f2_name][level2]['center']
        w2 = params[f2_name][level2]['width']

        mu1 = gaussian_mf(X1, c1, w1)
        mu2 = gaussian_mf(X2, c2, w2)
        activation = mu1 * mu2 * r['confidence']

        better = activation > confidence_map
        decision_map[better] = r['consequent_action']
        confidence_map[better] = activation[better]

    # Color map: ALLOW=green, RATE_LIMIT=orange, BLOCK=red
    from matplotlib.colors import ListedColormap
    cmap_actions = ListedColormap(['#E8F5E9', '#FFF3E0', '#FFEBEE'])  # light versions
    cmap_overlay = ListedColormap(['#4CAF50', '#FF9800', '#F44336'])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    # Background: decision regions
    masked = np.ma.masked_where(decision_map < 0, decision_map)
    ax.pcolormesh(X1, X2, masked, cmap=cmap_actions, alpha=0.7, shading='auto')

    # Overlay: confidence contours
    ax.contour(X1, X2, confidence_map, levels=5, colors='gray', linewidths=0.5, alpha=0.5)

    # Mark membership function centers
    levels = ['Low', 'Medium', 'High']
    markers = ['v', 's', '^']
    for lvl, mkr in zip(levels, markers):
        c1 = params[f1_name][lvl]['center']
        c2 = params[f2_name][lvl]['center']
        ax.plot(c1, c2, mkr, color='black', markersize=10, markeredgewidth=1.5,
                markerfacecolor='white', label=f'Center: ({lvl},{lvl})', zorder=5)

    # Rule annotations
    for r in rules[:5]:  # top 5 rules
        ant_dict = {a[0]: a[1] for a in r['antecedents']}
        l1 = ant_dict.get(f1_name)
        l2 = ant_dict.get(f2_name)
        if l1 and l2:
            cx = params[f1_name][l1]['center']
            cy = params[f2_name][l2]['center']
            ax.annotate(f"R{r['rank']}:{r['consequent']}", (cx, cy),
                        fontsize=7, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                        ha='center', va='bottom', xytext=(0, 8), textcoords='offset points')

    legend_elements = [mpatches.Patch(facecolor=c, edgecolor='black', label=n)
                       for c, n in zip(['#4CAF50', '#FF9800', '#F44336'], ACTION_NAMES)]
    ax.legend(handles=legend_elements, loc='upper right', title='Decision Region')
    ax.set_xlabel(f1_name)
    ax.set_ylabel(f2_name)
    short = LINK_SHORT[LINK_NAMES.index(link_name)]
    ax.set_title(f'XAI: Feature-Action Decision Surface — {short} — {ds_label}')
    safe_ln = short.replace('-', '_')
    save_fig(fig, charts_dir / f'fig08b_decision_{safe_ln}.png', f'fig08b_decision_{safe_ln}')


def fig08c_xai_membership_across_zones(afras_dir, charts_dir, ds_label):
    """XAI: How the same feature has different membership functions across zones.
    Shows zone-specific adaptation — key interpretability contribution."""
    # Find a shared feature across links
    shared_features = {}
    for ln in LINK_NAMES:
        p = afras_dir / ln / 'phase5_membership_params.json'
        if p.exists():
            params = load_json(p)
            for f in params.keys():
                if f not in shared_features:
                    shared_features[f] = []
                shared_features[f].append((ln, params[f]))

    # Pick the most shared feature
    best_feat = max(shared_features.keys(), key=lambda f: len(shared_features[f]))
    entries = shared_features[best_feat]

    if len(entries) < 2:
        print('    [SKIP] fig08c — not enough zones share a feature')
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    # Get global range
    all_lo = min(e[1]['range'][0] for e in entries)
    all_hi = max(e[1]['range'][1] for e in entries)
    x = np.linspace(all_lo - 1, all_hi + 1, 500)

    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    for idx, (ln, fp) in enumerate(entries):
        short = LINK_SHORT[LINK_NAMES.index(ln)]
        # Plot High membership only (most relevant for attack detection)
        c = fp['High']['center']
        w = fp['High']['width']
        sigma = w / (2 * np.sqrt(2 * np.log(2)))
        y = np.exp(-0.5 * ((x - c) / max(sigma, 1e-8)) ** 2)
        ax.plot(x, y, linestyle=linestyles[idx % len(linestyles)],
                color=LINK_COLORS[idx], linewidth=2, label=f'{short} (High)')

    ax.set_xlabel(f'{best_feat} (normalized)')
    ax.set_ylabel('Membership Degree μ(x)')
    ax.set_title(f'XAI: Zone-Specific Membership Adaptation — "{best_feat}" — {ds_label}')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(title='Link (Zone)', fontsize=8)
    save_fig(fig, charts_dir / 'fig08c_xai_zone_membership.png', 'fig08c_zone_membership')


def fig08d_xai_rule_interpretability(afras_dir, charts_dir, ds_label):
    """XAI: Rule interpretability summary — academic table-style figure.
    Shows IF-THEN rules with confidence and support for each link."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')

    # Build table data
    headers = ['Link', 'Rule', 'IF Condition', 'THEN Action', 'Conf.', 'Support']
    table_data = []
    for ln in LINK_NAMES:
        org_path = afras_dir / ln / 'phase6_organized.json'
        if not org_path.exists():
            continue
        rules = load_json(org_path)
        short = LINK_SHORT[LINK_NAMES.index(ln)]
        for r in rules[:3]:  # top 3 per link
            ant_str = ' ∧ '.join([f'{a[0]}={a[1]}' for a in r['antecedents']])
            table_data.append([
                short if r == rules[0] else '',
                f"R{r['rank']}",
                ant_str,
                r['consequent'],
                f"{r['confidence']:.3f}",
                f"{r['support']:,}"
            ])

    if not table_data:
        plt.close(fig)
        print('    [SKIP] fig08d — no rules')
        return

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center', colWidths=[0.08, 0.05, 0.35, 0.12, 0.08, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Color action cells
    action_colors = {'BLOCK': '#FFCDD2', 'RATE_LIMIT': '#FFF9C4', 'ALLOW': '#C8E6C9'}
    for i, row in enumerate(table_data):
        action = row[3]
        if action in action_colors:
            table[i + 1, 3].set_facecolor(action_colors[action])

    ax.set_title(f'XAI: Interpretable Fuzzy Rules Summary — {ds_label}',
                 fontsize=13, fontweight='bold', pad=20)
    save_fig(fig, charts_dir / 'fig08d_xai_rules_table.png', 'fig08d_rules_table')


def fig08e_xai_feature_space_scatter(afras_dir, charts_dir, ds_label, link_name='Link_1_Core'):
    """XAI: Scatter of balanced data in 2D feature space with rule boundaries.
    Shows how fuzzy regions partition the feature space."""
    balanced_path = afras_dir / link_name / 'phase2_balanced.npz'
    params_path = afras_dir / link_name / 'phase5_membership_params.json'
    if not balanced_path.exists() or not params_path.exists():
        print(f'    [SKIP] fig08e — no data for {link_name}')
        return

    data = np.load(balanced_path)
    X = data['X_balanced']
    y = data['y_balanced']
    params = load_json(params_path)
    feat_names = list(params.keys())

    n = min(5000, len(X))
    idx = np.random.RandomState(42).choice(len(X), n, replace=False)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Scatter by class
    for c, name, color in zip([0, 1, 2], SEVERITY_NAMES, ['#F44336', '#FF9800', '#4CAF50']):
        mask = y[idx] == c
        if mask.any():
            ax.scatter(X[idx[mask], 0], X[idx[mask], 1], alpha=0.2, s=6,
                       c=color, label=name)

    # Draw membership function boundaries (center lines)
    for feat_idx, fn in enumerate(feat_names[:2]):
        for level in ['Low', 'Medium', 'High']:
            c = params[fn][level]['center']
            if feat_idx == 0:
                ax.axvline(c, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
                ax.text(c, ax.get_ylim()[1], f'{level}', ha='center', va='bottom',
                        fontsize=7, color='gray', fontstyle='italic')
            else:
                ax.axhline(c, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
                ax.text(ax.get_xlim()[1], c, f'{level}', ha='right', va='bottom',
                        fontsize=7, color='gray', fontstyle='italic')

    ax.legend(markerscale=3, title='True Severity')
    ax.set_xlabel(feat_names[0])
    ax.set_ylabel(feat_names[1])
    short = LINK_SHORT[LINK_NAMES.index(link_name)]
    ax.set_title(f'XAI: Feature Space with Fuzzy Boundaries — {short} — {ds_label}')
    safe_ln = short.replace('-', '_')
    save_fig(fig, charts_dir / f'fig08e_feature_space_{safe_ln}.png', f'fig08e_space_{safe_ln}')


def fig09_afras_rules_count(main_results, charts_dir, ds_label):
    """AFRAS: Total rules per link grouped bar."""
    b0 = main_results.get('phase_b0', {})
    if not b0:
        print('    [SKIP] fig09')
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(5)
    w = 0.25
    allow_vals = [b0.get(ln, {}).get('allow_rules', 0) for ln in LINK_NAMES]
    limit_vals = [b0.get(ln, {}).get('limit_rules', 0) for ln in LINK_NAMES]
    block_vals = [b0.get(ln, {}).get('block_rules', 0) for ln in LINK_NAMES]

    ax.bar(x - w, block_vals, w, label='BLOCK', color=ACTION_COLORS[2], alpha=0.8)
    ax.bar(x, limit_vals, w, label='RATE_LIMIT', color=ACTION_COLORS[1], alpha=0.8)
    ax.bar(x + w, allow_vals, w, label='ALLOW', color=ACTION_COLORS[0], alpha=0.8)
    ax.set_xlabel('Link')
    ax.set_ylabel('Number of Rules')
    ax.set_title(f'AFRAS-FL: Rules Count per Link — {ds_label}')
    ax.set_xticks(x)
    ax.set_xticklabels(LINK_SHORT)
    ax.legend()
    save_fig(fig, charts_dir / 'fig09_afras_rules_count.png', 'fig09_rules_count')


def fig10_afras_confidence(main_results, charts_dir, ds_label):
    """AFRAS: Average confidence per link bar chart."""
    b0 = main_results.get('phase_b0', {})
    if not b0:
        print('    [SKIP] fig10')
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    confs = [b0.get(ln, {}).get('avg_confidence', 0) for ln in LINK_NAMES]
    bars = ax.bar(LINK_SHORT, confs, color=LINK_COLORS, alpha=0.8, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, confs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f'{v:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Average Rule Confidence')
    ax.set_title(f'AFRAS-FL: Rule Confidence per Link — {ds_label}')
    ax.set_ylim(0, 1.1)
    save_fig(fig, charts_dir / 'fig10_afras_confidence.png', 'fig10_confidence')


# ═══════════════════════════════════════════════════════════
#  SECTION 2: TRAINING PHASE CHARTS (12 figures)
# ═══════════════════════════════════════════════════════════

def fig11_mosar_breakdown(phase_results_dir, charts_dir, ds_label, phase_name='mosar_independent'):
    """MOSAR reward decomposition — stacked bar per link."""
    res_path = phase_results_dir / 'results.json'
    if not res_path.exists():
        print(f'    [SKIP] fig11 — no {phase_name} results')
        return
    res = load_json(res_path)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    components = ['R_detection', 'R_accuracy', 'R_latency', 'R_confidence', 'R_impact']
    comp_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    comp_labels = ['Detection', 'Accuracy', 'Latency', 'Confidence', 'Impact']

    x = np.arange(5)
    bottom = np.zeros(5)
    for comp, color, label in zip(components, comp_colors, comp_labels):
        vals = []
        for ln in LINK_NAMES:
            if ln in res and 'mosar_components' in res[ln]:
                v = res[ln]['mosar_components'].get(comp, 0)
                vals.append(max(0, v))  # clip negatives for stacking
            else:
                vals.append(0)
        ax.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.8, width=0.6)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('MOSAR Reward Component')
    ax.set_title(f'MOSAR Reward Decomposition — {ds_label}')
    ax.legend(loc='upper right')
    save_fig(fig, charts_dir / 'fig11_mosar_breakdown.png', 'fig11_mosar')


def fig12_b1_msr_per_link(phase_dir, charts_dir, ds_label):
    """Phase B1: MSR per link bar chart."""
    res_path = phase_dir / 'results.json'
    if not res_path.exists():
        print('    [SKIP] fig12')
        return
    res = load_json(res_path)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    msrs = [res.get(ln, {}).get('msr', 0) for ln in LINK_NAMES]
    bars = ax.bar(LINK_SHORT, msrs, color=LINK_COLORS, alpha=0.8, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, msrs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.001, f'{v:.4f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Mitigation Success Rate')
    ax.set_title(f'Phase B1 (Independent MOSAR-DQN): MSR per Link — {ds_label}')
    ax.set_ylim(min(msrs) - 0.02, 1.005)
    save_fig(fig, charts_dir / 'fig12_b1_msr.png', 'fig12_b1_msr')


def _plot_convergence(phase_dir, charts_dir, ds_label, phase_label, fig_name, color):
    """Generic convergence plot for FL phases."""
    res_path = phase_dir / 'results.json'
    if not res_path.exists():
        print(f'    [SKIP] {fig_name}')
        return
    res = load_json(res_path)

    history = res.get('history', {})
    global_msr = history.get('global_msr', [])
    if not global_msr:
        print(f'    [SKIP] {fig_name} — no global_msr history')
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    rounds = range(1, len(global_msr) + 1)
    ax.plot(rounds, global_msr, 'o-', color=color, linewidth=2, markersize=4, label='Global MSR')

    # Per-agent lines
    per_agent = history.get('per_agent', {})
    for agent_id in sorted(per_agent.keys(), key=lambda x: int(x)):
        agent_msr = per_agent[agent_id].get('msr', [])
        if agent_msr:
            ax.plot(rounds[:len(agent_msr)], agent_msr, '--', alpha=0.4,
                    color=LINK_COLORS[int(agent_id)], linewidth=1,
                    label=LINK_SHORT[int(agent_id)])

    ax.set_xlabel('FL Round')
    ax.set_ylabel('Mitigation Success Rate')
    ax.set_title(f'{phase_label} Convergence — {ds_label}')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    save_fig(fig, charts_dir / f'{fig_name}.png', fig_name)


def fig13_fedavg_convergence(ds_dir, charts_dir, ds_label):
    _plot_convergence(ds_dir / 'mosar_fedavg', charts_dir, ds_label,
                      'Phase B2 (FedAvg)', 'fig13_fedavg_convergence', PHASE_COLORS['B2'])


def fig14_fedprox_convergence(ds_dir, charts_dir, ds_label):
    _plot_convergence(ds_dir / 'mosar_fedprox', charts_dir, ds_label,
                      'Phase B3 (FedProx)', 'fig14_fedprox_convergence', PHASE_COLORS['B3'])


def fig15_fuzzy_convergence(ds_dir, charts_dir, ds_label):
    _plot_convergence(ds_dir / 'fuzzy_retrain', charts_dir, ds_label,
                      'Phase B4 (Fuzzy-Augmented)', 'fig15_fuzzy_convergence', PHASE_COLORS['B4'])


def fig16_phase_comparison_msr(main_results, charts_dir, ds_label):
    """Phase comparison: MSR grouped bar."""
    phases = ['phase_b1', 'phase_b2', 'phase_b3', 'phase_b4']
    labels = ['B1\n(Independent)', 'B2\n(FedAvg)', 'B3\n(FedProx)', 'B4\n(Fuzzy)']
    colors = [PHASE_COLORS['B1'], PHASE_COLORS['B2'], PHASE_COLORS['B3'], PHASE_COLORS['B4']]

    msrs = [main_results.get(p, {}).get('mean_msr', 0) for p in phases]
    stds = [main_results.get(p, {}).get('std_msr', 0) for p in phases]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax.bar(labels, msrs, yerr=stds, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5, capsize=5)
    for b, v in zip(bars, msrs):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003, f'{v:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Mean MSR')
    ax.set_title(f'Phase Comparison: Mitigation Success Rate — {ds_label}')
    ax.set_ylim(min(msrs) - 0.05, 1.02)
    save_fig(fig, charts_dir / 'fig16_phase_msr.png', 'fig16_phase_msr')


def fig17_phase_comparison_fpr(main_results, charts_dir, ds_label):
    """Phase comparison: FPR grouped bar."""
    phases = ['phase_b1', 'phase_b2', 'phase_b3', 'phase_b4']
    labels = ['B1', 'B2', 'B3', 'B4']
    colors = [PHASE_COLORS['B1'], PHASE_COLORS['B2'], PHASE_COLORS['B3'], PHASE_COLORS['B4']]

    fprs = [main_results.get(p, {}).get('mean_fpr', 0) for p in phases]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax.bar(labels, fprs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, fprs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.001, f'{v:.5f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('False Positive Rate')
    ax.set_title(f'Phase Comparison: FPR — {ds_label}')
    save_fig(fig, charts_dir / 'fig17_phase_fpr.png', 'fig17_phase_fpr')


def fig18_phase_comparison_reward(main_results, charts_dir, ds_label):
    """Phase comparison: Mean Reward."""
    phases = ['phase_b1', 'phase_b2', 'phase_b3', 'phase_b4']
    labels = ['B1', 'B2', 'B3', 'B4']
    colors = [PHASE_COLORS['B1'], PHASE_COLORS['B2'], PHASE_COLORS['B3'], PHASE_COLORS['B4']]
    rewards = [main_results.get(p, {}).get('mean_reward', 0) for p in phases]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax.bar(labels, rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, rewards):
        ax.text(b.get_x() + b.get_width()/2, v + 0.002, f'{v:.4f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Mean MOSAR Reward')
    ax.set_title(f'Phase Comparison: MOSAR Reward — {ds_label}')
    save_fig(fig, charts_dir / 'fig18_phase_reward.png', 'fig18_phase_reward')


def fig19_phase_comparison_jfi(main_results, charts_dir, ds_label):
    """Phase comparison: Jain Fairness Index."""
    phases = ['phase_b1', 'phase_b2', 'phase_b3', 'phase_b4']
    labels = ['B1', 'B2', 'B3', 'B4']
    colors = [PHASE_COLORS['B1'], PHASE_COLORS['B2'], PHASE_COLORS['B3'], PHASE_COLORS['B4']]
    jfis = [main_results.get(p, {}).get('jain_fairness', 0) for p in phases]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax.bar(labels, jfis, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, jfis):
        ax.text(b.get_x() + b.get_width()/2, v + 0.001, f'{v:.4f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Jain Fairness Index')
    ax.set_title(f'Phase Comparison: Fairness — {ds_label}')
    ax.set_ylim(min(jfis) - 0.02, 1.005)
    save_fig(fig, charts_dir / 'fig19_phase_jfi.png', 'fig19_phase_jfi')


def fig20_convergence_overlay(ds_dir, charts_dir, ds_label):
    """Overlay: B2 + B3 + B4 convergence on same plot."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    plotted = False
    for phase_sub, label, color in [
        ('mosar_fedavg', 'B2 (FedAvg)', PHASE_COLORS['B2']),
        ('mosar_fedprox', 'B3 (FedProx)', PHASE_COLORS['B3']),
        ('fuzzy_retrain', 'B4 (Fuzzy)', PHASE_COLORS['B4']),
    ]:
        res_path = ds_dir / phase_sub / 'results.json'
        if not res_path.exists():
            continue
        res = load_json(res_path)
        gmsr = res.get('history', {}).get('global_msr', [])
        if gmsr:
            ax.plot(range(1, len(gmsr)+1), gmsr, 'o-', color=color,
                    linewidth=2, markersize=4, label=label)
            plotted = True

    if not plotted:
        plt.close(fig)
        print('    [SKIP] fig20 — no convergence data')
        return
    ax.set_xlabel('FL Round')
    ax.set_ylabel('Global MSR')
    ax.set_title(f'FL Convergence Comparison — {ds_label}')
    ax.legend()
    ax.set_ylim(0, 1.05)
    save_fig(fig, charts_dir / 'fig20_convergence_overlay.png', 'fig20_convergence')


def fig21_per_agent_heatmap(ds_dir, charts_dir, ds_label, phase_sub='mosar_fedprox'):
    """Per-agent MSR heatmap across 30 rounds."""
    res_path = ds_dir / phase_sub / 'results.json'
    if not res_path.exists():
        print('    [SKIP] fig21')
        return
    res = load_json(res_path)
    per_agent = res.get('history', {}).get('per_agent', {})
    if not per_agent:
        print('    [SKIP] fig21 — no per_agent data')
        return

    n_agents = len(per_agent)
    n_rounds = len(next(iter(per_agent.values())).get('msr', []))
    matrix = np.zeros((n_agents, n_rounds))
    for agent_id in sorted(per_agent.keys(), key=lambda x: int(x)):
        msr_arr = per_agent[agent_id].get('msr', [])
        matrix[int(agent_id), :len(msr_arr)] = msr_arr

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('FL Round')
    ax.set_ylabel('Agent (Link)')
    ax.set_yticks(range(5))
    ax.set_yticklabels(LINK_SHORT)
    ax.set_xticks(range(0, n_rounds, 5))
    ax.set_xticklabels(range(1, n_rounds + 1, 5))
    fig.colorbar(im, ax=ax, label='MSR')
    phase_label = 'FedProx' if 'prox' in phase_sub else 'FedAvg'
    ax.set_title(f'Per-Agent MSR Heatmap ({phase_label}) — {ds_label}')
    save_fig(fig, charts_dir / 'fig21_agent_heatmap.png', 'fig21_heatmap')


def fig22_confusion_matrix(ds_dir, charts_dir, ds_label, phase_sub='mosar_independent'):
    """Severity-Action confusion matrix (3×3 heatmap)."""
    res_path = ds_dir / phase_sub / 'results.json'
    if not res_path.exists():
        print('    [SKIP] fig22')
        return
    res = load_json(res_path)

    # Aggregate across all links
    total_matrix = np.zeros((3, 3))
    for ln in LINK_NAMES:
        if ln in res and 'severity_action_matrix' in res[ln]:
            mat = np.array(res[ln]['severity_action_matrix'])
            if mat.shape == (3, 3):
                total_matrix += mat

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    # Normalize per row
    row_sums = total_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_matrix = total_matrix / row_sums

    im = ax.imshow(norm_matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(SEVERITY_NAMES)
    ax.set_xlabel('Agent Action (Predicted)')
    ax.set_ylabel('True Severity')
    for i in range(3):
        for j in range(3):
            count = int(total_matrix[i, j])
            pct = norm_matrix[i, j] * 100
            ax.text(j, i, f'{count:,}\n({pct:.1f}%)', ha='center', va='center',
                    color='white' if norm_matrix[i, j] > 0.5 else 'black', fontsize=9)
    fig.colorbar(im, ax=ax, label='Proportion')
    ax.set_title(f'Severity-Action Confusion Matrix — {ds_label}')
    save_fig(fig, charts_dir / 'fig22_confusion_matrix.png', 'fig22_confusion')


# ═══════════════════════════════════════════════════════════
#  SECTION 3: MASTER AGENT CHARTS (6 figures)
# ═══════════════════════════════════════════════════════════

def fig23_health_scores(main_results, charts_dir, ds_label):
    """Master Agent: Health score per link with error bars."""
    ma = main_results.get('master_agent', {})
    if not ma:
        print('    [SKIP] fig23')
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    means = ma['health_mean']
    stds = ma['health_std']
    bars = ax.bar(LINK_SHORT, means, yerr=stds, color=LINK_COLORS, alpha=0.8,
                  edgecolor='black', linewidth=0.5, capsize=5)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, v + max(stds) + 0.003, f'{v:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Health Score')
    ax.set_title(f'Master Agent: Link Health Scores — {ds_label}')
    ax.set_ylim(min(means) - 0.05, max(means) + 0.05)
    save_fig(fig, charts_dir / 'fig23_health_scores.png', 'fig23_health')


def fig24_health_heatmap(main_results, charts_dir, ds_label):
    """Master Agent: Health score heatmap (50 episodes × 5 links)."""
    ma = main_results.get('master_agent', {})
    all_health = ma.get('all_health', [])
    if not all_health:
        print('    [SKIP] fig24 — no all_health data')
        return

    matrix = np.array(all_health)  # (50, 5)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    im = ax.imshow(matrix.T, cmap='RdYlGn', aspect='auto',
                   vmin=matrix.min() - 0.01, vmax=matrix.max() + 0.01)
    ax.set_xlabel('Evaluation Episode')
    ax.set_ylabel('Link')
    ax.set_yticks(range(5))
    ax.set_yticklabels(LINK_SHORT)
    fig.colorbar(im, ax=ax, label='Health Score')
    ax.set_title(f'Master Agent: Health Score Heatmap — {ds_label}')
    save_fig(fig, charts_dir / 'fig24_health_heatmap.png', 'fig24_heatmap')


def fig25_jfi_trajectory(main_results, charts_dir, ds_label):
    """Master Agent: JFI trajectory over episodes."""
    ma = main_results.get('master_agent', {})
    jfi_before = ma.get('all_jfi_before', [])
    jfi_after = ma.get('all_jfi_after', [])
    if not jfi_before:
        print('    [SKIP] fig25')
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    eps = range(1, len(jfi_before) + 1)
    ax.plot(eps, jfi_before, 'o-', color='#1565C0', linewidth=1.5, markersize=3,
            label=f'JFI (mean={np.mean(jfi_before):.4f})')
    ax.axhline(y=np.mean(jfi_before), color='#F44336', linestyle='--', alpha=0.7,
               label=f'Mean = {np.mean(jfi_before):.4f}')
    ax.set_xlabel('Evaluation Episode')
    ax.set_ylabel('Jain Fairness Index')
    ax.set_title(f'Master Agent: JFI Trajectory — {ds_label}')
    ax.legend()
    ax.set_ylim(min(jfi_before) - 0.01, 1.001)
    save_fig(fig, charts_dir / 'fig25_jfi_trajectory.png', 'fig25_jfi')


def fig26_routing_probabilities(main_results, charts_dir, ds_label):
    """Master Agent: Routing probabilities pie chart."""
    ma = main_results.get('master_agent', {})
    rp = ma.get('routing_prob_mean', {})
    if not rp:
        print('    [SKIP] fig26')
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    labels = list(rp.keys())
    values = list(rp.values())
    explode = [0.05 if v == max(values) else 0 for v in values]
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                       colors=LINK_COLORS, explode=explode,
                                       startangle=90, pctdistance=0.85)
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    ax.set_title(f'Master Agent: Routing Probabilities — {ds_label}')
    save_fig(fig, charts_dir / 'fig26_routing_probs.png', 'fig26_routing')


def fig27_best_link_frequency(main_results, charts_dir, ds_label):
    """Master Agent: Best link frequency bar chart."""
    ma = main_results.get('master_agent', {})
    freq = ma.get('best_link_frequency', {})
    if not freq:
        print('    [SKIP] fig27')
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    # Ensure all links appear
    all_freq = {ls: freq.get(ls, 0) for ls in LINK_SHORT}
    bars = ax.bar(all_freq.keys(), all_freq.values(), color=LINK_COLORS,
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x() + b.get_width()/2, h + 0.5, f'{int(h)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Times Selected as Best')
    ax.set_title(f'Master Agent: Best Link Selection Frequency — {ds_label}')
    save_fig(fig, charts_dir / 'fig27_best_link.png', 'fig27_best_link')


def fig28_utilization(main_results, charts_dir, ds_label):
    """Master Agent: Utilization before/after."""
    ma = main_results.get('master_agent', {})
    last = ma.get('last_result', {})
    u_before = last.get('utilization_before', [])
    u_after = last.get('utilization_after', [])
    if not u_before:
        print('    [SKIP] fig28')
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(5)
    w = 0.35
    ax.bar(x - w/2, u_before, w, label='Before', color='#EF5350', alpha=0.8)
    ax.bar(x + w/2, u_after, w, label='After', color='#42A5F5', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('Utilization')
    ax.set_title(f'Master Agent: Link Utilization — {ds_label}')
    ax.legend()
    save_fig(fig, charts_dir / 'fig28_utilization.png', 'fig28_utilization')


# ═══════════════════════════════════════════════════════════
#  SECTION 4: CROSS-DATASET CHARTS (5 figures)
# ═══════════════════════════════════════════════════════════

def load_both_datasets():
    """Load results from both datasets if available."""
    results = {}
    for ds in ['ciciot2023', 'diad2024']:
        path = RESULTS_BASE / ds / 'results.json'
        if path.exists():
            results[ds] = load_json(path)
    return results


def fig29_cross_msr_with_errorbars(charts_dir):
    """Cross-dataset: MSR with error bars — proves generalization."""
    both = load_both_datasets()
    if len(both) < 2:
        print('    [SKIP] fig29 — need both datasets')
        return

    phases = ['phase_b1', 'phase_b2', 'phase_b3', 'phase_b4']
    labels = ['B1\n(MOSAR-DQN)', 'B2\n(FedAvg)', 'B3\n(FedProx)', 'B4\n(Fuzzy-Aug.)']
    x = np.arange(len(phases))
    w = 0.35

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for idx, (ds, ds_label, color, hatch) in enumerate([
        ('ciciot2023', 'CICIoT2023', '#1565C0', ''),
        ('diad2024', 'DIAD2024', '#E65100', '//'),
    ]):
        msrs = [both[ds].get(p, {}).get('mean_msr', 0) for p in phases]
        stds = [both[ds].get(p, {}).get('std_msr', 0) for p in phases]
        offset = -w/2 + idx * w
        bars = ax.bar(x + offset, msrs, w, yerr=stds, label=ds_label,
                      color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
                      capsize=4, hatch=hatch)
        for b, v, s in zip(bars, msrs, stds):
            ax.text(b.get_x() + b.get_width()/2, v + s + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mitigation Success Rate (MSR)')
    ax.set_title('Cross-Dataset Generalization: MSR Across Training Phases')
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 1.08)
    ax.axhline(y=0.95, color='red', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(3.5, 0.952, 'Target: 0.95', fontsize=8, color='red', alpha=0.6)
    save_fig(fig, charts_dir / 'fig29_cross_msr.png', 'fig29_cross_msr')


def fig30_cross_multi_metric(charts_dir):
    """Cross-dataset: Multi-metric comparison (MSR, 1-FPR, 1-FNR, JFI) — proves robustness."""
    both = load_both_datasets()
    if len(both) < 2:
        print('    [SKIP] fig30')
        return

    # Use best phase per dataset (B1 for CICIoT, B1 for DIAD)
    metrics = ['mean_msr', 'mean_fpr', 'mean_fnr', 'jain_fairness']
    metric_labels = ['MSR', '1 − FPR\n(Specificity)', '1 − FNR\n(Sensitivity)', 'Fairness\n(JFI)']
    best_phase = 'phase_b1'

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(metrics))
    w = 0.35

    for idx, (ds, ds_label, color, hatch) in enumerate([
        ('ciciot2023', 'CICIoT2023', '#1565C0', ''),
        ('diad2024', 'DIAD2024', '#E65100', '//'),
    ]):
        p = both[ds].get(best_phase, {})
        vals = [
            p.get('mean_msr', 0),
            1.0 - p.get('mean_fpr', 0),
            1.0 - p.get('mean_fnr', 0),
            p.get('jain_fairness', 0),
        ]
        offset = -w/2 + idx * w
        bars = ax.bar(x + offset, vals, w, label=ds_label, color=color,
                      alpha=0.8, edgecolor='black', linewidth=0.5, hatch=hatch)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.003, f'{v:.4f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Score (higher is better)')
    ax.set_title(f'Cross-Dataset: Multi-Metric Security Assessment (Best Phase: {best_phase.replace("phase_", "").upper()})')
    ax.legend(framealpha=0.9)
    ax.set_ylim(0.95, 1.005)
    save_fig(fig, charts_dir / 'fig30_cross_multi_metric.png', 'fig30_multi_metric')


def fig31_cross_health_radar(charts_dir):
    """Cross-dataset: Health score radar — proves zone-adaptive performance."""
    both = load_both_datasets()
    if len(both) < 2:
        print('    [SKIP] fig31')
        return

    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), subplot_kw=dict(polar=True))
    for ds, ds_label, color, ls in [
        ('ciciot2023', 'CICIoT2023', '#1565C0', '-'),
        ('diad2024', 'DIAD2024', '#E65100', '--'),
    ]:
        ma = both[ds].get('master_agent', {})
        vals = ma.get('health_mean', [0]*5)
        stds = ma.get('health_std', [0]*5)
        vals_closed = vals + vals[:1]
        stds_closed = stds + stds[:1]
        ax.plot(angles, vals_closed, 'o-', color=color, linewidth=2,
                linestyle=ls, label=f'{ds_label} (μ={np.mean(vals):.3f})')
        ax.fill(angles, vals_closed, alpha=0.1, color=color)
        # Error band
        upper = [v + s for v, s in zip(vals_closed, stds_closed)]
        lower = [v - s for v, s in zip(vals_closed, stds_closed)]
        ax.fill_between(angles, lower, upper, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LINK_SHORT, fontsize=9)
    ax.set_title('Cross-Dataset: Zone-Adaptive Link Health Profile', pad=20)
    ax.legend(loc='lower right', fontsize=8)
    save_fig(fig, charts_dir / 'fig31_cross_health_radar.png', 'fig31_radar')


def fig32_cross_rules_structure(charts_dir):
    """Cross-dataset: AFRAS-FL rule structure comparison — proves domain adaptation."""
    both = load_both_datasets()
    if len(both) < 2:
        print('    [SKIP] fig32')
        return

    x = np.arange(5)
    w = 0.15

    fig, ax = plt.subplots(figsize=(10, FIG_H))
    for ds_idx, (ds, ds_label, base_color) in enumerate([
        ('ciciot2023', 'CIC', '#1565C0'),
        ('diad2024', 'DIAD', '#E65100'),
    ]):
        b0 = both[ds].get('phase_b0', {})
        for act_idx, (act, act_color_suffix) in enumerate([
            ('block_rules', 0.9), ('limit_rules', 0.6), ('allow_rules', 0.3)
        ]):
            vals = [b0.get(ln, {}).get(act, 0) for ln in LINK_NAMES]
            offset = (ds_idx * 3 + act_idx - 2.5) * w
            alpha = act_color_suffix
            label_str = f'{ds_label}-{act.split("_")[0].upper()}'
            c = base_color if act_idx == 0 else (
                '#2196F3' if ds_idx == 0 else '#FF9800') if act_idx == 1 else (
                '#90CAF9' if ds_idx == 0 else '#FFE0B2')
            ax.bar(x + offset, vals, w, label=label_str, color=c,
                   edgecolor='black', linewidth=0.3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('Number of Fuzzy Rules')
    ax.set_title('Cross-Dataset: AFRAS-FL Rule Structure Adaptation')
    ax.legend(ncol=3, fontsize=7, loc='upper right')
    save_fig(fig, charts_dir / 'fig32_cross_rules_structure.png', 'fig32_rules_struct')


def fig33_cross_convergence_comparison(charts_dir):
    """Cross-dataset: FL convergence overlay — proves training stability."""
    both_data = {}
    for ds in ['ciciot2023', 'diad2024']:
        for phase_sub, phase_label in [
            ('mosar_fedavg', 'FedAvg'), ('mosar_fedprox', 'FedProx'), ('fuzzy_retrain', 'Fuzzy')
        ]:
            path = RESULTS_BASE / ds / phase_sub / 'results.json'
            if path.exists():
                res = load_json(path)
                gmsr = res.get('history', {}).get('global_msr', [])
                if gmsr:
                    both_data[(ds, phase_label)] = gmsr

    if len(both_data) < 4:
        print('    [SKIP] fig33 — insufficient convergence data')
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ds_styles = {'ciciot2023': '-', 'diad2024': '--'}
    ds_labels = {'ciciot2023': 'CIC', 'diad2024': 'DIAD'}
    phase_colors = {'FedAvg': '#4CAF50', 'FedProx': '#FF9800', 'Fuzzy': '#9C27B0'}

    for (ds, phase), gmsr in sorted(both_data.items()):
        rounds = range(1, len(gmsr) + 1)
        ax.plot(rounds, gmsr, linestyle=ds_styles[ds], color=phase_colors[phase],
                linewidth=2 if ds == 'ciciot2023' else 1.5, alpha=0.9,
                label=f'{ds_labels[ds]}-{phase}')

    ax.set_xlabel('Federated Learning Round')
    ax.set_ylabel('Global MSR')
    ax.set_title('Cross-Dataset: FL Convergence Stability Analysis')
    ax.legend(ncol=2, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.95, color='red', linestyle=':', alpha=0.3)
    save_fig(fig, charts_dir / 'fig33_cross_convergence.png', 'fig33_convergence')


def fig33b_cross_dataset_summary_table(charts_dir):
    """Cross-dataset: Academic summary table figure — for thesis defense."""
    both = load_both_datasets()
    if len(both) < 2:
        print('    [SKIP] fig33b')
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    phases = ['phase_b1', 'phase_b2', 'phase_b3', 'phase_b4']
    phase_labels = ['B1 (MOSAR-DQN)', 'B2 (FedAvg)', 'B3 (FedProx)', 'B4 (Fuzzy-Aug.)']

    headers = ['Phase', 'MSR (CIC)', 'MSR (DIAD)', 'FPR (CIC)', 'FPR (DIAD)',
               'JFI (CIC)', 'JFI (DIAD)', 'Reward (CIC)', 'Reward (DIAD)']
    rows = []
    for p, pl in zip(phases, phase_labels):
        row = [pl]
        for metric in ['mean_msr', 'mean_fpr', 'jain_fairness', 'mean_reward']:
            for ds in ['ciciot2023', 'diad2024']:
                v = both[ds].get(p, {}).get(metric, 0)
                fmt = f'{v:.4f}' if metric != 'mean_reward' else f'{v:.3f}'
                row.append(fmt)
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.14] + [0.095] * 8)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=7)

    # Highlight best MSR per dataset
    for i in range(len(rows)):
        for j in [1, 2]:  # MSR columns
            val = float(rows[i][j])
            if val >= 0.99:
                table[i + 1, j].set_facecolor('#C8E6C9')
            elif val < 0.90:
                table[i + 1, j].set_facecolor('#FFCDD2')

    ax.set_title('Cross-Dataset Performance Summary: Strategy B',
                 fontsize=13, fontweight='bold', pad=20)
    save_fig(fig, charts_dir / 'fig33b_cross_summary_table.png', 'fig33b_summary')


# ═══════════════════════════════════════════════════════════
#  SECTION 5: STRATEGY A vs B COMPARISON (5 figures)
# ═══════════════════════════════════════════════════════════

def _load_strategy_a(ds_key):
    path = STRATEGY_A_BASE / ds_key / 'results.json'
    if path.exists():
        return load_json(path)
    return None


def _get_ab_data(ds_key):
    """Load both Strategy A and B results for a dataset."""
    a_res = _load_strategy_a(ds_key)
    b_path = RESULTS_BASE / ds_key / 'results.json'
    b_res = load_json(b_path) if b_path.exists() else None
    return a_res, b_res


def fig34_avb_best_vs_best(ds_key, charts_dir, ds_label):
    """A vs B: Best phase comparison — the key proof of superiority.
    Compares best A (FedProx) vs best B (MOSAR-DQN) across all metrics."""
    a_res, b_res = _get_ab_data(ds_key)
    if not b_res:
        print('    [SKIP] fig34')
        return

    # Find best phase for each strategy
    a_phases = {'A-Ind': 'phase_a', 'A-FedAvg': 'phase_b', 'A-FedProx': 'phase_c'}
    b_phases = {'B-MOSAR': 'phase_b1', 'B-FedAvg': 'phase_b2', 'B-FedProx': 'phase_b3', 'B-Fuzzy': 'phase_b4'}

    best_a_label, best_a_msr = 'A-Ind', 0
    if a_res:
        for label, key in a_phases.items():
            msr = a_res.get(key, {}).get('mean_msr', 0)
            if msr > best_a_msr:
                best_a_msr = msr
                best_a_label = label

    best_b_label, best_b_msr = 'B-MOSAR', 0
    for label, key in b_phases.items():
        msr = b_res.get(key, {}).get('mean_msr', 0)
        if msr > best_b_msr:
            best_b_msr = msr
            best_b_label = label

    # Get full metrics for best phases
    a_best = a_res.get(a_phases[best_a_label], {}) if a_res else {}
    b_best = b_res.get(b_phases[best_b_label], {})

    metrics = ['MSR', 'Specificity\n(1−FPR)', 'Sensitivity\n(1−FNR)', 'Fairness\n(JFI)', 'Reward']
    a_vals = [
        a_best.get('mean_msr', 0),
        1 - a_best.get('mean_fpr', 0),
        1 - a_best.get('mean_fnr', 0),
        a_best.get('jain_fairness', 0),
        a_best.get('mean_reward', 0),
    ]
    b_vals = [
        b_best.get('mean_msr', 0),
        1 - b_best.get('mean_fpr', 0),
        1 - b_best.get('mean_fnr', 0),
        b_best.get('jain_fairness', 0),
        b_best.get('mean_reward', 0),
    ]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, FIG_H))
    bars_a = ax.bar(x - w/2, a_vals, w, label=f'Strategy A ({best_a_label})',
                    color='#78909C', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars_b = ax.bar(x + w/2, b_vals, w, label=f'Strategy B ({best_b_label})',
                    color='#1565C0', alpha=0.85, edgecolor='black', linewidth=0.5)

    # Annotate improvement
    for i, (av, bv) in enumerate(zip(a_vals, b_vals)):
        improvement = bv - av
        if a_vals[i] > 0:
            pct = (improvement / av) * 100
        else:
            pct = 0
        color = '#2E7D32' if improvement > 0 else '#C62828'
        arrow = '↑' if improvement > 0 else '↓'
        ax.text(x[i] + w/2, max(av, bv) + 0.008,
                f'{arrow}{abs(pct):.1f}%', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=color)
        # Values
        ax.text(x[i] - w/2, av + 0.002, f'{av:.4f}', ha='center', va='bottom', fontsize=7)
        ax.text(x[i] + w/2, bv + 0.002, f'{bv:.4f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score (higher is better)')
    ax.set_title(f'Strategy A vs B: Best Phase Comparison — {ds_label}')
    ax.legend(loc='lower right', framealpha=0.9)
    save_fig(fig, charts_dir / 'fig34_avb_best.png', 'fig34_avb_best')


def fig35_avb_msr_progression(ds_key, charts_dir, ds_label):
    """A vs B: MSR progression through phases — shows how B maintains advantage."""
    a_res, b_res = _get_ab_data(ds_key)
    if not b_res:
        print('    [SKIP] fig35')
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Strategy A progression
    if a_res:
        a_x = [1, 2, 3]
        a_labels = ['Independent', 'FedAvg', 'FedProx']
        a_msrs = [a_res.get(k, {}).get('mean_msr', 0) for k in ['phase_a', 'phase_b', 'phase_c']]
        a_stds = [a_res.get(k, {}).get('std_msr', 0) for k in ['phase_a', 'phase_b', 'phase_c']]
        ax.errorbar(a_x, a_msrs, yerr=a_stds, fmt='s--', color='#78909C',
                    linewidth=2, markersize=10, capsize=5, label='Strategy A', zorder=3)
        for xi, yi in zip(a_x, a_msrs):
            ax.text(xi, yi - 0.03, f'{yi:.3f}', ha='center', fontsize=8, color='#546E7A')

    # Strategy B progression
    b_x = [1, 2, 3, 4]
    b_labels = ['MOSAR-DQN', 'FedAvg', 'FedProx', 'Fuzzy-Aug.']
    b_msrs = [b_res.get(k, {}).get('mean_msr', 0)
              for k in ['phase_b1', 'phase_b2', 'phase_b3', 'phase_b4']]
    b_stds = [b_res.get(k, {}).get('std_msr', 0)
              for k in ['phase_b1', 'phase_b2', 'phase_b3', 'phase_b4']]
    ax.errorbar(b_x, b_msrs, yerr=b_stds, fmt='o-', color='#1565C0',
                linewidth=2.5, markersize=10, capsize=5, label='Strategy B', zorder=4)
    for xi, yi in zip(b_x, b_msrs):
        ax.text(xi, yi + 0.02, f'{yi:.3f}', ha='center', fontsize=8,
                fontweight='bold', color='#0D47A1')

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Phase 1\n(Independent)', 'Phase 2\n(FedAvg)',
                        'Phase 3\n(FedProx)', 'Phase 4\n(Fuzzy)'])
    ax.set_ylabel('Mean MSR')
    ax.set_title(f'Strategy A vs B: Training Phase Progression — {ds_label}')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0, 1.08)
    ax.axhline(y=0.95, color='red', linestyle=':', alpha=0.3)
    save_fig(fig, charts_dir / 'fig35_avb_progression.png', 'fig35_avb_progression')


def fig36_avb_reward_improvement(ds_key, charts_dir, ds_label):
    """A vs B: MOSAR vs simple reward — proves reward function superiority."""
    a_res, b_res = _get_ab_data(ds_key)
    if not b_res:
        print('    [SKIP] fig36')
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Best reward per link: Strategy A vs B
    a_ind_path = STRATEGY_A_BASE / ds_key / 'independent' / 'results.json'
    b_ind_path = RESULTS_BASE / ds_key / 'mosar_independent' / 'results.json'

    a_rewards, b_rewards = [], []
    for ln in LINK_NAMES:
        if a_ind_path.exists():
            a_data = load_json(a_ind_path)
            a_rewards.append(a_data.get(ln, {}).get('avg_reward', 0))
        else:
            a_rewards.append(0)
        if b_ind_path.exists():
            b_data = load_json(b_ind_path)
            b_rewards.append(b_data.get(ln, {}).get('avg_reward', 0))
        else:
            b_rewards.append(0)

    x = np.arange(5)
    w = 0.35
    bars_a = ax.bar(x - w/2, a_rewards, w, label='Strategy A (Simple Reward)',
                    color='#78909C', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars_b = ax.bar(x + w/2, b_rewards, w, label='Strategy B (MOSAR Reward)',
                    color='#1565C0', alpha=0.85, edgecolor='black', linewidth=0.5)

    # Improvement arrows
    for i, (a, b) in enumerate(zip(a_rewards, b_rewards)):
        if a > 0:
            pct = ((b - a) / abs(a)) * 100
            color = '#2E7D32' if pct > 0 else '#C62828'
            ax.annotate(f'{pct:+.0f}%', xy=(x[i], max(a, b) + 0.01),
                        ha='center', fontsize=8, fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('Average Reward per Link')
    ax.set_title(f'Reward Function Impact: Simple vs MOSAR — {ds_label}')
    ax.legend(loc='lower right')
    save_fig(fig, charts_dir / 'fig36_avb_reward.png', 'fig36_avb_reward')


def fig37_avb_health_improvement(ds_key, charts_dir, ds_label):
    """A vs B: Master Agent health — proves better load balancing."""
    a_res, b_res = _get_ab_data(ds_key)
    if not b_res:
        print('    [SKIP] fig37')
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(5)
    w = 0.35

    b_ma = b_res.get('master_agent', {})
    b_health = b_ma.get('health_mean', [0]*5)
    b_std = b_ma.get('health_std', [0]*5)

    ax.bar(x + w/2, b_health, w, yerr=b_std, label='Strategy B (MOSAR+Fuzzy)',
           color='#1565C0', alpha=0.85, edgecolor='black', linewidth=0.5, capsize=4)

    if a_res:
        a_ma = a_res.get('master_agent', {})
        a_health = a_ma.get('health_mean', [0]*5)
        a_std = a_ma.get('health_std', [0]*5)
        ax.bar(x - w/2, a_health, w, yerr=a_std, label='Strategy A (Simple)',
               color='#78909C', alpha=0.85, edgecolor='black', linewidth=0.5, capsize=4)

        # Improvement per link
        for i, (a, b) in enumerate(zip(a_health, b_health)):
            if a > 0:
                pct = ((b - a) / a) * 100
                color = '#2E7D32' if pct > 0 else '#C62828'
                ax.text(x[i], max(a, b) + max(a_std[i], b_std[i]) + 0.01,
                        f'{pct:+.1f}%', ha='center', fontsize=8,
                        fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('Link Health Score')
    ax.set_title(f'Master Agent: Health Score Improvement — {ds_label}')
    ax.legend(loc='lower right')
    save_fig(fig, charts_dir / 'fig37_avb_health.png', 'fig37_avb_health')


def fig38_avb_summary_table(ds_key, charts_dir, ds_label):
    """A vs B: Academic summary table with improvement percentages."""
    a_res, b_res = _get_ab_data(ds_key)
    if not b_res:
        print('    [SKIP] fig38')
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    headers = ['Metric', 'Strategy A\n(Best Phase)', 'Strategy B\n(Best Phase)', 'Improvement']

    # Find best phases
    a_best_msr, a_best_key = 0, 'phase_a'
    if a_res:
        for k in ['phase_a', 'phase_b', 'phase_c']:
            m = a_res.get(k, {}).get('mean_msr', 0)
            if m > a_best_msr:
                a_best_msr, a_best_key = m, k
    a_best = a_res.get(a_best_key, {}) if a_res else {}

    b_best_msr, b_best_key = 0, 'phase_b1'
    for k in ['phase_b1', 'phase_b2', 'phase_b3', 'phase_b4']:
        m = b_res.get(k, {}).get('mean_msr', 0)
        if m > b_best_msr:
            b_best_msr, b_best_key = m, k
    b_best = b_res.get(b_best_key, {})

    def imp(a, b, invert=False):
        if a == 0:
            return '—'
        diff = b - a
        if invert:
            diff = a - b
        pct = (diff / abs(a)) * 100
        return f'{"+" if pct >= 0 else ""}{pct:.1f}%'

    rows = [
        ['MSR', f"{a_best.get('mean_msr', 0):.4f}", f"{b_best.get('mean_msr', 0):.4f}",
         imp(a_best.get('mean_msr', 0), b_best.get('mean_msr', 0))],
        ['FPR (lower=better)', f"{a_best.get('mean_fpr', 0):.5f}", f"{b_best.get('mean_fpr', 0):.5f}",
         imp(a_best.get('mean_fpr', 0), b_best.get('mean_fpr', 0), invert=True)],
        ['FNR (lower=better)', f"{a_best.get('mean_fnr', 0):.5f}", f"{b_best.get('mean_fnr', 0):.5f}",
         imp(a_best.get('mean_fnr', 0), b_best.get('mean_fnr', 0), invert=True)],
        ['Jain Fairness', f"{a_best.get('jain_fairness', 0):.4f}", f"{b_best.get('jain_fairness', 0):.4f}",
         imp(a_best.get('jain_fairness', 0), b_best.get('jain_fairness', 0))],
        ['Reward', f"{a_best.get('mean_reward', 0):.4f}", f"{b_best.get('mean_reward', 0):.4f}",
         imp(a_best.get('mean_reward', 0), b_best.get('mean_reward', 0))],
    ]

    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.22, 0.22, 0.22, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    for j in range(len(headers)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Color improvement column
    for i, row in enumerate(rows):
        imp_val = row[3]
        if '+' in imp_val:
            table[i + 1, 3].set_facecolor('#C8E6C9')
            table[i + 1, 3].set_text_props(fontweight='bold', color='#2E7D32')
        elif imp_val != '—' and '-' in imp_val:
            table[i + 1, 3].set_facecolor('#FFCDD2')
            table[i + 1, 3].set_text_props(fontweight='bold', color='#C62828')

    a_phase_label = a_best_key.replace('phase_', '').upper()
    b_phase_label = b_best_key.replace('phase_', '').upper()
    ax.set_title(f'Strategy A ({a_phase_label}) vs B ({b_phase_label}): Quantitative Comparison — {ds_label}',
                 fontsize=13, fontweight='bold', pad=20)
    save_fig(fig, charts_dir / 'fig38_avb_summary.png', 'fig38_avb_summary')


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

DS_CONFIG = {
    'ciciot2023': {'label': 'CICIoT2023'},
    'diad2024': {'label': 'DIAD2024'},
}


def generate_dataset_charts(ds_key):
    ds_label = DS_CONFIG[ds_key]['label']
    ds_dir = RESULTS_BASE / ds_key
    base_charts = ds_dir / 'charts'
    afras_dir = ds_dir / 'afras_fl'

    main_path = ds_dir / 'results.json'
    if not main_path.exists():
        print(f'  [ERROR] No results.json for {ds_key}')
        return
    main_results = load_json(main_path)

    print(f'\n  Generating charts for {ds_label}...')
    print(f'  Output: {base_charts}')

    # ── Category 1: AFRAS-FL Pipeline (per-link) ──
    charts_dir = base_charts / '01_AFRAS_FL_Pipeline'
    os.makedirs(charts_dir, exist_ok=True)
    print(f'\n  --- 01_AFRAS_FL_Pipeline ---')
    for i, ln in enumerate(LINK_NAMES):
        short = LINK_SHORT[i]
        ln_dir = charts_dir / short.replace('-', '_')
        os.makedirs(ln_dir, exist_ok=True)
        fig01_denoising(afras_dir, ln_dir, ds_label, link_name=ln, ds_key=ds_key)
        fig03_boundary_detection(afras_dir, ln_dir, ds_label, link_name=ln)
        fig04_adaptive_grid(afras_dir, ln_dir, ds_label, link_name=ln)
        fig05_membership_functions(afras_dir, ln_dir, ds_label, link_name=ln)
        fig07_rule_priority(afras_dir, ln_dir, ds_label, link_name=ln)
    # Cross-link comparison charts
    fig02_nurbs_smote(afras_dir, charts_dir, ds_label)
    fig06_rule_confidence_support(afras_dir, charts_dir, ds_label)
    fig08_rules_per_link_heatmap(main_results, charts_dir, ds_label)
    fig09_afras_rules_count(main_results, charts_dir, ds_label)
    fig10_afras_confidence(main_results, charts_dir, ds_label)

    # ── Category 2: XAI Interpretability (per-link + comparison) ──
    charts_dir = base_charts / '02_XAI_Interpretability'
    os.makedirs(charts_dir, exist_ok=True)
    print(f'\n  --- 02_XAI_Interpretability ---')
    for i, ln in enumerate(LINK_NAMES):
        short = LINK_SHORT[i]
        ln_dir = charts_dir / short.replace('-', '_')
        os.makedirs(ln_dir, exist_ok=True)
        fig08b_xai_decision_surface(afras_dir, ln_dir, ds_label, link_name=ln)
        fig08e_xai_feature_space_scatter(afras_dir, ln_dir, ds_label, link_name=ln)
    # Cross-link comparison
    fig08c_xai_membership_across_zones(afras_dir, charts_dir, ds_label)
    fig08d_xai_rule_interpretability(afras_dir, charts_dir, ds_label)

    # ── Category 3: MOSAR-DQN Training ──
    charts_dir = base_charts / '03_MOSAR_DQN_Training'
    os.makedirs(charts_dir, exist_ok=True)
    print(f'\n  --- 03_MOSAR_DQN_Training ---')
    fig11_mosar_breakdown(ds_dir / 'mosar_independent', charts_dir, ds_label)
    fig12_b1_msr_per_link(ds_dir / 'mosar_independent', charts_dir, ds_label)
    fig22_confusion_matrix(ds_dir, charts_dir, ds_label)

    # ── Category 4: Federated Learning ──
    charts_dir = base_charts / '04_Federated_Learning'
    os.makedirs(charts_dir, exist_ok=True)
    print(f'\n  --- 04_Federated_Learning ---')
    fig13_fedavg_convergence(ds_dir, charts_dir, ds_label)
    fig14_fedprox_convergence(ds_dir, charts_dir, ds_label)
    fig15_fuzzy_convergence(ds_dir, charts_dir, ds_label)
    fig20_convergence_overlay(ds_dir, charts_dir, ds_label)
    fig21_per_agent_heatmap(ds_dir, charts_dir, ds_label)

    # ── Category 5: Phase Comparison ──
    charts_dir = base_charts / '05_Phase_Comparison'
    os.makedirs(charts_dir, exist_ok=True)
    print(f'\n  --- 05_Phase_Comparison ---')
    fig16_phase_comparison_msr(main_results, charts_dir, ds_label)
    fig17_phase_comparison_fpr(main_results, charts_dir, ds_label)
    fig18_phase_comparison_reward(main_results, charts_dir, ds_label)
    fig19_phase_comparison_jfi(main_results, charts_dir, ds_label)

    # ── Category 6: Master Agent ──
    charts_dir = base_charts / '06_Master_Agent'
    os.makedirs(charts_dir, exist_ok=True)
    print(f'\n  --- 06_Master_Agent ---')
    fig23_health_scores(main_results, charts_dir, ds_label)
    fig24_health_heatmap(main_results, charts_dir, ds_label)
    fig25_jfi_trajectory(main_results, charts_dir, ds_label)
    fig26_routing_probabilities(main_results, charts_dir, ds_label)
    fig27_best_link_frequency(main_results, charts_dir, ds_label)
    fig28_utilization(main_results, charts_dir, ds_label)

    # ── Category 7: Strategy A vs B ──
    charts_dir = base_charts / '07_Strategy_A_vs_B'
    os.makedirs(charts_dir, exist_ok=True)
    print(f'\n  --- 07_Strategy_A_vs_B ---')
    fig34_avb_best_vs_best(ds_key, charts_dir, ds_label)
    fig35_avb_msr_progression(ds_key, charts_dir, ds_label)
    fig36_avb_reward_improvement(ds_key, charts_dir, ds_label)
    fig37_avb_health_improvement(ds_key, charts_dir, ds_label)
    fig38_avb_summary_table(ds_key, charts_dir, ds_label)

    print(f'\n  Done! Charts saved to {base_charts}/')


def generate_cross_dataset_charts():
    """Generate cross-dataset comparison charts."""
    charts_dir = RESULTS_BASE / 'cross_dataset_charts' / '08_Cross_Dataset'
    os.makedirs(charts_dir, exist_ok=True)

    print('\n  --- 08_Cross_Dataset ---')
    fig29_cross_msr_with_errorbars(charts_dir)
    fig30_cross_multi_metric(charts_dir)
    fig31_cross_health_radar(charts_dir)
    fig32_cross_rules_structure(charts_dir)
    fig33_cross_convergence_comparison(charts_dir)
    fig33b_cross_dataset_summary_table(charts_dir)
    print(f'\n  Cross-dataset charts saved to {charts_dir}')


def main():
    parser = argparse.ArgumentParser(description='Strategy B IEEE Chart Generator')
    parser.add_argument('--dataset', default='all', choices=['ciciot2023', 'diad2024', 'all'])
    args = parser.parse_args()

    print('=' * 60)
    print('  FL-RLMAMS Strategy B — IEEE Chart Generator')
    print('=' * 60)

    if args.dataset == 'all':
        for ds in ['ciciot2023', 'diad2024']:
            if (RESULTS_BASE / ds / 'results.json').exists():
                generate_dataset_charts(ds)
        generate_cross_dataset_charts()
    else:
        generate_dataset_charts(args.dataset)

    print('\n' + '=' * 60)
    print('  All charts generated successfully!')
    print('=' * 60)


if __name__ == '__main__':
    main()
