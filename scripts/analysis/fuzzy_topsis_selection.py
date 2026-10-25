# -*- coding: utf-8 -*-
"""
============================================================================
FL-RLMAMS Stage 3: Fuzzy-TOPSIS Model Selection (Isolated per Dataset)
============================================================================
Applies Fuzzy-TOPSIS MCDM to select the optimal model for FL deployment
based on ISOLATED training results for each dataset independently.

3 Alternatives: LSTM-Only, Mamba-Only, CNN-Transformer
6 Criteria: F1-Macro, Accuracy, Minority Recall, Robustness, Speed, Efficiency
============================================================================
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================================
# Configuration
# ============================================================================
PROJECT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT / "experiments" / "results"
OUTPUT_DIR = RESULTS_DIR / "fuzzy_topsis"

VARIANT_NAMES = ['LSTM-Only', 'Mamba-Only', 'CNN-Transformer']

VARIANT_FAMILIES = {
    'LSTM-Only': 'RNN',
    'Mamba-Only': 'SSM',
    'CNN-Transformer': 'Transformer',
}

FAMILY_COLORS = {'RNN': '#3498DB', 'SSM': '#27AE60', 'Transformer': '#9B59B6'}

CRITERIA_NAMES = [
    'F1-Macro', 'Accuracy', 'Minority Recall',
    'Robustness', 'Inference Speed', 'Efficiency'
]

LINK_NAMES = ['Link_1_Core', 'Link_2_Aggregation_L', 'Link_3_Aggregation_R',
              'Link_4_Edge_L', 'Link_5_Edge_R']
LINK_SHORT = ['L1 Core', 'L2 Aggr_L', 'L3 Aggr_R', 'L4 Edge_L', 'L5 Edge_R']

DATASETS = {
    'ciciot2023': {
        'label': 'CICIoT2023',
        'results_file': RESULTS_DIR / 'ciciot2023' / 'results.json',
    },
    'diad2024': {
        'label': 'DIAD2024',
        'results_file': RESULTS_DIR / 'diad2024' / 'results.json',
    },
}

# Matplotlib academic style
plt.rcParams.update({
    'font.size': 12, 'font.family': 'serif',
    'axes.labelsize': 14, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ============================================================================
# Triangular Fuzzy Number (TFN)
# ============================================================================
class TFN:
    """Triangular Fuzzy Number (l, m, u)"""
    def __init__(self, l, m, u):
        self.l = float(l)
        self.m = float(m)
        self.u = float(u)

    def __repr__(self):
        return f'({self.l:.4f}, {self.m:.4f}, {self.u:.4f})'

    def __mul__(self, other):
        if isinstance(other, TFN):
            return TFN(self.l * other.l, self.m * other.m, self.u * other.u)
        return TFN(self.l * other, self.m * other, self.u * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, TFN):
            return TFN(self.l + other.l, self.m + other.m, self.u + other.u)
        return TFN(self.l + other, self.m + other, self.u + other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return TFN(self.l / other, self.m / other, self.u / other)
        if isinstance(other, TFN):
            return TFN(self.l / other.u, self.m / other.m, self.u / other.l)
        raise TypeError

    def defuzzify(self):
        return (self.l + self.m + self.u) / 3.0

    def distance_to(self, other):
        return np.sqrt((1/3) * ((self.l - other.l)**2 +
                                 (self.m - other.m)**2 +
                                 (self.u - other.u)**2))


# ============================================================================
# Linguistic Scales
# ============================================================================
LINGUISTIC_WEIGHTS = {
    'VL': TFN(0.0, 0.0, 0.1),
    'L':  TFN(0.0, 0.1, 0.3),
    'ML': TFN(0.1, 0.3, 0.5),
    'M':  TFN(0.3, 0.5, 0.7),
    'MH': TFN(0.5, 0.7, 0.9),
    'H':  TFN(0.7, 0.9, 1.0),
    'VH': TFN(0.9, 1.0, 1.0),
}

# Expert-assigned weights
CRITERIA_LINGUISTIC = ['VH', 'H', 'VH', 'H', 'MH', 'MH']


# ============================================================================
# Data Extraction — from isolated dataset results
# ============================================================================
def extract_decision_matrix(results):
    """
    Extract raw decision matrix from isolated dataset results.
    Returns (3 alternatives x 6 criteria) matrix.
    """
    agg = results['aggregated']
    params = results.get('params', {})

    n_alt = len(VARIANT_NAMES)
    raw_matrix = np.zeros((n_alt, 6))

    for i, vname in enumerate(VARIANT_NAMES):
        m = agg[vname]

        # C1: F1-Macro
        raw_matrix[i, 0] = m['f1_macro']
        # C2: Accuracy
        raw_matrix[i, 1] = m['accuracy']
        # C3: Minority Class Recall (Low=index 2) from confusion matrix
        cm = np.array(m['confusion_matrix'])
        raw_matrix[i, 2] = cm[2, 2] / cm[2].sum() if cm[2].sum() > 0 else 0
        # C4: Robustness = 1/CV where CV = f1_std / f1_mean
        mean_f1 = m['f1_macro']
        f1_std = m.get('f1_macro_std', 0.01)
        cv = f1_std / mean_f1 if mean_f1 > 0 else 1.0
        raw_matrix[i, 3] = 1.0 / cv
        # C5: Inference Speed = 1/inference_time_ms
        raw_matrix[i, 4] = 1.0 / m['inference_time_ms']
        # C6: Efficiency = 1M / n_params
        n_params = params.get(vname, 100000)
        raw_matrix[i, 5] = 1e6 / n_params

    return raw_matrix


# ============================================================================
# Fuzzy-TOPSIS Algorithm
# ============================================================================
def normalize_decision_matrix(raw_matrix):
    n_alt, n_crit = raw_matrix.shape
    norm_matrix = np.zeros_like(raw_matrix)
    for j in range(n_crit):
        col_max = raw_matrix[:, j].max()
        if col_max > 0:
            norm_matrix[:, j] = raw_matrix[:, j] / col_max
    return norm_matrix


def create_fuzzy_decision_matrix(norm_matrix, spread=0.05):
    n_alt, n_crit = norm_matrix.shape
    fuzzy_matrix = [[None]*n_crit for _ in range(n_alt)]
    for i in range(n_alt):
        for j in range(n_crit):
            x = norm_matrix[i, j]
            l = max(0, x - spread)
            u = min(1, x + spread)
            fuzzy_matrix[i][j] = TFN(l, x, u)
    return fuzzy_matrix


def get_fuzzy_weights():
    return [LINGUISTIC_WEIGHTS[ling] for ling in CRITERIA_LINGUISTIC]


def compute_weighted_fuzzy_matrix(fuzzy_matrix, weights):
    n_alt = len(fuzzy_matrix)
    n_crit = len(weights)
    weighted = [[None]*n_crit for _ in range(n_alt)]
    for i in range(n_alt):
        for j in range(n_crit):
            weighted[i][j] = fuzzy_matrix[i][j] * weights[j]
    return weighted


def compute_fpis_fnis(weighted_matrix):
    n_alt = len(weighted_matrix)
    n_crit = len(weighted_matrix[0])
    fpis, fnis = [], []
    for j in range(n_crit):
        l_vals = [weighted_matrix[i][j].l for i in range(n_alt)]
        m_vals = [weighted_matrix[i][j].m for i in range(n_alt)]
        u_vals = [weighted_matrix[i][j].u for i in range(n_alt)]
        fpis.append(TFN(max(l_vals), max(m_vals), max(u_vals)))
        fnis.append(TFN(min(l_vals), min(m_vals), min(u_vals)))
    return fpis, fnis


def compute_distances(weighted_matrix, fpis, fnis):
    n_alt = len(weighted_matrix)
    n_crit = len(fpis)
    d_plus = np.zeros(n_alt)
    d_minus = np.zeros(n_alt)
    for i in range(n_alt):
        for j in range(n_crit):
            d_plus[i] += weighted_matrix[i][j].distance_to(fpis[j])
            d_minus[i] += weighted_matrix[i][j].distance_to(fnis[j])
    return d_plus, d_minus


def compute_closeness_coefficient(d_plus, d_minus):
    return d_minus / (d_plus + d_minus + 1e-10)


def fuzzy_topsis(raw_matrix):
    norm_matrix = normalize_decision_matrix(raw_matrix)
    fuzzy_matrix = create_fuzzy_decision_matrix(norm_matrix, spread=0.05)
    weights = get_fuzzy_weights()
    weighted_matrix = compute_weighted_fuzzy_matrix(fuzzy_matrix, weights)
    fpis, fnis = compute_fpis_fnis(weighted_matrix)
    d_plus, d_minus = compute_distances(weighted_matrix, fpis, fnis)
    cc = compute_closeness_coefficient(d_plus, d_minus)
    ranking = np.argsort(-cc)
    return {
        'raw_matrix': raw_matrix, 'norm_matrix': norm_matrix,
        'weights': weights, 'weighted_matrix': weighted_matrix,
        'fpis': fpis, 'fnis': fnis,
        'd_plus': d_plus, 'd_minus': d_minus,
        'cc': cc, 'ranking': ranking,
    }


# ============================================================================
# Sensitivity Analysis
# ============================================================================
def sensitivity_analysis(raw_matrix):
    scenarios = {
        'S1: Equal Weights':       ['M', 'M', 'M', 'M', 'M', 'M'],
        'S2: Performance Focus':   ['VH', 'VH', 'VH', 'M', 'M', 'M'],
        'S3: Speed Focus':         ['M', 'M', 'M', 'M', 'VH', 'M'],
        'S4: Efficiency Focus':    ['M', 'M', 'M', 'M', 'M', 'VH'],
        'S5: Robustness Focus':    ['M', 'M', 'M', 'VH', 'M', 'M'],
        'S6: Original (Proposed)': CRITERIA_LINGUISTIC,
    }
    results = {}
    for scenario_name, ling_weights in scenarios.items():
        norm_matrix = normalize_decision_matrix(raw_matrix)
        fuzzy_matrix = create_fuzzy_decision_matrix(norm_matrix, spread=0.05)
        weights = [LINGUISTIC_WEIGHTS[w] for w in ling_weights]
        weighted_matrix = compute_weighted_fuzzy_matrix(fuzzy_matrix, weights)
        fpis, fnis = compute_fpis_fnis(weighted_matrix)
        d_plus, d_minus = compute_distances(weighted_matrix, fpis, fnis)
        cc = compute_closeness_coefficient(d_plus, d_minus)
        ranking = np.argsort(-cc)
        results[scenario_name] = {
            'cc': cc.copy(), 'ranking': ranking.copy(),
            'weights_linguistic': ling_weights,
        }
    return results


# ============================================================================
# Charts (fig13-fig17) — per dataset
# ============================================================================

def plot_closeness_coefficients(cc, ranking, charts_dir, ds_label, save_name='fig13'):
    """fig13: CC ranking bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_idx = ranking
    names = [VARIANT_NAMES[i] for i in sorted_idx]
    values = [cc[i] for i in sorted_idx]
    colors = [FAMILY_COLORS[VARIANT_FAMILIES[n]] for n in names]

    bars = ax.barh(range(len(names)), values, color=colors, edgecolor='white', height=0.5)
    for idx, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2,
                f'CC={val:.4f}', va='center', fontsize=11, fontweight='bold')
        ax.text(0.01, bar.get_y() + bar.get_height()/2,
                f'#{idx+1}', va='center', fontsize=12, fontweight='bold', color='white')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel('Closeness Coefficient (CC)')
    ax.set_title(f'Fuzzy-TOPSIS Ranking \u2014 {ds_label} (Isolated)',
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)

    legend_patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10)

    plt.tight_layout()
    path = charts_dir / f'{save_name}_fuzzy_topsis_ranking.png'
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  Saved: {path.name}')


def plot_criteria_radar(raw_matrix, cc, ranking, charts_dir, ds_label, save_name='fig14'):
    """fig14: Radar chart - 3 models across 6 criteria."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    norm = normalize_decision_matrix(raw_matrix)
    angles = np.linspace(0, 2*np.pi, len(CRITERIA_NAMES), endpoint=False).tolist()
    angles += angles[:1]

    colors_top = ['#E74C3C', '#3498DB', '#27AE60']
    for rank, idx in enumerate(ranking):
        values = norm[idx].tolist()
        values += values[:1]
        label = f'#{rank+1} {VARIANT_NAMES[idx]} (CC={cc[idx]:.4f})'
        ax.plot(angles, values, 'o-', linewidth=2.5, markersize=7,
                label=label, color=colors_top[rank])
        ax.fill(angles, values, alpha=0.1, color=colors_top[rank])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CRITERIA_NAMES, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Fuzzy-TOPSIS \u2014 Criteria Profile ({ds_label})',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=10)

    plt.tight_layout()
    path = charts_dir / f'{save_name}_fuzzy_topsis_radar.png'
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  Saved: {path.name}')


def plot_sensitivity_analysis(sensitivity_results, charts_dir, ds_label, save_name='fig15'):
    """fig15: Heatmap of rankings across sensitivity scenarios."""
    fig, ax = plt.subplots(figsize=(12, 4))
    scenarios = list(sensitivity_results.keys())
    n_scenarios = len(scenarios)
    n_models = len(VARIANT_NAMES)

    rank_matrix = np.zeros((n_models, n_scenarios), dtype=int)
    for j, (sname, sres) in enumerate(sensitivity_results.items()):
        for rank_pos, model_idx in enumerate(sres['ranking']):
            rank_matrix[model_idx, j] = rank_pos + 1

    im = ax.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=3)

    for i in range(n_models):
        for j in range(n_scenarios):
            rank = rank_matrix[i, j]
            color = 'white' if rank == 1 or rank == 3 else 'black'
            ax.text(j, i, str(rank), ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)

    ax.set_xticks(range(n_scenarios))
    ax.set_xticklabels([s.split(':')[0] for s in scenarios], fontsize=10, rotation=30, ha='right')
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(VARIANT_NAMES, fontsize=12)
    ax.set_xlabel('Weight Scenario')
    ax.set_ylabel('Model')
    ax.set_title(f'Sensitivity Analysis \u2014 Rank Stability ({ds_label})',
                 fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Rank (1=Best, 3=Worst)', fontsize=10)

    plt.tight_layout()
    path = charts_dir / f'{save_name}_sensitivity_analysis.png'
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  Saved: {path.name}')


def plot_distance_chart(d_plus, d_minus, cc, charts_dir, ds_label, save_name='fig16'):
    """fig16: D+ and D- distance visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(VARIANT_NAMES))
    width = 0.30

    ax1.bar(x - width/2, d_plus, width, label='D+ (to FPIS)', color='#E74C3C', alpha=0.8)
    ax1.bar(x + width/2, d_minus, width, label='D- (to FNIS)', color='#27AE60', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(VARIANT_NAMES, fontsize=10)
    ax1.set_ylabel('Distance')
    ax1.set_title('Distance to Ideal Solutions')
    ax1.legend()

    colors_fam = [FAMILY_COLORS[VARIANT_FAMILIES[v]] for v in VARIANT_NAMES]
    bars = ax2.bar(x, cc, color=colors_fam, edgecolor='white', width=0.5)
    for bar, val in zip(bars, cc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(VARIANT_NAMES, fontsize=10)
    ax2.set_ylabel('Closeness Coefficient (CC)')
    ax2.set_title('Closeness Coefficient Ranking')
    legend_patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax2.legend(handles=legend_patches, fontsize=9)

    plt.suptitle(f'Fuzzy-TOPSIS Distance Analysis \u2014 {ds_label}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = charts_dir / f'{save_name}_distance_analysis.png'
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  Saved: {path.name}')


def plot_recommendation_dashboard(raw_matrix, cc, ranking, results, charts_dir, ds_label, save_name='fig17'):
    """fig17: Comprehensive recommendation dashboard - 4 subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    norm = normalize_decision_matrix(raw_matrix)
    colors_fam = [FAMILY_COLORS[VARIANT_FAMILIES[v]] for v in VARIANT_NAMES]
    link_results = results.get('link_results', {})

    # (a) Per-Link F1-Macro comparison
    ax = axes[0, 0]
    x = np.arange(len(LINK_SHORT))
    width = 0.25
    variant_colors = ['#3498DB', '#27AE60', '#9B59B6']
    for vi, vname in enumerate(VARIANT_NAMES):
        f1_per_link = []
        for lname in LINK_NAMES:
            if lname in link_results and vname in link_results[lname]:
                f1_per_link.append(link_results[lname][vname]['f1_macro'])
            else:
                f1_per_link.append(0)
        ax.bar(x + (vi - 1) * width, f1_per_link, width, label=vname,
               color=variant_colors[vi], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(LINK_SHORT, fontsize=9)
    ax.set_ylabel('F1-Macro')
    ax.legend(fontsize=9)
    ax.set_title('(a) Per-Link F1-Macro Comparison', fontweight='bold')
    ax.set_ylim(0, 1.1)

    # (b) Pareto Front (F1 vs Speed)
    ax = axes[0, 1]
    for i, vname in enumerate(VARIANT_NAMES):
        ax.scatter(raw_matrix[i, 4], raw_matrix[i, 0],
                   s=250, c=colors_fam[i], edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(vname, (raw_matrix[i, 4] + 0.01, raw_matrix[i, 0] + 0.002),
                    fontsize=10, fontweight='bold')
    ax.set_xlabel('Inference Speed (1/ms)'); ax.set_ylabel('F1-Macro')
    ax.set_title('(b) Pareto Front: F1 vs Speed', fontweight='bold')
    legend_patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=legend_patches, fontsize=9)

    # (c) FL Deployment Score
    ax = axes[1, 0]
    fl_scores = []
    for i in range(len(VARIANT_NAMES)):
        det_qual = 0.6 * norm[i, 0] + 0.4 * norm[i, 2]
        fl_score = 0.30 * det_qual + 0.25 * norm[i, 3] + 0.25 * norm[i, 4] + 0.20 * norm[i, 5]
        fl_scores.append(fl_score)
    fl_ranking_idx = np.argsort(-np.array(fl_scores))
    fl_names = [VARIANT_NAMES[i] for i in fl_ranking_idx]
    fl_vals = [fl_scores[i] for i in fl_ranking_idx]
    fl_colors = [FAMILY_COLORS[VARIANT_FAMILIES[n]] for n in fl_names]
    bars = ax.barh(range(len(fl_names)), fl_vals, color=fl_colors, edgecolor='white', height=0.5)
    for idx_pos, (bar, val) in enumerate(zip(bars, fl_vals)):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(fl_names))); ax.set_yticklabels(fl_names, fontsize=11)
    ax.set_xlabel('FL Deployment Score')
    ax.set_title('(c) FL Deployment Suitability', fontweight='bold')
    ax.invert_yaxis()

    # (d) CC vs FL Score scatter
    ax = axes[1, 1]
    for i, vname in enumerate(VARIANT_NAMES):
        ax.scatter(cc[i], fl_scores[i], s=300, c=colors_fam[i],
                   edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(vname, (cc[i] + 0.005, fl_scores[i] + 0.005),
                    fontsize=10, fontweight='bold')
    ax.set_xlabel('Fuzzy-TOPSIS CC Score'); ax.set_ylabel('FL Deployment Score')
    ax.set_title('(d) MCDM vs FL Deployment', fontweight='bold')
    ax.legend(handles=legend_patches, fontsize=9, loc='lower left')

    plt.suptitle(f'Stage 3: Model Recommendation \u2014 {ds_label} (Isolated)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = charts_dir / f'{save_name}_recommendation_dashboard.png'
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  Saved: {path.name}')


# ============================================================================
# Report Generation
# ============================================================================
def generate_report(topsis_results, sensitivity_results, raw_matrix, ds_label, output_dir):
    cc = topsis_results['cc']
    ranking = topsis_results['ranking']
    d_plus = topsis_results['d_plus']
    d_minus = topsis_results['d_minus']
    norm_matrix = topsis_results['norm_matrix']
    weights = topsis_results['weights']

    lines = []
    lines.append('=' * 80)
    lines.append(f'  FL-RLMAMS Stage 3: Fuzzy-TOPSIS Model Selection Report')
    lines.append(f'  Dataset: {ds_label} (Isolated Train & Test)')
    lines.append(f'  Top 3 Models: LSTM-Only, Mamba-Only, CNN-Transformer')
    lines.append('=' * 80)

    # 1. Raw Decision Matrix
    lines.append(f'\n1. RAW DECISION MATRIX ({ds_label})')
    lines.append('-' * 80)
    header = f'{"Model":20s} {"F1-Macro":>10s} {"Accuracy":>10s} {"MinRecall":>10s} {"Robust":>10s} {"Speed":>10s} {"Effic":>10s}'
    lines.append(header)
    for i, vname in enumerate(VARIANT_NAMES):
        r = raw_matrix[i]
        lines.append(f'{vname:20s} {r[0]:10.4f} {r[1]:10.4f} {r[2]:10.4f} '
                     f'{r[3]:10.2f} {r[4]:10.4f} {r[5]:10.2f}')

    # 2. Fuzzy Weights
    lines.append('\n2. FUZZY CRITERIA WEIGHTS')
    lines.append('-' * 80)
    for i, (cname, ling) in enumerate(zip(CRITERIA_NAMES, CRITERIA_LINGUISTIC)):
        w = weights[i]
        lines.append(f'  {cname:20s}: {ling:3s} -> TFN{w}')

    # 3. Normalized Matrix
    lines.append('\n3. NORMALIZED DECISION MATRIX')
    lines.append('-' * 80)
    lines.append(header)
    for i, vname in enumerate(VARIANT_NAMES):
        r = norm_matrix[i]
        lines.append(f'{vname:20s} {r[0]:10.4f} {r[1]:10.4f} {r[2]:10.4f} '
                     f'{r[3]:10.4f} {r[4]:10.4f} {r[5]:10.4f}')

    # 4. Distance & CC
    lines.append('\n4. DISTANCE ANALYSIS & CLOSENESS COEFFICIENT')
    lines.append('-' * 80)
    lines.append(f'{"Model":20s} {"D+ (FPIS)":>12s} {"D- (FNIS)":>12s} {"CC":>10s} {"Rank":>6s}')
    for rank_pos, idx in enumerate(ranking):
        vname = VARIANT_NAMES[idx]
        lines.append(f'{vname:20s} {d_plus[idx]:12.4f} {d_minus[idx]:12.4f} '
                     f'{cc[idx]:10.4f} {rank_pos+1:6d}')

    # 5. Final Ranking
    lines.append('\n5. FINAL RANKING')
    lines.append('-' * 80)
    for rank_pos, idx in enumerate(ranking):
        vname = VARIANT_NAMES[idx]
        family = VARIANT_FAMILIES[vname]
        marker = ' *** OPTIMAL FOR FL DEPLOYMENT ***' if rank_pos == 0 else ''
        lines.append(f'  #{rank_pos+1}  {vname:20s} [{family:12s}]  CC={cc[idx]:.4f}{marker}')

    # 6. Sensitivity Analysis
    lines.append('\n6. SENSITIVITY ANALYSIS')
    lines.append('-' * 80)
    for sname, sres in sensitivity_results.items():
        top1 = VARIANT_NAMES[sres['ranking'][0]]
        top1_cc = sres['cc'][sres['ranking'][0]]
        lines.append(f'  {sname:30s}: Winner = {top1:20s} (CC={top1_cc:.4f})')

    wins = {}
    for sname, sres in sensitivity_results.items():
        winner = VARIANT_NAMES[sres['ranking'][0]]
        wins[winner] = wins.get(winner, 0) + 1
    lines.append(f'\n  Scenario Wins:')
    for model, count in sorted(wins.items(), key=lambda x: -x[1]):
        lines.append(f'    {model:20s}: {count}/{len(sensitivity_results)} scenarios')

    # 7. Recommendation
    winner = VARIANT_NAMES[ranking[0]]
    lines.append('\n7. RECOMMENDATION')
    lines.append('=' * 80)
    lines.append(f'  OPTIMAL MODEL: {winner}')
    lines.append(f'  Fuzzy-TOPSIS CC: {cc[ranking[0]]:.4f}')
    lines.append(f'  Architecture: {VARIANT_FAMILIES[winner]}')
    lines.append(f'')
    lines.append(f'  This model achieves the best balance between detection')
    lines.append(f'  performance, robustness across heterogeneous SDN links, inference')
    lines.append(f'  speed, and communication efficiency for Federated Learning.')
    lines.append(f'')
    lines.append(f'  NEXT: Use this model for Phase 4 FL-RL Multi-Agent training.')
    lines.append('=' * 80)

    report = '\n'.join(lines)
    report_file = output_dir / 'fuzzy_topsis_report.txt'
    os.makedirs(report_file.parent, exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'  Saved: {report_file}')
    return report


# ============================================================================
# Process Single Dataset
# ============================================================================
def process_dataset(ds_key, ds_config):
    """Run full Fuzzy-TOPSIS pipeline for one dataset."""
    ds_label = ds_config['label']
    results_file = ds_config['results_file']
    ds_output_dir = OUTPUT_DIR / ds_key
    charts_dir = ds_output_dir / 'charts'
    os.makedirs(charts_dir, exist_ok=True)

    print(f'\n{"="*70}')
    print(f'  Fuzzy-TOPSIS: {ds_label} (Isolated)')
    print(f'{"="*70}')

    # Load results
    print(f'\n  Loading {results_file.name}...')
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract decision matrix
    raw_matrix = extract_decision_matrix(results)
    print(f'\n  Raw Decision Matrix ({ds_label}):')
    print(f'  {"Model":20s} {"F1":>8s} {"Acc":>8s} {"MinRec":>8s} {"Robust":>8s} {"Speed":>8s} {"Effic":>8s}')
    for i, vname in enumerate(VARIANT_NAMES):
        r = raw_matrix[i]
        print(f'  {vname:20s} {r[0]:.4f}  {r[1]:.4f}  {r[2]:.4f}  {r[3]:.2f}  {r[4]:.4f}  {r[5]:.2f}')

    # Run Fuzzy-TOPSIS
    print(f'\n  Running Fuzzy-TOPSIS...')
    topsis_results = fuzzy_topsis(raw_matrix)
    cc = topsis_results['cc']
    ranking = topsis_results['ranking']

    print(f'\n  === RANKING ({ds_label}) ===')
    for rank_pos, idx in enumerate(ranking):
        vname = VARIANT_NAMES[idx]
        marker = ' <-- OPTIMAL' if rank_pos == 0 else ''
        print(f'  #{rank_pos+1} {vname:20s} CC={cc[idx]:.4f}{marker}')

    # Sensitivity Analysis
    print(f'\n  Running sensitivity analysis...')
    sens_results = sensitivity_analysis(raw_matrix)
    for sname, sres in sens_results.items():
        winner = VARIANT_NAMES[sres['ranking'][0]]
        print(f'    {sname:30s}: Winner = {winner}')

    # Generate Charts
    print(f'\n  Generating charts (fig13-fig17)...')
    plot_closeness_coefficients(cc, ranking, charts_dir, ds_label, 'fig13')
    plot_criteria_radar(raw_matrix, cc, ranking, charts_dir, ds_label, 'fig14')
    plot_sensitivity_analysis(sens_results, charts_dir, ds_label, 'fig15')
    plot_distance_chart(topsis_results['d_plus'], topsis_results['d_minus'], cc,
                        charts_dir, ds_label, 'fig16')
    plot_recommendation_dashboard(raw_matrix, cc, ranking, results,
                                  charts_dir, ds_label, 'fig17')

    # Generate Report
    print(f'\n  Generating report...')
    report = generate_report(topsis_results, sens_results, raw_matrix, ds_label, ds_output_dir)

    # Save JSON summary
    summary = {
        'dataset': ds_key,
        'label': ds_label,
        'scope': f'{ds_label} — Isolated Train & Test',
        'ranking': [VARIANT_NAMES[i] for i in ranking],
        'closeness_coefficients': {VARIANT_NAMES[i]: float(cc[i]) for i in range(len(VARIANT_NAMES))},
        'optimal_model': VARIANT_NAMES[ranking[0]],
        'optimal_cc': float(cc[ranking[0]]),
        'sensitivity_winners': {
            sname: VARIANT_NAMES[sres['ranking'][0]]
            for sname, sres in sens_results.items()
        },
    }
    summary_file = ds_output_dir / 'fuzzy_topsis_results.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Saved: {summary_file}')

    print(f'\n  {ds_label}: OPTIMAL = {VARIANT_NAMES[ranking[0]]} (CC={cc[ranking[0]]:.4f})')
    return summary, report


# ============================================================================
# Main
# ============================================================================
def main():
    print('=' * 70)
    print('  FL-RLMAMS Stage 3: Fuzzy-TOPSIS Model Selection (Isolated)')
    print('=' * 70)

    all_summaries = {}
    for ds_key, ds_config in DATASETS.items():
        summary, report = process_dataset(ds_key, ds_config)
        all_summaries[ds_key] = summary
        print('\n' + report)

    # Combined summary
    combined = {
        'scope': 'Stage 3 — Isolated per Dataset',
        'datasets': all_summaries,
    }
    combined_file = OUTPUT_DIR / 'fuzzy_topsis_combined.json'
    os.makedirs(combined_file.parent, exist_ok=True)
    with open(combined_file, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f'\n  Saved combined: {combined_file}')

    # Final summary
    print('\n' + '=' * 70)
    print('  COMBINED RESULTS')
    print('=' * 70)
    for ds_key, s in all_summaries.items():
        print(f'  {s["label"]:15s}: Optimal = {s["optimal_model"]:20s} CC={s["optimal_cc"]:.4f}')
    print('=' * 70)
    print('  ALL DONE!')


if __name__ == '__main__':
    main()
