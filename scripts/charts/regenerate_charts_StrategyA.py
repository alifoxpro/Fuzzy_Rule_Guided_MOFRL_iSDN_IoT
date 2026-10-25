#!/usr/bin/env python3
"""
FL-RLMAMS Strategy A — IEEE-Quality Chart Regeneration (v2)
============================================================
- Reads ONLY from cached results (no retraining)
- Each figure = ONE standalone plot (NO subplots)
- IEEE journal style (IEEEtran)
- 30+ figures per dataset + cross-dataset comparisons

Usage:
    python scripts/regenerate_charts_StrategyA.py
    python scripts/regenerate_charts_StrategyA.py --dataset ciciot2023
    python scripts/regenerate_charts_StrategyA.py --dataset all
"""

import json, argparse, os, sys
import numpy as np
from pathlib import Path
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ============================================================================
# IEEE Style
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
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
    'lines.markersize': 5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
})

IEEE_W = 7.16   # double column width
IEEE_S = 3.5    # single column width
FIG_W, FIG_H = 7.16, 4.5

LINK_NAMES = [
    'Link_1_Core', 'Link_2_Aggregation_L', 'Link_3_Aggregation_R',
    'Link_4_Edge_L', 'Link_5_Edge_R',
]
LINK_SHORT = ['L1-Core', 'L2-AggrL', 'L3-AggrR', 'L4-EdgeL', 'L5-EdgeR']
LINK_COLORS = ['#E74C3C', '#3498DB', '#F39C12', '#27AE60', '#8E44AD']

ACTION_NAMES = ['ALLOW', 'RATE_LIMIT', 'BLOCK']
ACTION_COLORS = ['#27AE60', '#F39C12', '#E74C3C']
SEV_NAMES = ['High', 'Medium', 'Low']

PHASE_NAMES = ['Independent', 'FedAvg', 'FedProx']
PHASE_KEYS  = ['independent', 'fedavg', 'fedprox']
PHASE_COLORS  = {'Independent': '#E74C3C', 'FedAvg': '#3498DB', 'FedProx': '#27AE60'}
PHASE_HATCHES = {'Independent': '///', 'FedAvg': '...', 'FedProx': 'xxx'}
PHASE_MARKERS = {'Independent': 'o', 'FedAvg': 's', 'FedProx': 'D'}

DS_LABELS = {'ciciot2023': 'CICIoT2023', 'diad2024': 'DIAD2024'}
RESULTS_SRC  = Path('experiments/results_StrategyA')       # read data from here
RESULTS_OUT  = Path('experiments/results_StrategyA')      # save charts here

# ── helpers ──
def smooth(y, w=7):
    if len(y) < w: return np.array(y)
    return np.convolve(y, np.ones(w)/w, mode='valid')

def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f'    {path.name}')

def load_dataset(ds_key):
    ds_dir = RESULTS_SRC / ds_key
    data = {}
    for phase, folder in [('independent','independent'),('fedavg','fedavg'),
                           ('fedprox','fedprox'),('master','master_agent')]:
        fp = ds_dir / folder / 'results.json'
        if fp.exists():
            with open(fp) as f: data[phase] = json.load(f)
            print(f'  Loaded {fp}')
        else:
            data[phase] = {}
            print(f'  MISSING {fp}')
    return data

def get_global(data, pkey, metric):
    return data[pkey].get('global', {}).get(metric, 0)


# ============================================================================
#  FIG 01: Reward Convergence (Phase A)
# ============================================================================
def fig01(data, d, L):
    hist = data['independent'].get('history', {})
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.2))
    for i in range(5):
        k = str(i)
        if k in hist and 'rewards' in hist[k]:
            raw = hist[k]['rewards']
            sm = smooth(raw)
            ax.plot(range(len(raw)), raw, color=LINK_COLORS[i], alpha=0.15, linewidth=0.5)
            ax.plot(range(len(sm)), sm, color=LINK_COLORS[i], linewidth=1.5, label=LINK_SHORT[i])
    ax.set_xlabel('Episode'); ax.set_ylabel('Average Reward')
    ax.set_title(f'Phase A: Reward Convergence — {L}', fontweight='bold')
    ax.legend(ncol=3, framealpha=0.9); fig.tight_layout()
    save(fig, d/'fig01_reward_convergence.png')

# ============================================================================
#  FIG 02: TD Loss (log scale)
# ============================================================================
def fig02(data, d, L):
    hist = data['independent'].get('history', {})
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.2))
    for i in range(5):
        k = str(i)
        if k in hist and 'loss' in hist[k]:
            sm = smooth(hist[k]['loss'])
            ax.plot(range(len(sm)), sm, color=LINK_COLORS[i], linewidth=1.5, label=LINK_SHORT[i])
    ax.set_yscale('log')
    ax.set_xlabel('Episode'); ax.set_ylabel('TD Loss (log)')
    ax.set_title(f'Phase A: TD Loss Convergence — {L}', fontweight='bold')
    ax.legend(ncol=3, framealpha=0.9); fig.tight_layout()
    save(fig, d/'fig02_td_loss.png')

# ============================================================================
#  FIG 03: MSR per Agent (Phase A)
# ============================================================================
def fig03(data, d, L):
    hist = data['independent'].get('history', {})
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.2))
    for i in range(5):
        k = str(i)
        if k in hist and 'msr' in hist[k]:
            msr = hist[k]['msr']
            total_eps = len(hist[k].get('rewards', []))
            x = np.linspace(0, total_eps-1, len(msr)) if total_eps > 0 else range(len(msr))
            ax.plot(x, msr, 'o-', color=LINK_COLORS[i], linewidth=1.5,
                    markersize=5, label=LINK_SHORT[i])
    ax.set_xlabel('Episode'); ax.set_ylabel('MSR')
    ax.set_ylim(0.35, 1.05)
    ax.set_title(f'Phase A: Mitigation Success Rate — {L}', fontweight='bold')
    ax.legend(ncol=3, framealpha=0.9); fig.tight_layout()
    save(fig, d/'fig03_msr_training.png')

# ============================================================================
#  FIG 04: Epsilon Decay
# ============================================================================
def fig04(data, d, L):
    hist = data['independent'].get('history', {})
    fig, ax = plt.subplots(figsize=(IEEE_W, 2.8))
    if '0' in hist and 'epsilon' in hist['0']:
        eps = hist['0']['epsilon']
        ax.plot(range(len(eps)), eps, '-', color='#C0392B', linewidth=2)
        ax.fill_between(range(len(eps)), eps, alpha=0.1, color='#E74C3C')
        ax.axhline(y=0.01, color='gray', ls='--', lw=0.8, label='$\\epsilon_{min}$=0.01')
    ax.set_xlabel('Episode'); ax.set_ylabel('$\\epsilon$')
    ax.set_title(f'Exploration Decay — {L}', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig04_epsilon_decay.png')

# ============================================================================
#  FIG 05: Cumulative Reward per Agent
# ============================================================================
def fig05(data, d, L):
    hist = data['independent'].get('history', {})
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.2))
    for i in range(5):
        k = str(i)
        if k in hist and 'rewards' in hist[k]:
            cum = np.cumsum(hist[k]['rewards'])
            ax.plot(range(len(cum)), cum, color=LINK_COLORS[i], linewidth=1.5,
                    label=LINK_SHORT[i])
    ax.set_xlabel('Episode'); ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'Phase A: Cumulative Reward — {L}', fontweight='bold')
    ax.legend(ncol=3, framealpha=0.9); fig.tight_layout()
    save(fig, d/'fig05_cumulative_reward.png')

# ============================================================================
#  FIG 06: Phase Comparison — MSR (grouped bar)
# ============================================================================
def fig06(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    x = np.arange(4)  # MSR, FPR, FNR, JFI
    width = 0.22
    metrics = ['mean_msr', 'mean_fpr', 'mean_fnr', 'jain_fairness']
    mlabels = ['MSR', 'FPR', 'FNR', 'JFI']
    for pi, (pn, pk) in enumerate(zip(PHASE_NAMES, PHASE_KEYS)):
        vals = [get_global(data, pk, m) for m in metrics]
        bars = ax.bar(x + pi*width, vals, width, label=pn, color=PHASE_COLORS[pn],
                      hatch=PHASE_HATCHES[pn], edgecolor='black', lw=0.5, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x + width); ax.set_xticklabels(mlabels, fontweight='bold')
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.18)
    ax.set_title(f'Phase Comparison — {L}', fontweight='bold')
    ax.legend(framealpha=0.9); ax.set_axisbelow(True); fig.tight_layout()
    save(fig, d/'fig06_phase_comparison.png')

# ============================================================================
#  FIG 07: Per-Link MSR across Phases
# ============================================================================
def fig07(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    x = np.arange(5); width = 0.22
    for pi, (pn, pk) in enumerate(zip(PHASE_NAMES, PHASE_KEYS)):
        vals = [data[pk].get(ln, {}).get('msr', 0) for ln in LINK_NAMES]
        bars = ax.bar(x + pi*width, vals, width, label=pn, color=PHASE_COLORS[pn],
                      hatch=PHASE_HATCHES[pn], edgecolor='black', lw=0.5, zorder=3)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                        f'{v:.2f}', ha='center', fontsize=6, fontweight='bold', rotation=45)
    ax.set_xticks(x + width); ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('MSR'); ax.set_ylim(0, 1.18)
    ax.set_title(f'Per-Link MSR — {L}', fontweight='bold')
    ax.legend(fontsize=7); ax.set_axisbelow(True); fig.tight_layout()
    save(fig, d/'fig07_per_link_msr.png')

# ============================================================================
#  FIG 08: Per-Link Reward across Phases
# ============================================================================
def fig08(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    x = np.arange(5); width = 0.22
    for pi, (pn, pk) in enumerate(zip(PHASE_NAMES, PHASE_KEYS)):
        vals = [data[pk].get(ln, {}).get('avg_reward', 0) for ln in LINK_NAMES]
        ax.bar(x + pi*width, vals, width, label=pn, color=PHASE_COLORS[pn],
               hatch=PHASE_HATCHES[pn], edgecolor='black', lw=0.5, zorder=3)
    ax.set_xticks(x + width); ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('Average Reward')
    ax.set_title(f'Per-Link Reward — {L}', fontweight='bold')
    ax.legend(fontsize=7); ax.set_axisbelow(True); fig.tight_layout()
    save(fig, d/'fig08_per_link_reward.png')

# ============================================================================
#  FIG 09: FL MSR Convergence (FedAvg vs FedProx)
# ============================================================================
def fig09(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.2))
    for algo, key, color, marker in [('FedAvg','fedavg','#3498DB','s'),
                                      ('FedProx','fedprox','#27AE60','D')]:
        hist = data[key].get('history', {})
        if 'global_msr' in hist:
            rounds = hist.get('rounds', list(range(1, len(hist['global_msr'])+1)))
            raw = hist['global_msr']
            ax.plot(rounds, raw, marker, color=color, alpha=0.3, markersize=4)
            sm = smooth(raw, 5) if len(raw) >= 5 else raw
            offset = (len(raw) - len(sm)) // 2
            ax.plot(rounds[offset:offset+len(sm)], sm, '-', color=color,
                    linewidth=2.5, label=f'{algo} (MA-5)')
    ax.set_xlabel('FL Round'); ax.set_ylabel('Global MSR')
    ax.set_title(f'FL Convergence: MSR — {L}', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig09_fl_msr_convergence.png')

# ============================================================================
#  FIG 10: FL JFI Convergence
# ============================================================================
def fig10(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.2))
    for algo, key, color, marker in [('FedAvg','fedavg','#3498DB','s'),
                                      ('FedProx','fedprox','#27AE60','D')]:
        hist = data[key].get('history', {})
        if 'jain_fairness' in hist:
            rounds = hist.get('rounds', list(range(1, len(hist['jain_fairness'])+1)))
            ax.plot(rounds, hist['jain_fairness'], f'{marker}-', color=color,
                    linewidth=1.5, markersize=4, label=algo)
    ax.axhline(y=0.9, color='gray', ls='--', lw=0.8, label='Threshold (0.9)')
    ax.set_xlabel('FL Round'); ax.set_ylabel("Jain's Fairness Index")
    ax.set_title(f'FL Convergence: Fairness — {L}', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig10_fl_jfi_convergence.png')

# ============================================================================
#  FIG 11: Per-Agent FedAvg Convergence
# ============================================================================
def fig11(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.2))
    hist = data['fedavg'].get('history', {})
    pa = hist.get('per_agent', {})
    rounds = hist.get('rounds', [])
    for i in range(5):
        k = str(i)
        if k in pa and 'msr' in pa[k]:
            r = rounds[:len(pa[k]['msr'])]
            ax.plot(r, pa[k]['msr'], 'o-', label=LINK_SHORT[i], color=LINK_COLORS[i],
                    linewidth=1.2, markersize=3, alpha=0.85)
    ax.set_xlabel('FL Round'); ax.set_ylabel('MSR')
    ax.set_ylim(0.15, 1.05)
    ax.set_title(f'FedAvg: Per-Agent MSR — {L}', fontweight='bold')
    ax.legend(ncol=3, fontsize=7); fig.tight_layout()
    save(fig, d/'fig11_fedavg_per_agent.png')

# ============================================================================
#  FIG 12: Per-Agent FedProx Convergence
# ============================================================================
def fig12(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.2))
    hist = data['fedprox'].get('history', {})
    pa = hist.get('per_agent', {})
    rounds = hist.get('rounds', [])
    for i in range(5):
        k = str(i)
        if k in pa and 'msr' in pa[k]:
            r = rounds[:len(pa[k]['msr'])]
            ax.plot(r, pa[k]['msr'], 'o-', label=LINK_SHORT[i], color=LINK_COLORS[i],
                    linewidth=1.2, markersize=3, alpha=0.85)
    ax.set_xlabel('FL Round'); ax.set_ylabel('MSR')
    ax.set_ylim(0.15, 1.05)
    ax.set_title(f'FedProx: Per-Agent MSR — {L}', fontweight='bold')
    ax.legend(ncol=3, fontsize=7); fig.tight_layout()
    save(fig, d/'fig12_fedprox_per_agent.png')

# ============================================================================
#  FIG 13: FedAvg Per-Agent FPR
# ============================================================================
def fig13(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.2))
    hist = data['fedavg'].get('history', {})
    pa = hist.get('per_agent', {})
    rounds = hist.get('rounds', [])
    for i in range(5):
        k = str(i)
        if k in pa and 'fpr' in pa[k]:
            r = rounds[:len(pa[k]['fpr'])]
            ax.plot(r, pa[k]['fpr'], 'o-', label=LINK_SHORT[i], color=LINK_COLORS[i],
                    linewidth=1.2, markersize=3)
    ax.set_xlabel('FL Round'); ax.set_ylabel('FPR')
    ax.set_title(f'FedAvg: Per-Agent FPR — {L}', fontweight='bold')
    ax.legend(ncol=3, fontsize=7); fig.tight_layout()
    save(fig, d/'fig13_fedavg_fpr.png')

# ============================================================================
#  FIG 14: FedProx Per-Agent FPR
# ============================================================================
def fig14(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.2))
    hist = data['fedprox'].get('history', {})
    pa = hist.get('per_agent', {})
    rounds = hist.get('rounds', [])
    for i in range(5):
        k = str(i)
        if k in pa and 'fpr' in pa[k]:
            r = rounds[:len(pa[k]['fpr'])]
            ax.plot(r, pa[k]['fpr'], 'o-', label=LINK_SHORT[i], color=LINK_COLORS[i],
                    linewidth=1.2, markersize=3)
    ax.set_xlabel('FL Round'); ax.set_ylabel('FPR')
    ax.set_title(f'FedProx: Per-Agent FPR — {L}', fontweight='bold')
    ax.legend(ncol=3, fontsize=7); fig.tight_layout()
    save(fig, d/'fig14_fedprox_fpr.png')

# ============================================================================
#  FIG 15: Severity → Action Heatmap (FedProx, aggregated)
# ============================================================================
def fig15(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_S + 1.2, 3.5))
    best = data['fedprox']
    total = np.zeros((3,3), dtype=np.float64)
    for ln in LINK_NAMES:
        sam = best.get(ln, {}).get('severity_action_matrix', [[0]*3]*3)
        total += np.array(sam, dtype=np.float64)
    rs = total.sum(axis=1, keepdims=True); rs[rs==0] = 1
    norm = total / rs
    sns.heatmap(norm, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=ACTION_NAMES, yticklabels=SEV_NAMES,
                ax=ax, vmin=0, vmax=1, linewidths=1.2, linecolor='white',
                cbar_kws={'label': 'Proportion', 'shrink': 0.8},
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_xlabel('Agent Action'); ax.set_ylabel('True Severity')
    ax.set_title(f'Severity → Action (FedProx) — {L}', fontweight='bold')
    fig.tight_layout()
    save(fig, d/'fig15_severity_action_heatmap.png')

# ============================================================================
#  FIG 16: Per-Link Severity Heatmaps (5 individual)
# ============================================================================
def fig16(data, d, L):
    best = data['fedprox']
    for i, (ln, ls) in enumerate(zip(LINK_NAMES, LINK_SHORT)):
        fig, ax = plt.subplots(figsize=(IEEE_S + 0.5, 3.0))
        sam = best.get(ln, {}).get('severity_action_matrix', [[0]*3]*3)
        arr = np.array(sam, dtype=np.float64)
        rs = arr.sum(axis=1, keepdims=True); rs[rs==0] = 1
        norm = arr / rs
        sns.heatmap(norm, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=ACTION_NAMES, yticklabels=SEV_NAMES,
                    ax=ax, vmin=0, vmax=1, linewidths=1, linecolor='white',
                    annot_kws={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_xlabel('Action'); ax.set_ylabel('Severity')
        ax.set_title(f'{ls}: Severity → Action — {L}', fontweight='bold')
        fig.tight_layout()
        save(fig, d/f'fig16{chr(97+i)}_{ln}_heatmap.png')

# ============================================================================
#  FIG 17: Action Distribution per Link (stacked)
# ============================================================================
def fig17(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    best = data['fedprox']
    x = np.arange(5); bottom = np.zeros(5)
    for ai, aname in enumerate(ACTION_NAMES):
        vals = []
        for ln in LINK_NAMES:
            ad = best.get(ln, {}).get('action_distribution', {})
            total_a = sum(ad.values()) if ad else 1
            vals.append(ad.get(aname, 0) / max(total_a, 1))
        ax.bar(x, vals, 0.55, bottom=bottom, label=aname, color=ACTION_COLORS[ai],
               edgecolor='white', lw=0.8)
        for j, v in enumerate(vals):
            if v > 0.08:
                ax.text(x[j], bottom[j]+v/2, f'{v:.0%}', ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if ai==2 else 'black')
        bottom += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('Proportion'); ax.set_ylim(0, 1.05)
    ax.set_title(f'Action Distribution (FedProx) — {L}', fontweight='bold')
    ax.legend(loc='upper right'); fig.tight_layout()
    save(fig, d/'fig17_action_distribution.png')

# ============================================================================
#  FIG 18: FPR per Link across Phases
# ============================================================================
def fig18(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    x = np.arange(5); width = 0.22
    for pi, (pn, pk) in enumerate(zip(PHASE_NAMES, PHASE_KEYS)):
        vals = [data[pk].get(ln, {}).get('fpr', 0) for ln in LINK_NAMES]
        bars = ax.bar(x + pi*width, vals, width, label=pn, color=PHASE_COLORS[pn],
                      hatch=PHASE_HATCHES[pn], edgecolor='black', lw=0.5, zorder=3)
        for b, v in zip(bars, vals):
            if v > 0.01:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                        f'{v:.3f}', ha='center', fontsize=6, rotation=45)
    ax.set_xticks(x + width); ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('FPR')
    ax.set_title(f'Per-Link False Positive Rate — {L}', fontweight='bold')
    ax.legend(fontsize=7); ax.set_axisbelow(True); fig.tight_layout()
    save(fig, d/'fig18_per_link_fpr.png')

# ============================================================================
#  FIG 19: FNR per Link across Phases
# ============================================================================
def fig19(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    x = np.arange(5); width = 0.22
    for pi, (pn, pk) in enumerate(zip(PHASE_NAMES, PHASE_KEYS)):
        vals = [data[pk].get(ln, {}).get('fnr', 0) for ln in LINK_NAMES]
        ax.bar(x + pi*width, vals, width, label=pn, color=PHASE_COLORS[pn],
               hatch=PHASE_HATCHES[pn], edgecolor='black', lw=0.5, zorder=3)
    ax.set_xticks(x + width); ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('FNR')
    ax.set_title(f'Per-Link False Negative Rate — {L}', fontweight='bold')
    ax.legend(fontsize=7); ax.set_axisbelow(True); fig.tight_layout()
    save(fig, d/'fig19_per_link_fnr.png')

# ============================================================================
#  FIG 20: MSR Box Plot
# ============================================================================
def fig20(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_S + 2, 3.5))
    box_data, box_colors = [], []
    for pn, pk in zip(PHASE_NAMES, PHASE_KEYS):
        vals = [data[pk].get(ln, {}).get('msr', 0) for ln in LINK_NAMES]
        box_data.append(vals); box_colors.append(PHASE_COLORS[pn])
    bp = ax.boxplot(box_data, labels=PHASE_NAMES, patch_artist=True, widths=0.5,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black', markersize=6),
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    np.random.seed(42)
    for xi, (vals, c) in enumerate(zip(box_data, box_colors)):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter([xi+1+j for j in jitter], vals, color=c,
                   edgecolor='black', s=60, zorder=5, alpha=0.9, lw=0.5)
    ax.set_ylabel('MSR')
    ax.set_title(f'MSR Distribution — {L}', fontweight='bold')
    fig.tight_layout()
    save(fig, d/'fig20_msr_boxplot.png')

# ============================================================================
#  FIG 21: Radar — Multi-Metric Comparison
# ============================================================================
def fig21(data, d, L):
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    labels = ['MSR', '1−FPR', '1−FNR', 'JFI', 'Reward\n(×2.5)']
    n = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    for pn, pk in zip(PHASE_NAMES, PHASE_KEYS):
        g = data[pk].get('global', {})
        vals = [g.get('mean_msr',0), 1-g.get('mean_fpr',0), 1-g.get('mean_fnr',0),
                g.get('jain_fairness',0), min(g.get('mean_reward',0)*2.5, 1.0)]
        vals += vals[:1]
        ax.plot(angles, vals, f'{PHASE_MARKERS[pn]}-', lw=2, markersize=6,
                label=pn, color=PHASE_COLORS[pn])
        ax.fill(angles, vals, alpha=0.08, color=PHASE_COLORS[pn])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Multi-Metric Radar — {L}', fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    save(fig, d/'fig21_radar.png')

# ============================================================================
#  FIG 22: Master — Health Score Breakdown (stacked)
# ============================================================================
def fig22(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.8))
    ms = data['master']; last = ms.get('last_result', {})
    hc = last.get('health_components', {})
    x = np.arange(5)
    comp = [('capacity',0.40,'#2980B9'), ('latency',0.25,'#F39C12'),
            ('pkt_loss',0.20,'#E74C3C'), ('safety',0.15,'#27AE60')]
    bottom = np.zeros(5)
    for cname, cw, color in comp:
        vals = [hc.get(LINK_SHORT[i], {}).get(cname, 0) * cw for i in range(5)]
        ax.bar(x, vals, 0.55, bottom=bottom, label=f'{cname} ({cw})',
               color=color, edgecolor='white', lw=0.8)
        bottom += np.array(vals)
    hmean = ms.get('health_mean', [0]*5)
    for i in range(5):
        ax.text(i, bottom[i]+0.012, f'H={hmean[i]:.3f}', ha='center', fontsize=9,
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.15',
                facecolor='white', edgecolor='gray', alpha=0.85))
    ax.set_xticks(x); ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('Weighted Health Score'); ax.set_ylim(0, 1.08)
    ax.set_title(f'Master Agent: Health Score Breakdown — {L}', fontweight='bold')
    ax.legend(ncol=2, fontsize=8); fig.tight_layout()
    save(fig, d/'fig22_master_health.png')

# ============================================================================
#  FIG 23: Master — Routing Probabilities (horizontal bar)
# ============================================================================
def fig23(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.0))
    ms = data['master']
    ranking = ms.get('last_result', {}).get('ranking', LINK_SHORT)
    rp = ms.get('routing_prob_mean', {})
    names = ranking
    vals = [rp.get(n, 0.2) for n in names]
    colors = [LINK_COLORS[LINK_SHORT.index(n)] if n in LINK_SHORT else '#999' for n in names]
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='black', lw=0.5, height=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.004, bar.get_y()+bar.get_height()/2,
                f'{v:.1%}', va='center', fontsize=9, fontweight='bold')
    ax.text(vals[0]/2, 0, 'PRIMARY', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel('Routing Probability')
    ax.set_xlim(0, max(vals)*1.3); ax.invert_yaxis()
    ax.set_title(f'Master Agent: Routing Probabilities — {L}', fontweight='bold')
    fig.tight_layout()
    save(fig, d/'fig23_master_routing.png')

# ============================================================================
#  FIG 24: Master — Link Ranking (horizontal bar)
# ============================================================================
def fig24(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.0))
    ms = data['master']
    hmean = ms.get('health_mean', [0]*5)
    pairs = sorted(zip(LINK_SHORT, hmean), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in pairs]; vals = [p[1] for p in pairs]
    colors = [LINK_COLORS[LINK_SHORT.index(n)] for n in names]
    bars = ax.barh(range(5), vals, color=colors, edgecolor='black', lw=0.5, height=0.6)
    for j, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'#{i+1} {n}' for i, n in enumerate(names)])
    ax.set_xlabel('Health Score'); ax.invert_yaxis()
    ax.set_xlim(0, max(vals)*1.15)
    ax.set_title(f'Master Agent: Link Health Ranking — {L}', fontweight='bold')
    fig.tight_layout()
    save(fig, d/'fig24_master_ranking.png')

# ============================================================================
#  FIG 25: Master — Health Heatmap over Episodes
# ============================================================================
def fig25(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.0))
    ms = data['master']; ah = ms.get('all_health', [])
    if ah and len(ah) > 1:
        arr = np.array(ah).T
        sns.heatmap(arr, ax=ax, cmap='RdYlGn', vmin=0.5, vmax=1.0,
                    yticklabels=LINK_SHORT,
                    xticklabels=[str(i+1) if (i+1)%10==0 else '' for i in range(arr.shape[1])],
                    cbar_kws={'label':'Health Score','shrink':0.8},
                    linewidths=0.3, linecolor='white')
        ax.set_xlabel('Episode'); ax.set_ylabel('Link')
    else:
        ax.text(0.5, 0.5, 'No episode data', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
    ax.set_title(f'Master Agent: Health Over Episodes — {L}', fontweight='bold')
    fig.tight_layout()
    save(fig, d/'fig25_health_heatmap.png')

# ============================================================================
#  FIG 26: Master — JFI Before vs After
# ============================================================================
def fig26(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_S + 1, 3.2))
    ms = data['master']
    jb = ms.get('jfi_before_mean', 0); ja = ms.get('jfi_after_mean', 0)
    bars = ax.bar(['Before\nMaster', 'After\nMaster'], [jb, ja],
                  color=['#E74C3C','#27AE60'], edgecolor='black', lw=0.5, width=0.5)
    for b, v in zip(bars, [jb, ja]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(y=0.9, color='gray', ls='--', lw=0.8, label='Threshold')
    ax.set_ylabel('JFI'); ax.set_ylim(0, 1.1)
    ax.set_title(f'Fairness: Before vs After Master — {L}', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig26_jfi_before_after.png')

# ============================================================================
#  FIG 27: Master — Utilization Before vs After
# ============================================================================
def fig27(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.2))
    ms = data['master']; last = ms.get('last_result', {})
    ub = last.get('utilization_before', [0]*5)
    ua = last.get('utilization_after', [0]*5)
    x = np.arange(5)
    ax.bar(x-0.18, ub, 0.33, label='Before', color='#E74C3C', alpha=0.8,
           edgecolor='black', lw=0.5)
    ax.bar(x+0.18, ua, 0.33, label='After', color='#27AE60', alpha=0.8,
           edgecolor='black', lw=0.5)
    ax.axhline(y=0.85, color='gray', ls='--', lw=1, label='Overload (85%)')
    ax.set_xticks(x); ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('Utilization')
    ax.set_title(f'Link Utilization: Before vs After Master — {L}', fontweight='bold')
    ax.legend(fontsize=7); fig.tight_layout()
    save(fig, d/'fig27_utilization.png')

# ============================================================================
#  FIG 28: Reward vs Error Trade-off
# ============================================================================
def fig28(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    best = data['fedprox']
    x = np.arange(5)
    rewards = [best.get(ln,{}).get('avg_reward',0) for ln in LINK_NAMES]
    fpr = [best.get(ln,{}).get('fpr',0) for ln in LINK_NAMES]
    fnr = [best.get(ln,{}).get('fnr',0) for ln in LINK_NAMES]
    w = 0.25
    ax.bar(x-w, rewards, w, label='Reward', color='#27AE60', edgecolor='black', lw=0.5)
    ax.bar(x,   fpr,     w, label='FPR',    color='#E74C3C', edgecolor='black', lw=0.5)
    ax.bar(x+w, fnr,     w, label='FNR',    color='#F39C12', edgecolor='black', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('Value')
    ax.set_title(f'Reward vs Error Rates (FedProx) — {L}', fontweight='bold')
    ax.legend(); ax.set_axisbelow(True); fig.tight_layout()
    save(fig, d/'fig28_reward_vs_error.png')

# ============================================================================
#  FIG 29: MSR Improvement Waterfall (Independent → FedProx → +Master)
# ============================================================================
def fig29(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    ms = data['master']
    indep_msr = get_global(data, 'independent', 'mean_msr')
    fedprox_msr = get_global(data, 'fedprox', 'mean_msr')
    # Master doesn't change MSR directly but improves JFI
    indep_jfi = get_global(data, 'independent', 'jain_fairness')
    master_jfi = ms.get('jfi_after_mean', 0)

    labels = ['Indep.\nMSR', 'FedProx\nMSR', 'Indep.\nJFI', 'Master\nJFI']
    vals = [indep_msr, fedprox_msr, indep_jfi, master_jfi]
    colors = ['#3498DB', '#27AE60', '#E67E22', '#8E44AD']

    bars = ax.bar(labels, vals, color=colors, edgecolor='black', lw=0.5, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')

    # Arrows showing change
    for i in range(0, len(vals)-1, 2):
        diff = vals[i+1] - vals[i]
        color = '#27AE60' if diff >= 0 else '#E74C3C'
        mid_x = (i + i + 1) / 2
        ax.annotate(f'{diff:+.4f}', xy=(i+1, vals[i+1]), xytext=(mid_x+0.5, max(vals)+0.06),
                    fontsize=8, fontweight='bold', color=color, ha='center',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    ax.set_ylabel('Score'); ax.set_ylim(0, 1.2)
    ax.set_title(f'Performance Progression — {L}', fontweight='bold')
    fig.tight_layout()
    save(fig, d/'fig29_progression.png')

# ============================================================================
#  FIG 30: Communication Overhead (estimated)
# ============================================================================
def fig30(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_S + 2, 3.5))
    # DQN params = 14147, 4 bytes each → ~56KB per model
    model_size_kb = 14147 * 4 / 1024
    n_agents = 5
    fedavg_rounds = len(data['fedavg'].get('history', {}).get('rounds', []))
    fedprox_rounds = len(data['fedprox'].get('history', {}).get('rounds', []))

    # Each round: 5 uploads + 1 broadcast = 6 transfers
    labels = ['Independent', 'FedAvg', 'FedProx']
    transfers = [0, fedavg_rounds * (n_agents + 1), fedprox_rounds * (n_agents + 1)]
    total_kb = [0, transfers[1] * model_size_kb, transfers[2] * model_size_kb]
    total_mb = [t/1024 for t in total_kb]

    colors = [PHASE_COLORS[p] for p in PHASE_NAMES]
    bars = ax.bar(labels, total_mb, color=colors, edgecolor='black', lw=0.5, width=0.5)
    for b, v, t in zip(bars, total_mb, transfers):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                f'{v:.1f} MB\n({t} msgs)', ha='center', fontsize=8, fontweight='bold')
    ax.set_ylabel('Total Communication (MB)')
    ax.set_title(f'Communication Overhead — {L}', fontweight='bold')
    fig.tight_layout()
    save(fig, d/'fig30_communication_overhead.png')

# ============================================================================
#  FIG 31: Summary Table
# ============================================================================
def fig31(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.8))
    ax.set_axis_off()
    ms = data['master']
    cols = ['Phase', 'MSR', 'FPR', 'FNR', 'JFI', 'Reward']
    rows = []; best_idx = 0; best_val = 0
    for ri, (pn, pk) in enumerate(zip(PHASE_NAMES, PHASE_KEYS)):
        g = data[pk].get('global', {})
        msr = g.get('mean_msr', 0)
        if msr > best_val: best_val = msr; best_idx = ri
        rows.append([pn, f'{msr:.4f}', f'{g.get("mean_fpr",0):.4f}',
                     f'{g.get("mean_fnr",0):.4f}', f'{g.get("jain_fairness",0):.4f}',
                     f'{g.get("mean_reward",0):.4f}'])
    bf = ms.get('best_link_frequency', {})
    bl = max(bf, key=bf.get) if bf else 'N/A'
    jb = ms.get('jfi_before_mean', 0); ja = ms.get('jfi_after_mean', 0)
    rows.append(['Master Agent', f'Best: {bl}', '—', '—',
                 f'{ja:.4f}', f'JFI {ja-jb:+.4f}'])

    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.8)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#1A252F')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for j in range(len(cols)):
        tbl[best_idx+1, j].set_facecolor('#D5F5E3')
    for j in range(len(cols)):
        tbl[len(rows), j].set_facecolor('#FEF9E7')
    ax.set_title(f'Strategy A: Results Summary — {L}', fontsize=13, fontweight='bold', pad=15)
    fig.tight_layout()
    save(fig, d/'fig31_summary_table.png')

# ============================================================================
#  FIG 32: Master — Health Component Radar (per link)
# ============================================================================
def fig32(data, d, L):
    ms = data['master']; last = ms.get('last_result', {})
    hc = last.get('health_components', {})
    if not hc: return
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    labels = ['Capacity', 'Latency', 'Pkt Loss', 'Safety']
    n = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    for i in range(5):
        comp = hc.get(LINK_SHORT[i], {})
        vals = [comp.get('capacity',0), comp.get('latency',0),
                comp.get('pkt_loss',0), comp.get('safety',0)]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', lw=2, markersize=5, label=LINK_SHORT[i],
                color=LINK_COLORS[i])
        ax.fill(angles, vals, alpha=0.05, color=LINK_COLORS[i])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Health Components per Link — {L}', fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=7)
    fig.tight_layout()
    save(fig, d/'fig32_health_radar.png')

# ============================================================================
#  FIG 33: Convergence Speed — Episodes to reach MSR > 0.8
# ============================================================================
def fig33(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    hist = data['independent'].get('history', {})
    ep_to_80 = []
    for i in range(5):
        k = str(i)
        msr_data = hist.get(k, {}).get('msr', [])
        found = len(msr_data)  # default: never reached
        for j, v in enumerate(msr_data):
            if v >= 0.8:
                found = j; break
        ep_to_80.append(found)

    # Map eval index to episode
    total_eps = len(hist.get('0', {}).get('rewards', []))
    n_evals = len(hist.get('0', {}).get('msr', []))
    scale = total_eps / max(n_evals - 1, 1) if n_evals > 1 else 1

    ep_to_80_scaled = [int(e * scale) for e in ep_to_80]

    bars = ax.bar(LINK_SHORT, ep_to_80_scaled, color=LINK_COLORS, edgecolor='black', lw=0.5)
    for b, v in zip(bars, ep_to_80_scaled):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                f'{v}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Episode')
    ax.set_title(f'Convergence Speed: Episodes to MSR ≥ 0.8 — {L}', fontweight='bold')
    fig.tight_layout()
    save(fig, d/'fig33_convergence_speed.png')

# ============================================================================
#  FIG 34: Confusion Matrix — Aggregated (FedProx, all links)
# ============================================================================
def fig34(data, d, L):
    """Confusion Matrix: True Severity → Agent Action (= Predicted Class)."""
    fig, ax = plt.subplots(figsize=(IEEE_S + 1.5, 4.0))
    best = data['fedprox']
    total = np.zeros((3, 3), dtype=np.float64)
    for ln in LINK_NAMES:
        sam = best.get(ln, {}).get('severity_action_matrix', [[0]*3]*3)
        total += np.array(sam, dtype=np.float64)

    # Labels: rows = True (High/Med/Low), cols = Predicted (BLOCK/RATE_LIMIT/ALLOW)
    # Reorder to match standard confusion matrix: diagonal = correct
    # Mapping: High→BLOCK(2), Medium→RATE_LIMIT(1), Low→ALLOW(0)
    # severity_action_matrix is [sev][action] where sev: H=0,M=1,L=2 and action: ALLOW=0,RATE=1,BLOCK=2
    # Reorder so diagonal = correct: High→BLOCK, Med→RATE, Low→ALLOW
    # New order: rows=[High,Med,Low], cols=[BLOCK,RATE_LIMIT,ALLOW]
    cm = np.array([
        [total[0, 2], total[0, 1], total[0, 0]],  # High → [BLOCK, RATE, ALLOW]
        [total[1, 2], total[1, 1], total[1, 0]],  # Med  → [BLOCK, RATE, ALLOW]
        [total[2, 2], total[2, 1], total[2, 0]],  # Low  → [BLOCK, RATE, ALLOW]
    ])

    # Normalize by row
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    # Raw counts for annotation
    annot_text = np.empty_like(cm_norm, dtype=object)
    for i in range(3):
        for j in range(3):
            annot_text[i, j] = f'{cm_norm[i,j]:.3f}\n({int(cm[i,j]):,})'

    row_labels = ['High\n(Attack)', 'Medium\n(Suspicious)', 'Low\n(Benign)']
    col_labels = ['BLOCK\n(Correct→High)', 'RATE_LIMIT\n(Correct→Med)', 'ALLOW\n(Correct→Low)']

    sns.heatmap(cm_norm, annot=annot_text, fmt='', cmap='RdYlGn',
                xticklabels=col_labels, yticklabels=row_labels,
                ax=ax, vmin=0, vmax=1, linewidths=1.5, linecolor='white',
                cbar_kws={'label': 'Proportion', 'shrink': 0.8},
                annot_kws={'fontsize': 9, 'fontweight': 'bold'})
    ax.set_xlabel('Agent Action (Predicted)', fontweight='bold')
    ax.set_ylabel('True Severity', fontweight='bold')
    ax.set_title(f'Confusion Matrix (FedProx) — {L}', fontweight='bold')
    fig.tight_layout()
    save(fig, d/'fig34_confusion_matrix.png')

# ============================================================================
#  FIG 35: Per-Link Confusion Matrix (5 individual)
# ============================================================================
def fig35(data, d, L):
    best = data['fedprox']
    for li, (ln, ls) in enumerate(zip(LINK_NAMES, LINK_SHORT)):
        fig, ax = plt.subplots(figsize=(IEEE_S + 1.0, 3.8))
        sam = best.get(ln, {}).get('severity_action_matrix', [[0]*3]*3)
        raw = np.array(sam, dtype=np.float64)

        cm = np.array([
            [raw[0,2], raw[0,1], raw[0,0]],
            [raw[1,2], raw[1,1], raw[1,0]],
            [raw[2,2], raw[2,1], raw[2,0]],
        ])
        rs = cm.sum(axis=1, keepdims=True); rs[rs==0] = 1
        cm_norm = cm / rs

        annot = np.empty_like(cm_norm, dtype=object)
        for i in range(3):
            for j in range(3):
                annot[i,j] = f'{cm_norm[i,j]:.3f}\n({int(cm[i,j]):,})'

        sns.heatmap(cm_norm, annot=annot, fmt='', cmap='RdYlGn',
                    xticklabels=['BLOCK', 'RATE_LIM', 'ALLOW'],
                    yticklabels=['High', 'Medium', 'Low'],
                    ax=ax, vmin=0, vmax=1, linewidths=1.2, linecolor='white',
                    annot_kws={'fontsize': 9, 'fontweight': 'bold'})
        ax.set_xlabel('Action (Predicted)'); ax.set_ylabel('Severity (True)')
        ax.set_title(f'{ls}: Confusion Matrix — {L}', fontweight='bold')
        fig.tight_layout()
        save(fig, d/f'fig35{chr(97+li)}_{ln}_confusion.png')

# ============================================================================
#  FIG 36: Classification Report — Precision, Recall, F1 per Class
# ============================================================================
def fig36(data, d, L):
    """Compute Precision, Recall, F1 from severity_action_matrix."""
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.8))
    best = data['fedprox']
    total = np.zeros((3, 3), dtype=np.float64)
    for ln in LINK_NAMES:
        sam = best.get(ln, {}).get('severity_action_matrix', [[0]*3]*3)
        total += np.array(sam, dtype=np.float64)

    # Reorder: rows=[H,M,L], cols=[BLOCK,RATE,ALLOW] so diagonal=correct
    cm = np.array([
        [total[0,2], total[0,1], total[0,0]],
        [total[1,2], total[1,1], total[1,0]],
        [total[2,2], total[2,1], total[2,0]],
    ])

    classes = ['High→BLOCK', 'Med→RATE_LIM', 'Low→ALLOW']
    precision, recall, f1 = [], [], []
    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp    # other classes predicted as c
        fn = cm[c, :].sum() - tp    # class c predicted as other
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precision.append(p); recall.append(r); f1.append(f)

    x = np.arange(3)
    width = 0.22
    bars_p = ax.bar(x - width, precision, width, label='Precision',
                    color='#3498DB', edgecolor='black', lw=0.5)
    bars_r = ax.bar(x,         recall,    width, label='Recall',
                    color='#E67E22', edgecolor='black', lw=0.5)
    bars_f = ax.bar(x + width, f1,        width, label='F1-Score',
                    color='#27AE60', edgecolor='black', lw=0.5)

    for bars in [bars_p, bars_r, bars_f]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+0.01, f'{h:.3f}',
                    ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.15)
    ax.set_title(f'Classification Report (FedProx) — {L}', fontweight='bold')
    ax.legend(); ax.set_axisbelow(True); fig.tight_layout()
    save(fig, d/'fig36_classification_report.png')

# ============================================================================
#  FIG 37: Per-Phase Accuracy (diagonal sum / total)
# ============================================================================
def fig37(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_S + 2, 3.5))
    accuracies = []
    for pn, pk in zip(PHASE_NAMES, PHASE_KEYS):
        total = np.zeros((3, 3), dtype=np.float64)
        for ln in LINK_NAMES:
            sam = data[pk].get(ln, {}).get('severity_action_matrix', [[0]*3]*3)
            total += np.array(sam, dtype=np.float64)
        # Correct: High→BLOCK(col2), Med→RATE(col1), Low→ALLOW(col0)
        correct = total[0,2] + total[1,1] + total[2,0]
        total_sum = total.sum()
        acc = correct / total_sum if total_sum > 0 else 0
        accuracies.append(acc)

    colors = [PHASE_COLORS[p] for p in PHASE_NAMES]
    bars = ax.bar(PHASE_NAMES, accuracies, color=colors, edgecolor='black', lw=0.5, width=0.5)
    for b, v in zip(bars, accuracies):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f'{v:.4f}\n({v*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.15)
    ax.set_title(f'Decision Accuracy per Phase — {L}', fontweight='bold')
    fig.tight_layout()
    save(fig, d/'fig37_accuracy_per_phase.png')

# ============================================================================
#  FIG 38: Confusion Matrix Comparison (Independent vs FedProx)
# ============================================================================
def fig38(data, d, L):
    """Difference heatmap: FedProx - Independent confusion."""
    fig, ax = plt.subplots(figsize=(IEEE_S + 1.5, 4.0))

    def get_cm_norm(pk):
        total = np.zeros((3, 3), dtype=np.float64)
        for ln in LINK_NAMES:
            sam = data[pk].get(ln, {}).get('severity_action_matrix', [[0]*3]*3)
            total += np.array(sam, dtype=np.float64)
        cm = np.array([
            [total[0,2], total[0,1], total[0,0]],
            [total[1,2], total[1,1], total[1,0]],
            [total[2,2], total[2,1], total[2,0]],
        ])
        rs = cm.sum(axis=1, keepdims=True); rs[rs==0] = 1
        return cm / rs

    cm_indep = get_cm_norm('independent')
    cm_fedprox = get_cm_norm('fedprox')
    diff = cm_fedprox - cm_indep

    annot = np.empty_like(diff, dtype=object)
    for i in range(3):
        for j in range(3):
            annot[i,j] = f'{diff[i,j]:+.3f}'

    sns.heatmap(diff, annot=annot, fmt='', cmap='RdBu_r', center=0,
                xticklabels=['BLOCK', 'RATE_LIM', 'ALLOW'],
                yticklabels=['High', 'Medium', 'Low'],
                ax=ax, vmin=-0.3, vmax=0.3, linewidths=1.2, linecolor='white',
                cbar_kws={'label': 'Difference (FedProx − Independent)'},
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    ax.set_xlabel('Action (Predicted)'); ax.set_ylabel('Severity (True)')
    ax.set_title(f'Confusion Diff: FedProx − Independent — {L}', fontweight='bold')
    fig.tight_layout()
    save(fig, d/'fig38_confusion_diff.png')

# ============================================================================
#  FIG 39: Per-Link F1-Score across Phases
# ============================================================================
def fig39(data, d, L):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    x = np.arange(5); width = 0.22

    for pi, (pn, pk) in enumerate(zip(PHASE_NAMES, PHASE_KEYS)):
        f1_per_link = []
        for ln in LINK_NAMES:
            sam = data[pk].get(ln, {}).get('severity_action_matrix', [[0]*3]*3)
            raw = np.array(sam, dtype=np.float64)
            cm = np.array([
                [raw[0,2], raw[0,1], raw[0,0]],
                [raw[1,2], raw[1,1], raw[1,0]],
                [raw[2,2], raw[2,1], raw[2,0]],
            ])
            # Macro F1
            f1s = []
            for c in range(3):
                tp = cm[c,c]
                fp = cm[:,c].sum() - tp
                fn = cm[c,:].sum() - tp
                p = tp/(tp+fp) if (tp+fp) > 0 else 0
                r = tp/(tp+fn) if (tp+fn) > 0 else 0
                f = 2*p*r/(p+r) if (p+r) > 0 else 0
                f1s.append(f)
            f1_per_link.append(np.mean(f1s))

        bars = ax.bar(x + pi*width, f1_per_link, width, label=pn,
                      color=PHASE_COLORS[pn], hatch=PHASE_HATCHES[pn],
                      edgecolor='black', lw=0.5, zorder=3)
        for b, v in zip(bars, f1_per_link):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f'{v:.2f}', ha='center', fontsize=6, fontweight='bold', rotation=45)

    ax.set_xticks(x + width); ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('Macro F1-Score'); ax.set_ylim(0, 1.15)
    ax.set_title(f'Per-Link Macro F1-Score — {L}', fontweight='bold')
    ax.legend(fontsize=7); ax.set_axisbelow(True); fig.tight_layout()
    save(fig, d/'fig39_per_link_f1.png')

# ============================================================================
#  CROSS-DATASET: fig_cross_01 — MSR Comparison
# ============================================================================
def fig_cross_01(all_data, d):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    ds_keys = list(all_data.keys())
    ds_labels = [DS_LABELS[k] for k in ds_keys]
    ds_colors = ['#2980B9', '#E67E22']
    x = np.arange(3); width = 0.3
    for di, dk in enumerate(ds_keys):
        vals = [get_global(all_data[dk], pk, 'mean_msr') for pk in PHASE_KEYS]
        bars = ax.bar(x + di*width, vals, width, label=ds_labels[di],
                      color=ds_colors[di], edgecolor='black', lw=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')
    ax.set_xticks(x + width/2); ax.set_xticklabels(PHASE_NAMES)
    ax.set_ylabel('MSR'); ax.set_ylim(0, 1.15)
    ax.set_title('Cross-Dataset: MSR Comparison', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig_cross_01_msr.png')

# ============================================================================
#  CROSS-DATASET: fig_cross_02 — FPR Comparison
# ============================================================================
def fig_cross_02(all_data, d):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    ds_keys = list(all_data.keys())
    ds_labels = [DS_LABELS[k] for k in ds_keys]
    ds_colors = ['#2980B9', '#E67E22']
    x = np.arange(3); width = 0.3
    for di, dk in enumerate(ds_keys):
        vals = [get_global(all_data[dk], pk, 'mean_fpr') for pk in PHASE_KEYS]
        ax.bar(x + di*width, vals, width, label=ds_labels[di],
               color=ds_colors[di], edgecolor='black', lw=0.5)
    ax.set_xticks(x + width/2); ax.set_xticklabels(PHASE_NAMES)
    ax.set_ylabel('FPR')
    ax.set_title('Cross-Dataset: FPR Comparison', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig_cross_02_fpr.png')

# ============================================================================
#  CROSS-DATASET: fig_cross_03 — Health Scores
# ============================================================================
def fig_cross_03(all_data, d):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    ds_keys = list(all_data.keys())
    ds_labels = [DS_LABELS[k] for k in ds_keys]
    ds_colors = ['#2980B9', '#E67E22']
    x = np.arange(5); width = 0.35
    for di, dk in enumerate(ds_keys):
        hm = all_data[dk]['master'].get('health_mean', [0]*5)
        bars = ax.bar(x + di*width, hm, width, label=ds_labels[di],
                      color=ds_colors[di], edgecolor='black', lw=0.5)
        for b, v in zip(bars, hm):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f'{v:.3f}', ha='center', fontsize=6, fontweight='bold')
    ax.set_xticks(x + width/2); ax.set_xticklabels(LINK_SHORT, fontsize=8)
    ax.set_ylabel('Health Score'); ax.set_ylim(0, 1.05)
    ax.set_title('Cross-Dataset: Master Health Scores', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig_cross_03_health.png')

# ============================================================================
#  CROSS-DATASET: fig_cross_04 — JFI
# ============================================================================
def fig_cross_04(all_data, d):
    fig, ax = plt.subplots(figsize=(IEEE_S + 2, 3.5))
    ds_keys = list(all_data.keys())
    ds_labels = [DS_LABELS[k] for k in ds_keys]
    ds_colors = ['#2980B9', '#E67E22']
    x = np.arange(2); width = 0.3
    for di, dk in enumerate(ds_keys):
        ms = all_data[dk]['master']
        vals = [ms.get('jfi_before_mean', 0), ms.get('jfi_after_mean', 0)]
        bars = ax.bar(x + di*width, vals, width, label=ds_labels[di],
                      color=ds_colors[di], edgecolor='black', lw=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                    f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x + width/2); ax.set_xticklabels(['Before Master', 'After Master'])
    ax.set_ylabel('JFI'); ax.set_ylim(0.85, 1.02)
    ax.set_title('Cross-Dataset: JFI Comparison', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig_cross_04_jfi.png')

# ============================================================================
#  CROSS-DATASET: fig_cross_05 — Radar
# ============================================================================
def fig_cross_05(all_data, d):
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    labels = ['MSR', '1−FPR', '1−FNR', 'JFI', 'Reward\n(×2.5)']
    n = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    ds_colors = ['#2980B9', '#E67E22']; ds_markers = ['o', 'D']
    for di, (dk, dd) in enumerate(all_data.items()):
        # find best phase
        best_pk = max(PHASE_KEYS, key=lambda pk: get_global(dd, pk, 'mean_msr'))
        g = dd[best_pk].get('global', {})
        vals = [g.get('mean_msr',0), 1-g.get('mean_fpr',0), 1-g.get('mean_fnr',0),
                g.get('jain_fairness',0), min(g.get('mean_reward',0)*2.5, 1.0)]
        vals += vals[:1]
        ax.plot(angles, vals, f'{ds_markers[di]}-', lw=2, markersize=6,
                label=f'{DS_LABELS[dk]} ({best_pk.title()})', color=ds_colors[di])
        ax.fill(angles, vals, alpha=0.1, color=ds_colors[di])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title('Cross-Dataset: Best Phase Radar', fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1))
    fig.tight_layout()
    save(fig, d/'fig_cross_05_radar.png')

# ============================================================================
#  CROSS-DATASET: fig_cross_06 — Ranking Side-by-Side
# ============================================================================
def fig_cross_06(all_data, d):
    ds_keys = list(all_data.keys())
    if len(ds_keys) < 2: return
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    x = np.arange(5); width = 0.35
    ds_colors = ['#2980B9', '#E67E22']
    for di, dk in enumerate(ds_keys):
        rp = all_data[dk]['master'].get('routing_prob_mean', {})
        vals = [rp.get(ls, 0.2) for ls in LINK_SHORT]
        bars = ax.bar(x + di*width, vals, width, label=DS_LABELS[dk],
                      color=ds_colors[di], edgecolor='black', lw=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                    f'{v:.1%}', ha='center', fontsize=6, fontweight='bold')
    ax.set_xticks(x + width/2); ax.set_xticklabels(LINK_SHORT)
    ax.set_ylabel('Routing Probability')
    ax.set_title('Cross-Dataset: Routing Probabilities', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig_cross_06_routing.png')


# ============================================================================
#  CROSS-DATASET: fig_cross_07 — Accuracy Comparison
# ============================================================================
def _compute_accuracy(data, pk):
    total = np.zeros((3, 3), dtype=np.float64)
    for ln in LINK_NAMES:
        sam = data[pk].get(ln, {}).get('severity_action_matrix', [[0]*3]*3)
        total += np.array(sam, dtype=np.float64)
    correct = total[0,2] + total[1,1] + total[2,0]
    s = total.sum()
    return correct / s if s > 0 else 0

def _compute_prf(data, pk):
    total = np.zeros((3, 3), dtype=np.float64)
    for ln in LINK_NAMES:
        sam = data[pk].get(ln, {}).get('severity_action_matrix', [[0]*3]*3)
        total += np.array(sam, dtype=np.float64)
    cm = np.array([
        [total[0,2], total[0,1], total[0,0]],
        [total[1,2], total[1,1], total[1,0]],
        [total[2,2], total[2,1], total[2,0]],
    ])
    precision, recall, f1 = [], [], []
    for c in range(3):
        tp = cm[c,c]; fp = cm[:,c].sum()-tp; fn = cm[c,:].sum()-tp
        p = tp/(tp+fp) if (tp+fp)>0 else 0
        r = tp/(tp+fn) if (tp+fn)>0 else 0
        f = 2*p*r/(p+r) if (p+r)>0 else 0
        precision.append(p); recall.append(r); f1.append(f)
    return np.mean(precision), np.mean(recall), np.mean(f1)

def fig_cross_07(all_data, d):
    """Accuracy per phase, cross-dataset."""
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    ds_keys = list(all_data.keys())
    ds_colors = ['#2980B9', '#E67E22']
    x = np.arange(3); width = 0.3
    for di, dk in enumerate(ds_keys):
        accs = [_compute_accuracy(all_data[dk], pk) for pk in PHASE_KEYS]
        bars = ax.bar(x + di*width, accs, width, label=DS_LABELS[dk],
                      color=ds_colors[di], edgecolor='black', lw=0.5)
        for b, v in zip(bars, accs):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x + width/2); ax.set_xticklabels(PHASE_NAMES)
    ax.set_ylabel('Decision Accuracy'); ax.set_ylim(0, 1.15)
    ax.set_title('Cross-Dataset: Decision Accuracy', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig_cross_07_accuracy.png')

# ============================================================================
#  CROSS-DATASET: fig_cross_08 — F1-Score Comparison
# ============================================================================
def fig_cross_08(all_data, d):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    ds_keys = list(all_data.keys())
    ds_colors = ['#2980B9', '#E67E22']
    x = np.arange(3); width = 0.3
    for di, dk in enumerate(ds_keys):
        f1s = [_compute_prf(all_data[dk], pk)[2] for pk in PHASE_KEYS]
        bars = ax.bar(x + di*width, f1s, width, label=DS_LABELS[dk],
                      color=ds_colors[di], edgecolor='black', lw=0.5)
        for b, v in zip(bars, f1s):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x + width/2); ax.set_xticklabels(PHASE_NAMES)
    ax.set_ylabel('Macro F1-Score'); ax.set_ylim(0, 1.15)
    ax.set_title('Cross-Dataset: Macro F1-Score', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig_cross_08_f1.png')

# ============================================================================
#  CROSS-DATASET: fig_cross_09 — Precision / Recall / F1 (best phase)
# ============================================================================
def fig_cross_09(all_data, d):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.8))
    ds_keys = list(all_data.keys())
    ds_colors = ['#2980B9', '#E67E22']
    classes = ['High→BLOCK', 'Med→RATE', 'Low→ALLOW']
    x = np.arange(3)
    width = 0.15

    for di, dk in enumerate(ds_keys):
        # Best phase by MSR
        best_pk = max(PHASE_KEYS, key=lambda pk: get_global(all_data[dk], pk, 'mean_msr'))
        total = np.zeros((3,3), dtype=np.float64)
        for ln in LINK_NAMES:
            sam = all_data[dk][best_pk].get(ln,{}).get('severity_action_matrix',[[0]*3]*3)
            total += np.array(sam, dtype=np.float64)
        cm = np.array([
            [total[0,2], total[0,1], total[0,0]],
            [total[1,2], total[1,1], total[1,0]],
            [total[2,2], total[2,1], total[2,0]],
        ])
        f1s = []
        for c in range(3):
            tp = cm[c,c]; fp = cm[:,c].sum()-tp; fn = cm[c,:].sum()-tp
            p = tp/(tp+fp) if (tp+fp)>0 else 0
            r = tp/(tp+fn) if (tp+fn)>0 else 0
            f = 2*p*r/(p+r) if (p+r)>0 else 0
            f1s.append(f)

        offset = di * (width * 1.2)
        bars = ax.bar(x + offset, f1s, width, label=f'{DS_LABELS[dk]} ({best_pk.title()})',
                      color=ds_colors[di], edgecolor='black', lw=0.5)
        for b, v in zip(bars, f1s):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')

    ax.set_xticks(x + width * 0.6); ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel('F1-Score'); ax.set_ylim(0, 1.15)
    ax.set_title('Cross-Dataset: Per-Class F1 (Best Phase)', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig_cross_09_per_class_f1.png')

# ============================================================================
#  CROSS-DATASET: fig_cross_10 — Per-Link F1 Comparison
# ============================================================================
def fig_cross_10(all_data, d):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    ds_keys = list(all_data.keys())
    ds_colors = ['#2980B9', '#E67E22']
    x = np.arange(5); width = 0.35

    for di, dk in enumerate(ds_keys):
        best_pk = max(PHASE_KEYS, key=lambda pk: get_global(all_data[dk], pk, 'mean_msr'))
        f1_per_link = []
        for ln in LINK_NAMES:
            sam = all_data[dk][best_pk].get(ln,{}).get('severity_action_matrix',[[0]*3]*3)
            raw = np.array(sam, dtype=np.float64)
            cm = np.array([
                [raw[0,2], raw[0,1], raw[0,0]],
                [raw[1,2], raw[1,1], raw[1,0]],
                [raw[2,2], raw[2,1], raw[2,0]],
            ])
            f1s = []
            for c in range(3):
                tp = cm[c,c]; fp = cm[:,c].sum()-tp; fn = cm[c,:].sum()-tp
                p = tp/(tp+fp) if (tp+fp)>0 else 0
                r = tp/(tp+fn) if (tp+fn)>0 else 0
                f = 2*p*r/(p+r) if (p+r)>0 else 0
                f1s.append(f)
            f1_per_link.append(np.mean(f1s))

        bars = ax.bar(x + di*width, f1_per_link, width, label=DS_LABELS[dk],
                      color=ds_colors[di], edgecolor='black', lw=0.5)
        for b, v in zip(bars, f1_per_link):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f'{v:.3f}', ha='center', fontsize=6, fontweight='bold')

    ax.set_xticks(x + width/2); ax.set_xticklabels(LINK_SHORT, fontsize=8)
    ax.set_ylabel('Macro F1'); ax.set_ylim(0, 1.15)
    ax.set_title('Cross-Dataset: Per-Link F1 (Best Phase)', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig_cross_10_per_link_f1.png')

# ============================================================================
#  CROSS-DATASET: fig_cross_11 — Reward Comparison
# ============================================================================
def fig_cross_11(all_data, d):
    fig, ax = plt.subplots(figsize=(IEEE_W, 3.5))
    ds_keys = list(all_data.keys())
    ds_colors = ['#2980B9', '#E67E22']
    x = np.arange(3); width = 0.3
    for di, dk in enumerate(ds_keys):
        vals = [get_global(all_data[dk], pk, 'mean_reward') for pk in PHASE_KEYS]
        bars = ax.bar(x + di*width, vals, width, label=DS_LABELS[dk],
                      color=ds_colors[di], edgecolor='black', lw=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')
    ax.set_xticks(x + width/2); ax.set_xticklabels(PHASE_NAMES)
    ax.set_ylabel('Mean Reward')
    ax.set_title('Cross-Dataset: Reward Comparison', fontweight='bold')
    ax.legend(); fig.tight_layout()
    save(fig, d/'fig_cross_11_reward.png')

# ============================================================================
#  CROSS-DATASET: fig_cross_12 — Summary Table
# ============================================================================
def fig_cross_12(all_data, d):
    fig, ax = plt.subplots(figsize=(IEEE_W, 4.5))
    ax.set_axis_off()
    ds_keys = list(all_data.keys())

    cols = ['Dataset', 'Phase', 'MSR', 'FPR', 'Accuracy', 'F1', 'JFI']
    rows = []
    for dk in ds_keys:
        for pn, pk in zip(PHASE_NAMES, PHASE_KEYS):
            g = all_data[dk][pk].get('global', {})
            acc = _compute_accuracy(all_data[dk], pk)
            _, _, f1 = _compute_prf(all_data[dk], pk)
            rows.append([
                DS_LABELS[dk], pn,
                f'{g.get("mean_msr",0):.4f}',
                f'{g.get("mean_fpr",0):.4f}',
                f'{acc:.4f}',
                f'{f1:.4f}',
                f'{g.get("jain_fairness",0):.4f}',
            ])

    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.6)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#1A252F')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # Highlight best MSR per dataset
    for di, dk in enumerate(ds_keys):
        best_ri = 0; best_val = 0
        for ri in range(3):
            row_idx = di * 3 + ri + 1
            val = float(rows[di*3 + ri][2])
            if val > best_val: best_val = val; best_ri = row_idx
        for j in range(len(cols)):
            tbl[best_ri, j].set_facecolor('#D5F5E3')

    ax.set_title('Cross-Dataset: Complete Results Summary', fontsize=13,
                 fontweight='bold', pad=15)
    fig.tight_layout()
    save(fig, d/'fig_cross_12_summary.png')


# ============================================================================
#  Main
# ============================================================================
def generate_for_dataset(ds_key, data, charts_dir):
    L = DS_LABELS[ds_key]
    os.makedirs(charts_dir, exist_ok=True)
    print(f'\n  Generating IEEE charts for {L}...')

    # Phase A: Training
    fig01(data, charts_dir, L)
    fig02(data, charts_dir, L)
    fig03(data, charts_dir, L)
    fig04(data, charts_dir, L)
    fig05(data, charts_dir, L)

    # Phase Comparison
    fig06(data, charts_dir, L)
    fig07(data, charts_dir, L)
    fig08(data, charts_dir, L)

    # FL Convergence
    fig09(data, charts_dir, L)
    fig10(data, charts_dir, L)
    fig11(data, charts_dir, L)
    fig12(data, charts_dir, L)
    fig13(data, charts_dir, L)
    fig14(data, charts_dir, L)

    # Agent Behavior
    fig15(data, charts_dir, L)
    fig16(data, charts_dir, L)  # 5 individual heatmaps
    fig17(data, charts_dir, L)
    fig18(data, charts_dir, L)
    fig19(data, charts_dir, L)

    # Statistics
    fig20(data, charts_dir, L)
    fig21(data, charts_dir, L)

    # Master Agent
    fig22(data, charts_dir, L)
    fig23(data, charts_dir, L)
    fig24(data, charts_dir, L)
    fig25(data, charts_dir, L)
    fig26(data, charts_dir, L)
    fig27(data, charts_dir, L)

    # Analysis
    fig28(data, charts_dir, L)
    fig29(data, charts_dir, L)
    fig30(data, charts_dir, L)
    fig31(data, charts_dir, L)
    fig32(data, charts_dir, L)
    fig33(data, charts_dir, L)

    # Confusion Matrix & Classification
    fig34(data, charts_dir, L)     # Aggregated confusion matrix
    fig35(data, charts_dir, L)     # 5 per-link confusion matrices
    fig36(data, charts_dir, L)     # Precision / Recall / F1
    fig37(data, charts_dir, L)     # Accuracy per phase
    fig38(data, charts_dir, L)     # Confusion diff (FedProx - Independent)
    fig39(data, charts_dir, L)     # Per-link Macro F1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', choices=['ciciot2023','diad2024','all'])
    args = parser.parse_args()

    print('='*70)
    print('  FL-RLMAMS Strategy A: IEEE Chart Regeneration v2')
    print('  NO retraining — cached results only')
    print('  Each figure = standalone (no subplots)')
    print('='*70)

    os.chdir(Path(__file__).resolve().parent.parent.parent)

    datasets = ['ciciot2023','diad2024'] if args.dataset == 'all' else [args.dataset]
    all_data = {}

    print(f'  Source: {RESULTS_SRC}')
    print(f'  Output: {RESULTS_OUT}')

    for dk in datasets:
        src_dir = RESULTS_SRC / dk
        if not src_dir.exists():
            print(f'\n  WARNING: {src_dir} not found'); continue
        print(f'\n  Loading {DS_LABELS[dk]}...')
        data = load_dataset(dk)
        if not data.get('independent') or not data.get('master'):
            print(f'  ERROR: Missing results for {dk}'); continue
        all_data[dk] = data
        out_charts = RESULTS_OUT / dk / 'charts'
        generate_for_dataset(dk, data, out_charts)

    if len(all_data) >= 2:
        cross_dir = RESULTS_OUT / 'cross_dataset'
        os.makedirs(cross_dir, exist_ok=True)
        print(f'\n  Generating cross-dataset comparisons...')
        fig_cross_01(all_data, cross_dir)
        fig_cross_02(all_data, cross_dir)
        fig_cross_03(all_data, cross_dir)
        fig_cross_04(all_data, cross_dir)
        fig_cross_05(all_data, cross_dir)
        fig_cross_06(all_data, cross_dir)
        fig_cross_07(all_data, cross_dir)
        fig_cross_08(all_data, cross_dir)
        fig_cross_09(all_data, cross_dir)
        fig_cross_10(all_data, cross_dir)
        fig_cross_11(all_data, cross_dir)
        fig_cross_12(all_data, cross_dir)

    print(f'\n{"="*70}')
    total = 0
    for dk in all_data:
        n = len(list((RESULTS_OUT / dk / 'charts').glob('fig*.png')))
        print(f'  {DS_LABELS[dk]}: {n} charts')
        total += n
    if len(all_data) >= 2:
        nc = len(list((RESULTS_OUT / 'cross_dataset').glob('*.png')))
        print(f'  Cross-dataset: {nc} charts')
        total += nc
    print(f'  TOTAL: {total} figures')
    print(f'  Saved to: {RESULTS_OUT}')
    print('='*70)


if __name__ == '__main__':
    main()
