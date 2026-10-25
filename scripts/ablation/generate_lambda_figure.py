# -*- coding: utf-8 -*-
"""
============================================================================
Generate the lambda sensitivity figure for paper 4.

Reads:  paper4-Fuzzy-Augmented-Reward-Shaping/ablation/lambda_ablation.json
Writes: paper4-Fuzzy-Augmented-Reward-Shaping/figures/ablation/
            fig_lambda_sensitivity.png
            fig_lambda_sensitivity.pdf

The figure has three panels (left -> right):
  (a) MSR vs lambda, both datasets, with error bars (one sigma)
  (b) JFI vs lambda
  (c) FPR vs lambda (log scale on Y)

Only the B4 phase (fuzzy-augmented) is plotted -- that is the phase whose
reward depends on lambda; the other phases are identical across runs.

Usage:
    python scripts/ablation/generate_lambda_figure.py
============================================================================
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
PAPER = PROJECT / 'paper4-Fuzzy-Augmented-Reward-Shaping'
ABL_DIR = PAPER / 'ablation'
FIG_DIR = PAPER / 'figures' / 'ablation'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# IEEE-style figure parameters to match the rest of the paper
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

DATASET_COLORS = {
    'ciciot2023': '#E74C3C',
    'diad2024':   '#3498DB',
}
DATASET_LABELS = {
    'ciciot2023': 'CICIoT2023',
    'diad2024':   'DIAD2024',
}
DATASET_MARKERS = {
    'ciciot2023': 'o',
    'diad2024':   's',
}


def load_rows():
    path = ABL_DIR / 'lambda_ablation.json'
    if not path.exists():
        print(f'ERROR: {path} does not exist. Run aggregate_lambda_results.py first.')
        sys.exit(1)
    with open(path, 'r') as f:
        data = json.load(f)
    return data['rows']


def group_by_dataset(rows, phase='phase_b4'):
    """Return {dataset: sorted list of (lam, MSR, std, JFI, FPR)} for the
    given phase only."""
    groups = {}
    for r in rows:
        if r.get('phase') != phase:
            continue
        ds = r['dataset']
        groups.setdefault(ds, []).append(
            (r['lambda'], r['MSR'], r['MSR_std'], r['JFI'], r['FPR'])
        )
    for ds in groups:
        groups[ds].sort(key=lambda t: t[0])
    return groups


def plot_panel(ax, groups, y_index, y_label, err_index=None, logy=False):
    for ds, rows in groups.items():
        if not rows:
            continue
        lams = [r[0] for r in rows]
        ys = [r[y_index] if r[y_index] is not None else np.nan for r in rows]
        if err_index is not None:
            errs = [r[err_index] if r[err_index] is not None else 0.0 for r in rows]
            ax.errorbar(
                lams, ys, yerr=errs,
                color=DATASET_COLORS[ds],
                marker=DATASET_MARKERS[ds],
                markersize=7,
                linewidth=2,
                capsize=4,
                label=DATASET_LABELS[ds],
            )
        else:
            ax.plot(
                lams, ys,
                color=DATASET_COLORS[ds],
                marker=DATASET_MARKERS[ds],
                markersize=7,
                linewidth=2,
                label=DATASET_LABELS[ds],
            )
    ax.set_xlabel(r'Fuzzy shaping weight $\lambda$')
    ax.set_ylabel(y_label)
    if logy:
        ax.set_yscale('log')
    # Mark the paper's default lambda = 0.25
    ax.axvline(0.25, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(
        0.25, ax.get_ylim()[1] if not logy else ax.get_ylim()[1],
        r' $\lambda^{\star}=0.25$',
        color='gray', fontsize=8, va='top', ha='left',
    )
    ax.legend(loc='best', framealpha=0.85)


def main():
    rows = load_rows()
    groups = group_by_dataset(rows, phase='phase_b4')

    if not groups:
        print('ERROR: no phase_b4 rows found in lambda_ablation.json.')
        print('       Make sure the sweep has completed and the aggregator'
              ' has run.')
        sys.exit(1)

    # Report what we found
    print('Lambda values found per dataset:')
    for ds, rs in groups.items():
        lams = [r[0] for r in rs]
        print(f'  {ds}: {lams}')

    # Build the 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    # Panel (a): MSR with error bars
    plot_panel(axes[0], groups, y_index=1, y_label='Mean Success Rate (MSR)',
               err_index=2)
    axes[0].set_title('(a) MSR sensitivity')

    # Panel (b): JFI
    plot_panel(axes[1], groups, y_index=3, y_label='Jain Fairness Index (JFI)',
               err_index=None)
    axes[1].set_title('(b) Fairness sensitivity')

    # Panel (c): FPR (log scale)
    plot_panel(axes[2], groups, y_index=4, y_label='False-Positive Rate (FPR)',
               err_index=None, logy=True)
    axes[2].set_title('(c) FPR sensitivity')

    plt.tight_layout()

    png_path = FIG_DIR / 'fig_lambda_sensitivity.png'
    pdf_path = FIG_DIR / 'fig_lambda_sensitivity.pdf'
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    print()
    print(f'Wrote: {png_path}')
    print(f'Wrote: {pdf_path}')


if __name__ == '__main__':
    main()
