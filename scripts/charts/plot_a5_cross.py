# -*- coding: utf-8 -*-
"""Unified A5 round-drift figure for the results section.

Reads the four a5_round_drift.csv profiles (2 datasets x 2 aggregators) and
renders a 2-panel figure:
  (a) Balanced accuracy vs FL round   -- detection quality + ranking flip
  (b) False-alarm rate (FPR) vs round -- false-alarm instability / collapse

Colour = dataset, line style = aggregator. Saves PNG + PDF (vector for LaTeX).
Run:  python scripts/charts/plot_a5_cross.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent.parent / 'experiments' / 'results_A5run'
OUT = BASE

CONFIGS = [
    # (dataset_key, phase, label, colour, linestyle)
    ('ciciot2023', 'fedavg',  'CICIoT2023 · FedAvg',  '#2C6FBB', '-'),
    ('ciciot2023', 'fedprox', 'CICIoT2023 · FedProx', '#2C6FBB', '--'),
    ('diad2024',   'fedavg',  'DIAD2024 · FedAvg',    '#D1495B', '-'),
    ('diad2024',   'fedprox', 'DIAD2024 · FedProx',   '#D1495B', '--'),
]


def load(ds, ph):
    p = BASE / ds / ph / 'a5_round_drift.csv'
    rows = list(csv.DictReader(open(p)))
    r = [int(x['round']) for x in rows]
    bal = [float(x['balanced']) for x in rows]
    fpr = [float(x['fpr']) for x in rows]
    loss = [float(x['mean_loss']) if x.get('mean_loss') not in (None, '')
            else float('nan') for x in rows]
    return r, bal, fpr, loss


def main():
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'legend.fontsize': 8, 'figure.dpi': 200,
    })
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))

    for ds, ph, label, c, ls in CONFIGS:
        r, bal, fpr, loss = load(ds, ph)
        ax1.plot(r, bal, ls, color=c, marker='o', ms=3, lw=1.6, label=label)
        ax2.plot(r, fpr, ls, color=c, marker='o', ms=3, lw=1.6, label=label)
        ax3.plot(r, loss, ls, color=c, marker='o', ms=3, lw=1.6, label=label)

    ax1.set_title('(a) Detection quality — balanced accuracy')
    ax1.set_xlabel('FL communication round')
    ax1.set_ylabel('Balanced accuracy  (higher = better)')
    ax1.set_ylim(0.35, 0.78)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')

    ax2.set_title('(b) False alarm — FPR')
    ax2.set_xlabel('FL communication round')
    ax2.set_ylabel('False-alarm rate  (lower = better)')
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.3)
    # annotate the CICIoT2023-FedAvg mid-training collapse at round 7
    ax2.annotate('mid-training\ndetection collapse\n(round 7)',
                 xy=(7, 0.0), xytext=(11, 0.22),
                 arrowprops=dict(arrowstyle='->', color='#2C6FBB', lw=1.2),
                 fontsize=7.5, color='#2C6FBB')

    ax3.set_title('(c) Model convergence — training loss')
    ax3.set_xlabel('FL communication round')
    ax3.set_ylabel('Mean training loss  (log scale)')
    ax3.set_yscale('log')
    ax3.grid(True, which='both', alpha=0.3)
    ax3.legend(loc='upper right')
    # shade the "converged" region: loss is essentially flat from ~round 4 on,
    # while panels (a)/(b) keep drifting -> convergence != detection.
    ax3.axvspan(4, 30, color='0.75', alpha=0.25)
    ax3.annotate('loss converged & flat,\nyet detection (a,b) still drifts',
                 xy=(17, 0.24), xytext=(9, 0.42),
                 fontsize=7.5, color='0.25')

    fig.tight_layout()
    png = OUT / 'a5_cross_dataset.png'
    pdf = OUT / 'a5_cross_dataset.pdf'
    fig.savefig(png, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print('Wrote:')
    print('  ', png)
    print('  ', pdf)


if __name__ == '__main__':
    main()
