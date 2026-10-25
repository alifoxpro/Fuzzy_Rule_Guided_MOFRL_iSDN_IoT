#!/usr/bin/env python
"""Aggregate the major-revision result artefacts into LaTeX fragments and a
JSON summary that we can splice straight into the manuscript.

Outputs (under experiments/paper_fragments/):
  * multiseed_5seed.tex   -- 4-seed mean-of-run replication table
  * ablation_6cell.tex    -- 6-cell ablation table (four cells reported)
  * major_revision_summary.json  -- machine-readable roll-up
"""

import json
from pathlib import Path
from statistics import mean, stdev

PROJECT = Path(__file__).resolve().parents[2]
RES_ROOT = PROJECT / 'experiments'
OUT_DIR = RES_ROOT / 'paper_fragments'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Where each configuration lives on disk.
SEED_FOLDERS = {
    'seed_42':   'results_StrategyB',
    'seed_123':  'results_StrategyB_s123',
    'seed_2026': 'results_StrategyB_s2026',
    'seed_7':    'results_StrategyB_seed_7',
    'seed_314':  'results_StrategyB_seed_314',
}

CELL_FOLDERS = {
    'e_complete':          'results_StrategyB',              # cell (e) full R_total
    'b_mosar_only':        'results_StrategyB_cell_b_mosar_only',
    'd_mosar_afras':       'results_StrategyB_cell_d_mosar_afras',
}

PHASE_DIRS = {
    'B2 (FedAvg)':  'mosar_fedavg',
    'B3 (FedProx)': 'mosar_fedprox',
    'B4 (Fuzzy)':   'fuzzy_retrain',
}

DATASETS = ['ciciot2023', 'diad2024']


def load_global(folder, dataset, phase_dir):
    p = RES_ROOT / folder / dataset / phase_dir / 'results.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())['global']
    except Exception as exc:  # pragma: no cover
        print(f'  warn: could not parse {p}: {exc}')
        return None


def summarise_seeds():
    out = {ds: {ph: {} for ph in PHASE_DIRS} for ds in DATASETS}
    for seed, folder in SEED_FOLDERS.items():
        for ds in DATASETS:
            for ph, ph_dir in PHASE_DIRS.items():
                g = load_global(folder, ds, ph_dir)
                if g is not None:
                    out[ds][ph][seed] = g
    return out


def emit_multiseed_table(seed_data):
    lines = [
        r'\begin{tabular}{@{}llccc@{}}',
        r'\toprule',
        r'Dataset & Phase & Mean MSR & $\pm\sigma$ (seed-level) & $n$ \\',
        r'\midrule',
    ]
    for ds in DATASETS:
        for ph in PHASE_DIRS:
            msrs = [seed_data[ds][ph][s]['mean_msr']
                    for s in SEED_FOLDERS if s in seed_data[ds][ph]]
            n = len(msrs)
            if n == 0:
                m_str, s_str = '--', '--'
            else:
                m_str = f'${mean(msrs):.4f}$'
                s_str = f'${stdev(msrs):.4f}$' if n > 1 else '$0.0000$'
            lines.append(f'  {ds:11s} & {ph:14s} & {m_str} & {s_str} & ${n}$ \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT_DIR / 'multiseed_5seed.tex').write_text('\n'.join(lines))
    print(f'wrote {OUT_DIR/"multiseed_5seed.tex"}')


def emit_ablation_table():
    lines = [
        r'\begin{tabular}{@{}lllccc@{}}',
        r'\toprule',
        r'Cell & Configuration & Dataset & B3 MSR & B4 MSR & B4 FPR \\',
        r'\midrule',
    ]
    cells_ordered = [
        ('a', 'conventional scalar reward',                     None),
        ('b', 'MOSAR only ($\\lambda=0,\\mu=0$)',               CELL_FOLDERS['b_mosar_only']),
        ('c', 'MOSAR $+ R_{\\text{link}}$ ($\\lambda=0$)',      None),
        ('d', 'MOSAR $+$ AFRAS-FL ($\\mu=0$)',                  CELL_FOLDERS['d_mosar_afras']),
        ('e', 'complete $R_{\\text{total}}$',                    CELL_FOLDERS['e_complete']),
        ('f', 'standard PBRS $\\equiv$ cell (c)',                None),
    ]
    for cid, name, folder in cells_ordered:
        for ds in DATASETS:
            if folder is None:
                b3 = b4 = fpr = None
            else:
                g3 = load_global(folder, ds, PHASE_DIRS['B3 (FedProx)'])
                g4 = load_global(folder, ds, PHASE_DIRS['B4 (Fuzzy)'])
                b3  = g3['mean_msr']  if g3 else None
                b4  = g4['mean_msr']  if g4 else None
                fpr = g4['mean_fpr']  if g4 else None
            fmt = lambda v: f'${v:.4f}$' if v is not None else '--'
            lines.append(f'  ({cid}) & {name:38s} & {ds:11s} & {fmt(b3)} & {fmt(b4)} & {fmt(fpr)} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT_DIR / 'ablation_6cell.tex').write_text('\n'.join(lines))
    print(f'wrote {OUT_DIR/"ablation_6cell.tex"}')


def main():
    seed_data = summarise_seeds()

    summary = {'seed_msr': {ds: {ph: {s: v['mean_msr']
                                       for s, v in seed_data[ds][ph].items()}
                                  for ph in seed_data[ds]}
                             for ds in DATASETS},
               'cells': {}}
    for cell, folder in CELL_FOLDERS.items():
        summary['cells'][cell] = {}
        for ds in DATASETS:
            summary['cells'][cell][ds] = {}
            for ph, ph_dir in PHASE_DIRS.items():
                g = load_global(folder, ds, ph_dir)
                if g:
                    summary['cells'][cell][ds][ph] = {
                        'mean_msr': g['mean_msr'],
                        'jain_fairness': g['jain_fairness'],
                        'mean_fpr': g['mean_fpr'],
                        'mean_fnr': g['mean_fnr'],
                    }
    (RES_ROOT / 'major_revision_summary.json').write_text(
        json.dumps(summary, indent=2))
    print(f'wrote {RES_ROOT/"major_revision_summary.json"}')

    emit_multiseed_table(seed_data)
    emit_ablation_table()


if __name__ == '__main__':
    main()
