#!/usr/bin/env python
"""Final aggregator for the closed Major-revision (Tier 1 + Tier 2 + Tier 3).

Outputs (under experiments/paper_fragments/):
  * multiseed_5seed_final.tex   -- 5 seeds mean-of-run per benchmark
  * ablation_6cell_final.tex    -- 6-cell ablation with cells (a)/(b)/(c)/(d)/(e)/(f)
  * lambda_mu_sweep.tex         -- 2D lambda x mu preliminary sweep
  * major_revision_summary_final.json
"""

import json
from pathlib import Path
from statistics import mean, stdev

PROJECT = Path(__file__).resolve().parents[2]
RES_ROOT = PROJECT / 'experiments'
OUT_DIR = RES_ROOT / 'paper_fragments'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Seeds available per benchmark
SEED_FOLDERS_CICIOT = {
    'seed_42':   'results_StrategyB',
    'seed_123':  'results_StrategyB_s123',
    'seed_2026': 'results_StrategyB_s2026',
    'seed_7':    'results_StrategyB_seed_7',
    'seed_314':  'results_StrategyB_seed_314',
}
SEED_FOLDERS_DIAD = {
    'seed_42':   'results_StrategyB',
    'seed_123':  'results_StrategyB_s123',
    'seed_2026': 'results_StrategyB_seed_2026',
    'seed_7':    'results_StrategyB_seed_7',
    'seed_314':  'results_StrategyB_seed_314',
}

CELL_FOLDERS = {
    'a_scalar':          'results_StrategyB_cell_a_scalar',
    'b_mosar_only':      'results_StrategyB_cell_b_mosar_only',
    'e_complete':        'results_StrategyB',
    'd_mosar_afras':     'results_StrategyB_cell_d_mosar_afras',
}

SWEEP_POINTS = [
    ('lam=0.10, mu=0.10', 'results_StrategyB_lam0.10'),
    ('lam=0.25, mu=0.10', 'results_StrategyB'),
    ('lam=0.50, mu=0.10', 'results_StrategyB_lam0.50'),
    ('lam=0.25, mu=0.05', 'results_StrategyB_lam0.25_mu0.05'),
    ('lam=0.25, mu=0.20', 'results_StrategyB_lam0.25_mu0.20'),
    ('lam=0.10, mu=0.20', 'results_StrategyB_lam0.10_mu0.20'),
]

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
    except Exception as exc:
        print(f'  warn: {p}: {exc}')
        return None


def emit_multiseed_table():
    lines = [
        r'\begin{tabular}{@{}llccc@{}}',
        r'\toprule',
        r'Dataset & Phase & Mean MSR & $\pm\sigma$ (seed-level) & $n$ \\',
        r'\midrule',
    ]
    for ds in DATASETS:
        seeds_map = SEED_FOLDERS_CICIOT if ds == 'ciciot2023' else SEED_FOLDERS_DIAD
        for ph, ph_dir in PHASE_DIRS.items():
            msrs = []
            for s, folder in seeds_map.items():
                g = load_global(folder, ds, ph_dir)
                if g:
                    msrs.append(g['mean_msr'])
            n = len(msrs)
            m_str = f'${mean(msrs):.4f}$' if n else '--'
            s_str = f'${stdev(msrs):.4f}$' if n > 1 else ('$0.0000$' if n else '--')
            lines.append(f'  {ds:11s} & {ph:14s} & {m_str} & {s_str} & ${n}$ \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT_DIR / 'multiseed_5seed_final.tex').write_text('\n'.join(lines))
    print(f'wrote {OUT_DIR/"multiseed_5seed_final.tex"}')


def emit_ablation_table():
    lines = [
        r'\begin{tabular}{@{}lllccc@{}}',
        r'\toprule',
        r'Cell & Configuration & Dataset & B3 MSR & B4 MSR & B4 FPR \\',
        r'\midrule',
    ]
    cells_ordered = [
        ('a', 'conventional scalar reward',                 CELL_FOLDERS['a_scalar']),
        ('b', 'MOSAR only ($\\lambda=0,\\mu=0$)',           CELL_FOLDERS['b_mosar_only']),
        ('c', 'MOSAR $+ R_{\\text{link}}$ ($\\lambda=0$)',  None),
        ('d', 'MOSAR $+$ AFRAS-FL ($\\mu=0$)',              CELL_FOLDERS['d_mosar_afras']),
        ('e', 'complete $R_{\\text{total}}$',                CELL_FOLDERS['e_complete']),
        ('f', 'standard PBRS $\\equiv$ cell (c)',            None),
    ]
    for cid, name, folder in cells_ordered:
        for ds in DATASETS:
            if folder is None:
                b3 = b4 = fpr = None
            else:
                g3 = load_global(folder, ds, PHASE_DIRS['B3 (FedProx)'])
                g4 = load_global(folder, ds, PHASE_DIRS['B4 (Fuzzy)'])
                b3  = g3['mean_msr'] if g3 else None
                b4  = g4['mean_msr'] if g4 else None
                fpr = g4['mean_fpr'] if g4 else None
            fmt = lambda v: f'${v:.4f}$' if v is not None else '--'
            lines.append(f'  ({cid}) & {name:38s} & {ds:11s} & {fmt(b3)} & {fmt(b4)} & {fmt(fpr)} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT_DIR / 'ablation_6cell_final.tex').write_text('\n'.join(lines))
    print(f'wrote {OUT_DIR/"ablation_6cell_final.tex"}')


def emit_sweep_table():
    lines = [
        r'\begin{tabular}{@{}lcccc@{}}',
        r'\toprule',
        r'$(\lambda, \mu)$ configuration & B4 MSR & B4 JFI & B4 FPR & B4 Reward \\',
        r'\midrule',
    ]
    for name, folder in SWEEP_POINTS:
        g = load_global(folder, 'ciciot2023', PHASE_DIRS['B4 (Fuzzy)'])
        if g:
            lines.append(f'  {name:22s} & ${g["mean_msr"]:.4f}$ & ${g["jain_fairness"]:.4f}$ & ${g["mean_fpr"]:.4f}$ & ${g.get("mean_reward", 0):.4f}$ \\\\')
        else:
            lines.append(f'  {name:22s} & -- & -- & -- & -- \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT_DIR / 'lambda_mu_sweep.tex').write_text('\n'.join(lines))
    print(f'wrote {OUT_DIR/"lambda_mu_sweep.tex"}')


def main():
    emit_multiseed_table()
    emit_ablation_table()
    emit_sweep_table()

    summary = {'seeds': {}, 'cells': {}, 'sweep': {}}
    for ds in DATASETS:
        seeds_map = SEED_FOLDERS_CICIOT if ds == 'ciciot2023' else SEED_FOLDERS_DIAD
        summary['seeds'][ds] = {}
        for ph, ph_dir in PHASE_DIRS.items():
            summary['seeds'][ds][ph] = {
                s: load_global(f, ds, ph_dir).get('mean_msr') if load_global(f, ds, ph_dir) else None
                for s, f in seeds_map.items()
            }
    for cell, folder in CELL_FOLDERS.items():
        summary['cells'][cell] = {}
        for ds in DATASETS:
            summary['cells'][cell][ds] = {}
            for ph, ph_dir in PHASE_DIRS.items():
                g = load_global(folder, ds, ph_dir)
                if g:
                    summary['cells'][cell][ds][ph] = {'msr': g['mean_msr'], 'fpr': g['mean_fpr'], 'jfi': g['jain_fairness']}
    for name, folder in SWEEP_POINTS:
        g = load_global(folder, 'ciciot2023', PHASE_DIRS['B4 (Fuzzy)'])
        if g:
            summary['sweep'][name] = {'msr': g['mean_msr'], 'fpr': g['mean_fpr'], 'jfi': g['jain_fairness']}
    (RES_ROOT / 'major_revision_summary_final.json').write_text(json.dumps(summary, indent=2))
    print(f'wrote {RES_ROOT/"major_revision_summary_final.json"}')


if __name__ == '__main__':
    main()
