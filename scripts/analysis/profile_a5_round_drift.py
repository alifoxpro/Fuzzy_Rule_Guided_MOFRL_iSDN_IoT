# -*- coding: utf-8 -*-
"""
============================================================================
A5 — Round-Drift Reliability Profiling (offline)
============================================================================
Re-evaluates the PURE global FL model saved at every round (round_ckpts/
global_round_NN.pt) to measure the round-drift axis:

  A5_drift = std of detection sensitivity over the post-convergence tail
  A5_gap   = sensitivity(best-detection round) - sensitivity(best-loss round)

This decouples MODEL convergence (mean training loss) from DETECTION
convergence (sensitivity / false-alarm), i.e. the headline result of the
profiling paper: "model convergence != detection convergence".

It does NOT retrain. It reuses the exact env-construction and evaluate_agent
of the training module that produced the checkpoints, so the numbers are
directly comparable to the online run.

Prerequisites (produced by the patched training loops):
  <phase_dir>/round_ckpts/global_round_01.pt ... _NN.pt
  <phase_dir>/results.json   (history.mean_loss — optional but needed for A5_gap)

Usage:
  python scripts/analysis/profile_a5_round_drift.py \
      --module base --dataset ciciot2023 \
      --phase-dir experiments/results/phase4_rl/ciciot2023/fedavg

  --module {base|A|B}   which training script produced the checkpoints
  --dataset {ciciot2023|diad2024}
  --phase-dir DIR       directory containing round_ckpts/ and results.json
  --episodes N          eval episodes per link per round (default 10)
  --tail-eps F          convergence-tail loss threshold, fraction of loss
                        range (default 0.10); tail = rounds with
                        loss <= Lmin + F*(Lmax-Lmin)
  --ds-dir DIR          override dataset dir holding mamba_cache/ & afras_fl/
                        (default: parent of --phase-dir)
============================================================================
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import argparse
import importlib
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN_DIR = Path(__file__).resolve().parent.parent / 'training'
sys.path.insert(0, str(TRAIN_DIR))

MODULE_MAP = {
    'base': 'train_rl_multiagent',
    'A': 'train_rl_multiagent_StrategyA',
    'B': 'train_rl_multiagent_StrategyB',
}


# ---------------------------------------------------------------------------
# Env construction (faithful per-module mirror of train_dataset)
# ---------------------------------------------------------------------------
def build_envs_base(mod, dataset_key, ds_dir):
    """Mirror env-building of train_rl_multiagent.py / _StrategyA.py."""
    from sklearn.preprocessing import StandardScaler

    ds_cfg = mod.DATASETS[dataset_key]
    with open(ds_cfg['json_file'], 'r') as f:
        feat_info = json.load(f)
    feature_order = feat_info['selected_features'] if isinstance(feat_info, dict) \
        else feat_info

    data_dir = ds_cfg['data_dir']
    link_files = sorted(data_dir.glob('Link_*.csv'))
    cache_dir = ds_dir / 'mamba_cache'

    envs = []
    for li, link_file in enumerate(link_files):
        link_name = link_file.stem
        X_raw, y_raw = mod.load_link_csv(link_file, feature_order)
        X_win, y_win = mod.create_windows(X_raw, y_raw, mod.WINDOW_SIZE, mod.STRIDE)

        # Scale (base/A scale windowed features per link)
        n_feat = X_win.shape[2]
        scaler = StandardScaler()
        X_flat = scaler.fit_transform(X_win.reshape(-1, n_feat))
        X_win = X_flat.reshape(X_win.shape).astype(np.float32)

        cache_softmax = cache_dir / f'{link_name}_softmax.npy'
        cache_embed = cache_dir / f'{link_name}_embed.npy'
        if not (cache_softmax.exists() and cache_embed.exists()):
            raise FileNotFoundError(
                f'Missing Mamba cache for {link_name} in {cache_dir}. '
                f'Run training first so the cache is populated.')
        softmax = np.load(cache_softmax)
        embeddings = np.load(cache_embed)

        # PCA 128D -> 16D, fit on same subset base training used (<=50000)
        from sklearn.decomposition import PCA
        sub = embeddings[:min(50000, len(embeddings))]
        pca = PCA(n_components=16, random_state=mod.SEED)
        pca.fit(sub)
        embed_pca = pca.transform(embeddings).astype(np.float32)

        env = mod.SDNEnvironment(X_win, y_win, softmax, embed_pca,
                                 link_idx=li, feature_order=feature_order)
        envs.append(env)
    return envs


def build_envs_strategyB(mod, dataset_key, ds_dir):
    """Mirror env-building of train_rl_multiagent_StrategyB.py (MOSAR+Fuzzy)."""
    from sklearn.decomposition import PCA

    ds_cfg = mod.DATASETS[dataset_key]
    with open(ds_cfg['json_file'], 'r') as f:
        feat_info = json.load(f)
    if isinstance(feat_info, dict):
        feature_order = feat_info.get('selected_features', list(feat_info.keys()))
    else:
        feature_order = feat_info

    cache_dir = ds_dir / 'mamba_cache'
    afras_base = ds_dir / 'afras_fl'

    envs = []
    for i, link_name in enumerate(mod.LINK_NAMES):
        csv_path = ds_cfg['data_dir'] / f'{link_name}.csv'
        X, y = mod.load_link_csv(csv_path, feature_order)
        X_w, y_w = mod.create_windows(X, y, mod.WINDOW_SIZE, mod.STRIDE)

        cache_softmax = cache_dir / f'{link_name}_softmax.npy'
        cache_embed = cache_dir / f'{link_name}_embed.npy'
        if not (cache_softmax.exists() and cache_embed.exists()):
            raise FileNotFoundError(
                f'Missing Mamba cache for {link_name} in {cache_dir}. '
                f'Run training first so the cache is populated.')
        softmax = np.load(cache_softmax)
        embed_raw = np.load(cache_embed)

        pca = PCA(n_components=16, random_state=mod.SEED)
        pca.fit(embed_raw)
        embed_pca = pca.transform(embed_raw).astype(np.float32)

        # Fuzzy engine from cached AFRAS-FL deployment (Phase B0 output)
        deploy_file = afras_base / link_name / 'phase7_deployed.json'
        if deploy_file.exists():
            with open(deploy_file, 'r') as f:
                pkg = json.load(f)
            fuzzy_engine = mod.FuzzyInferenceEngine(
                rules=pkg.get('rules', []),
                membership_params=pkg.get('membership_params', {}),
                feature_names=pkg.get('feature_names', []),
            )
        else:
            print(f'  [warn] no AFRAS deployment for {link_name}; empty fuzzy engine')
            fuzzy_engine = mod.FuzzyInferenceEngine([], {}, [])

        y_win = np.array([softmax[j].argmax() for j in range(len(softmax))],
                         dtype=np.int64)
        env = mod.SDNEnvironmentB(
            X_windows=X_w, y_windows=y_win,
            mamba_softmax=softmax, mamba_embed_pca=embed_pca,
            link_idx=i, feature_order=feature_order,
            fuzzy_engine=fuzzy_engine, use_mosar=True, use_fuzzy_reward=False,
        )
        envs.append(env)
    return envs


# ---------------------------------------------------------------------------
# A5 core
# ---------------------------------------------------------------------------
def round_of(ckpt_path):
    # global_round_07.pt -> 7
    stem = ckpt_path.stem
    return int(stem.split('_')[-1])


def profile(mod, envs, phase_dir, episodes):
    """Evaluate the pure global model at every saved round."""
    ckpt_dir = phase_dir / 'round_ckpts'
    ckpts = sorted(ckpt_dir.glob('global_round_*.pt'), key=round_of)
    if not ckpts:
        raise FileNotFoundError(f'No round checkpoints in {ckpt_dir}')

    agent = mod.DQNAgent()
    rows = []
    for ck in ckpts:
        rd = round_of(ck)
        params = torch.load(ck, weights_only=False, map_location=mod.DEVICE)
        agent.set_params(params)

        # Re-seed before each round so episode sampling is identical across
        # rounds; differences then reflect the model, not the sampling.
        mod.seed_everything(mod.SEED)

        fpr_l, fnr_l, msr_l = [], [], []
        for env in envs:
            m = mod.evaluate_agent(agent, env, n_episodes=episodes)
            fpr_l.append(m['fpr'])
            fnr_l.append(m['fnr'])
            msr_l.append(m['msr'])
        fpr = float(np.mean(fpr_l))
        fnr = float(np.mean(fnr_l))
        msr = float(np.mean(msr_l))
        rows.append({
            'round': rd,
            'sensitivity': 1.0 - fnr,   # TPR on attacks = 1 - FNR
            'fpr': fpr,
            'fnr': fnr,
            'msr': msr,
        })
        print(f'  round {rd:2d}: sens={1.0 - fnr:.4f} fpr={fpr:.4f} msr={msr:.4f}')
    return rows


def load_loss_series(phase_dir, rounds):
    """Return mean_loss aligned to `rounds`, or None if unavailable."""
    rj = phase_dir / 'results.json'
    if not rj.exists():
        return None
    with open(rj, 'r') as f:
        res = json.load(f)
    hist = res.get('history', {})
    loss = hist.get('mean_loss')
    hrounds = hist.get('rounds')
    if not loss or not hrounds:
        return None
    by_round = {int(r): float(l) for r, l in zip(hrounds, loss)
                if l is not None and np.isfinite(l)}
    return [by_round.get(r, np.nan) for r in rounds]


# Detection scalars scored as "higher is better". On attack-skewed data
# `sensitivity` (=1-FNR) saturates near 1.0 and hides round drift, so the
# primary metric defaults to `balanced` = 0.5*(sensitivity + specificity),
# which responds to BOTH missed attacks and false alarms.
HIGHER_BETTER = ('balanced', 'specificity', 'sensitivity', 'msr')


def metric_series(rows):
    """Derive all detection scalars per round from the raw fpr/fnr/msr rows."""
    sens = np.array([r['sensitivity'] for r in rows], dtype=float)   # 1 - FNR
    fpr = np.array([r['fpr'] for r in rows], dtype=float)
    msr = np.array([r['msr'] for r in rows], dtype=float)
    spec = 1.0 - fpr                                                 # specificity
    bal = 0.5 * (sens + spec)                                        # balanced acc
    return {'sensitivity': sens, 'specificity': spec, 'msr': msr,
            'balanced': bal, 'fpr': fpr}


def compute_a5(rows, loss_series, tail_frac=0.5, primary='balanced'):
    rounds = [r['round'] for r in rows]
    n = len(rounds)
    series = metric_series(rows)

    # Robust convergence tail: last `tail_frac` of rounds. Independent of the
    # loss curve (whose round-1 outlier otherwise distorts a range-based tail).
    k = max(2, int(round(tail_frac * n)))
    tail_mask = np.zeros(n, dtype=bool)
    tail_mask[-k:] = True

    # Best-loss round is only used for the convergence-vs-detection gap.
    if loss_series is not None and np.isfinite(np.asarray(loss_series)).any():
        loss = np.asarray(loss_series, dtype=float)
        best_loss_idx = int(np.nanargmin(loss))
        loss_available = True
    else:
        best_loss_idx = None
        loss_available = False

    def drift_gap(x):  # x: higher-is-better series
        best_idx = int(np.argmax(x))
        gap = (float('nan') if best_loss_idx is None
               else float(x[best_idx] - x[best_loss_idx]))
        return {
            'drift_tail_std': float(np.std(x[tail_mask])),
            'gap': gap,
            'min': float(np.min(x)), 'max': float(np.max(x)),
            'range': float(np.max(x) - np.min(x)),
            'mean_tail': float(np.mean(x[tail_mask])),
            'best_round': int(rounds[best_idx]),
            'value_at_best_loss_round': (None if best_loss_idx is None
                                         else float(x[best_loss_idx])),
            'final': float(x[-1]),
        }

    per_metric = {m: drift_gap(series[m]) for m in HIGHER_BETTER}
    # False-alarm drift reported explicitly (lower-is-better -> raw std/stats).
    fpr = series['fpr']
    per_metric['fpr'] = {
        'drift_tail_std': float(np.std(fpr[tail_mask])),
        'min': float(np.min(fpr)), 'max': float(np.max(fpr)),
        'range': float(np.max(fpr) - np.min(fpr)),
        'mean_tail': float(np.mean(fpr[tail_mask])),
        'final': float(fpr[-1]),
    }

    prim = per_metric[primary]
    return {
        'n_rounds': n,
        'primary_metric': primary,
        'tail_frac': tail_frac,
        'tail_rounds': [int(rounds[i]) for i in np.where(tail_mask)[0]],
        'loss_available': loss_available,
        'best_loss_round': (None if best_loss_idx is None
                            else int(rounds[best_loss_idx])),
        # Headline numbers, based on the non-saturating primary metric.
        'A5_drift_tail_std': prim['drift_tail_std'],
        'A5_gap': prim['gap'],
        'best_detection_round': prim['best_round'],
        'per_metric': per_metric,
    }


def save_outputs(phase_dir, rows, loss_series, summary):
    series = metric_series(rows)
    out_csv = phase_dir / 'a5_round_drift.csv'
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write('round,sensitivity,specificity,fpr,fnr,msr,balanced,mean_loss\n')
        for i, r in enumerate(rows):
            ls = '' if loss_series is None else (
                '' if not np.isfinite(loss_series[i]) else f'{loss_series[i]:.6f}')
            f.write(f"{r['round']},{r['sensitivity']:.6f},{series['specificity'][i]:.6f},"
                    f"{r['fpr']:.6f},{r['fnr']:.6f},{r['msr']:.6f},"
                    f"{series['balanced'][i]:.6f},{ls}\n")

    with open(phase_dir / 'a5_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Convergence-vs-detection figure, plotting the (non-saturating) primary metric.
    prim = summary.get('primary_metric', 'balanced')
    rounds = [r['round'] for r in rows]
    yprim = series[prim]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(rounds, yprim, 'o-', color='#27AE60', label=f'Detection: {prim}')
    ax1.plot(rounds, series['fpr'], '^-', color='#8E44AD', alpha=0.6,
             label='False alarm (FPR)')
    ax1.set_xlabel('FL round')
    ax1.set_ylabel('Detection quality', color='#27AE60')
    ax1.tick_params(axis='y', labelcolor='#27AE60')
    ax1.set_ylim(0, 1.02)
    if loss_series is not None and np.isfinite(np.asarray(loss_series)).any():
        ax2 = ax1.twinx()
        ax2.plot(rounds, loss_series, 's--', color='#E74C3C', alpha=0.8,
                 label='Mean training loss')
        ax2.set_ylabel('Mean training loss', color='#E74C3C')
        ax2.tick_params(axis='y', labelcolor='#E74C3C')
    ax1.axvline(summary['best_detection_round'], color='#27AE60', ls=':', alpha=0.5)
    if summary.get('best_loss_round') is not None:
        ax1.axvline(summary['best_loss_round'], color='#E74C3C', ls=':', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=8)
    gap = summary['A5_gap']
    gap_s = 'n/a' if gap != gap else f'{gap:+.4f}'   # NaN check
    plt.title(f"A5 round drift ({prim})  |  drift(std)={summary['A5_drift_tail_std']:.4f}  "
              f"gap={gap_s}")
    fig.tight_layout()
    fig.savefig(phase_dir / 'a5_round_drift.png', dpi=150)
    plt.close(fig)

    print(f'\n  Wrote:\n    {out_csv}\n    {phase_dir / "a5_summary.json"}'
          f'\n    {phase_dir / "a5_round_drift.png"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', choices=list(MODULE_MAP), default='base')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--phase-dir', required=True)
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--tail-frac', type=float, default=0.5,
                    help='fraction of final rounds treated as the converged tail')
    ap.add_argument('--metric', choices=list(HIGHER_BETTER), default='balanced',
                    help='primary detection scalar for A5 (default: balanced)')
    ap.add_argument('--ds-dir', default=None)
    args = ap.parse_args()

    mod = importlib.import_module(MODULE_MAP[args.module])
    phase_dir = Path(args.phase_dir).resolve()
    ds_dir = Path(args.ds_dir).resolve() if args.ds_dir else phase_dir.parent

    if args.dataset not in mod.DATASETS:
        raise SystemExit(f'--dataset must be one of {list(mod.DATASETS)}')

    print(f'A5 profiling | module={args.module} dataset={args.dataset}')
    print(f'  phase_dir = {phase_dir}')
    print(f'  ds_dir    = {ds_dir}')

    if args.module == 'B':
        envs = build_envs_strategyB(mod, args.dataset, ds_dir)
    else:
        envs = build_envs_base(mod, args.dataset, ds_dir)

    rows = profile(mod, envs, phase_dir, args.episodes)
    rounds = [r['round'] for r in rows]
    loss_series = load_loss_series(phase_dir, rounds)
    summary = compute_a5(rows, loss_series, args.tail_frac, args.metric)

    print('\n  === A5 summary ===')
    for k, v in summary.items():
        print(f'    {k}: {v}')

    save_outputs(phase_dir, rows, loss_series, summary)


if __name__ == '__main__':
    main()
