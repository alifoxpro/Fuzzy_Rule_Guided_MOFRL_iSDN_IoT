# -*- coding: utf-8 -*-
"""
============================================================================
A3 --- Severity-Slope profiling (stealth proxy)
============================================================================
Grades attack windows into L=4 stealth levels by linear interpolation toward
the per-link benign centroid in feature space:

    x(lambda) = (1-lambda) * x_attack + lambda * mu_benign,
    lambda in {0.0, 0.25, 0.5, 0.75}   (0 = original attack, 0.75 = stealthiest)

For each level the FINAL global model's sensitivity TPR(lambda) = 1 - FNR is
measured per link and averaged; A3 is the OLS slope of TPR against lambda.
Mamba outputs are recomputed for perturbed windows (cache is invalid for
them); the per-link PCA is re-fitted on the original cached embeddings, so
perturbed embeddings live in the same 16-D space used at training time.

Resume-safe: per-(link,lambda,phase) results are appended to progress.jsonl
and skipped on re-run.

Run:  python scripts/analysis/profile_a3_severity.py --dataset ciciot2023
============================================================================
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN_DIR = Path(__file__).resolve().parent.parent / 'training'
sys.path.insert(0, str(TRAIN_DIR))
import train_rl_multiagent as M

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

LAMBDAS = [0.0, 0.25, 0.5, 0.75]
PHASES = ['fedavg', 'fedprox']
N_PER_LINK = 5000
EVAL_EPISODES = 5


def load_link(link_file, feature_order):
    """Load one link: scaled windows + labels (mirrors build_envs_base)."""
    X_raw, y_raw = M.load_link_csv(link_file, feature_order)
    X_win, y_win = M.create_windows(X_raw, y_raw, M.WINDOW_SIZE, M.STRIDE)
    nf = X_win.shape[2]
    X_win = StandardScaler().fit_transform(
        X_win.reshape(-1, nf)).reshape(X_win.shape).astype(np.float32)
    return X_win, y_win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ciciot2023')
    ap.add_argument('--n-per-link', type=int, default=N_PER_LINK)
    args = ap.parse_args()
    ds = args.dataset

    base = Path('experiments/results_A5run').resolve()
    ds_dir = base / ds
    out_dir = ds_dir / 'a3_severity'
    out_dir.mkdir(parents=True, exist_ok=True)
    prog_path = out_dir / 'progress.jsonl'

    # resume cache
    done = {}
    if prog_path.exists():
        for line in prog_path.read_text().splitlines():
            r = json.loads(line)
            done[(r['link'], r['lam'], r['phase'])] = r
    print(f'A3 severity sweep | dataset={ds} | resume entries: {len(done)}',
          flush=True)

    ds_cfg = M.DATASETS[ds]
    with open(ds_cfg['json_file']) as f:
        feat = json.load(f)
    feature_order = feat['selected_features'] if isinstance(feat, dict) else feat
    cache_dir = ds_dir / 'mamba_cache'
    mamba_ckpt_dir = Path('experiments/results') / ds / 'checkpoints'

    # final global models per phase
    agents = {}
    for ph in PHASES:
        a = M.DQNAgent()
        gp = torch.load(ds_dir / ph / 'global_dqn.pt',
                        weights_only=False, map_location=M.DEVICE)
        a.set_params(gp)
        agents[ph] = a

    rng = np.random.default_rng(M.SEED)
    link_files = sorted(ds_cfg['data_dir'].glob('Link_*.csv'))

    # pooled benign centroid fallback for links without benign windows
    pooled_ben_sum, pooled_ben_n = None, 0

    per_link_data = []
    for lf in link_files:
        ln = lf.stem
        if all((ln, lam, ph) in done for lam in LAMBDAS for ph in PHASES):
            print(f'  {ln}: all cells cached, skipping data load', flush=True)
            per_link_data.append((ln, None, None, None))
            continue
        X, y = load_link(lf, feature_order)
        ben = X[y == 2]
        if len(ben) > 0:
            mu = ben.mean(axis=0)
            pooled_ben_sum = mu * len(ben) if pooled_ben_sum is None \
                else pooled_ben_sum + ben.sum(axis=0)
            pooled_ben_n += len(ben)
        else:
            mu = None                     # fill from pooled centroid later
        atk_idx = np.where(y == 0)[0]
        take = rng.choice(atk_idx, min(args.n_per_link, len(atk_idx)),
                          replace=False)
        per_link_data.append((ln, X[take].copy(), mu, len(atk_idx)))
        del X, y
        import gc; gc.collect()

    pooled_mu = (pooled_ben_sum / pooled_ben_n).astype(np.float32) \
        if pooled_ben_n else None

    for ln, X_atk, mu, n_atk in per_link_data:
        if X_atk is None:
            continue
        if mu is None:
            mu = pooled_mu
            print(f'  {ln}: no benign windows; using pooled centroid', flush=True)

        # per-link PCA fitted on the original cached embeddings (as in training)
        emb_cache = np.load(cache_dir / f'{ln}_embed.npy')
        pca = PCA(n_components=16, random_state=M.SEED)
        pca.fit(emb_cache[:min(50000, len(emb_cache))])
        extractor = M.MambaEmbeddingExtractor(
            mamba_ckpt_dir / f'{ln}_Mamba_Only.pt', n_features=X_atk.shape[2])

        for lam in LAMBDAS:
            if all((ln, lam, ph) in done for ph in PHASES):
                continue
            X_lam = ((1.0 - lam) * X_atk + lam * mu).astype(np.float32)
            softmax, emb = extractor.extract_all(X_lam)
            emb16 = pca.transform(emb).astype(np.float32)
            y_lam = np.zeros(len(X_lam), dtype=np.int64)      # all High
            env = M.SDNEnvironment(X_lam, y_lam, softmax, emb16,
                                   link_idx=0, feature_order=feature_order)
            env.neighbor_envs = []
            for ph in PHASES:
                if (ln, lam, ph) in done:
                    continue
                M.seed_everything(M.SEED)
                met = M.evaluate_agent(agents[ph], env,
                                       n_episodes=EVAL_EPISODES)
                row = {'link': ln, 'lam': lam, 'phase': ph,
                       'tpr': 1.0 - met['fnr'], 'fnr': met['fnr'],
                       'n_windows': int(len(X_lam))}
                done[(ln, lam, ph)] = row
                with open(prog_path, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f'  {ln} lam={lam} {ph}: TPR={row["tpr"]:.4f}',
                      flush=True)
            del X_lam, softmax, emb, emb16, env
        del extractor
        import gc; gc.collect()

    # ---------------- aggregate: mean TPR per (phase, lambda) + OLS slope ----
    links = sorted({k[0] for k in done})
    summary = {'dataset': ds, 'lambdas': LAMBDAS, 'n_per_link': args.n_per_link}
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {'fedavg': '#2C6FBB', 'fedprox': '#D1495B'}
    for ph in PHASES:
        tpr_mean = [float(np.mean([done[(ln, lam, ph)]['tpr'] for ln in links]))
                    for lam in LAMBDAS]
        beta, alpha0 = np.polyfit(LAMBDAS, tpr_mean, 1)
        summary[ph] = {
            'tpr_per_lambda': dict(zip(map(str, LAMBDAS), tpr_mean)),
            'A3_slope': float(beta), 'intercept': float(alpha0),
            'per_link': {ln: [done[(ln, lam, ph)]['tpr'] for lam in LAMBDAS]
                         for ln in links},
        }
        ax.plot(LAMBDAS, tpr_mean, 'o-', color=colors[ph],
                label=f'{ph}  (A3 slope $\\beta$={beta:+.3f})')
        print(f'== {ph}: TPR(lam)={[round(v,3) for v in tpr_mean]} '
              f'-> A3 slope={beta:+.4f}', flush=True)

    ax.set_xlabel(r'stealth level $\lambda$  (0 = original attack)')
    ax.set_ylabel('Sensitivity TPR  (mean over links)')
    ax.set_title(f'A3 severity slope ({ds}, final global model)')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'a3_severity.png', dpi=150)
    fig.savefig(out_dir / 'a3_severity.pdf')
    plt.close(fig)

    with open(out_dir / 'a3_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {out_dir / "a3_summary.json"} + figure', flush=True)
    print('A3_DONE', flush=True)


if __name__ == '__main__':
    main()
