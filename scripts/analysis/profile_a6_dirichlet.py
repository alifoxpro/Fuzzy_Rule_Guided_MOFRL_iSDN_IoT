# -*- coding: utf-8 -*-
"""
============================================================================
A6 --- Client-Heterogeneity Dirichlet-alpha sweep
============================================================================
Re-partitions the pooled 5-link data across K=5 federated clients using a
label-skewed Dirichlet(alpha) split, trains FedAvg for each
alpha in {0.1, 0.5, 1.0}, and measures per-client heterogeneity
(HetStd / HetGap) of the shared global detector.

Reuses the training module verbatim: SDNEnvironment, DQNAgent, train_fl_dqn,
evaluate_agent. The per-link Mamba cache is reused (softmax/embeddings are
per-window, so pooling + re-sharding needs no Mamba recompute).

Run:
  python scripts/analysis/profile_a6_dirichlet.py --dataset ciciot2023
============================================================================
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN_DIR = Path(__file__).resolve().parent.parent / 'training'
sys.path.insert(0, str(TRAIN_DIR))
import train_rl_multiagent as M

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

K = 5                      # federated clients (train_fl_dqn is fixed at 5)
ALPHAS = [0.1, 0.5, 1.0]
SEEDS = [0, 1, 2]          # >=3 seeds per alpha (reporting rule IV-D)
POOL_CAP = 300_000         # subsample pooled windows for tractable sweep

# Lighter-but-identical training budget across all alphas (heterogeneity only
# needs a sufficiently-trained detector, not full A5-grade convergence).
SWEEP_ROUNDS = 15
SWEEP_LOCAL_EPISODES = 3
SWEEP_MAX_STEPS = 1500


def load_pooled(dataset_key, ds_dir):
    """Load + scale + cache-load all 5 links, then pool into single arrays."""
    ds_cfg = M.DATASETS[dataset_key]
    with open(ds_cfg['json_file']) as f:
        feat = json.load(f)
    feature_order = feat['selected_features'] if isinstance(feat, dict) else feat
    cache_dir = ds_dir / 'mamba_cache'

    Xs, ys, sms, embs = [], [], [], []
    for link_file in sorted(ds_cfg['data_dir'].glob('Link_*.csv')):
        ln = link_file.stem
        X_raw, y_raw = M.load_link_csv(link_file, feature_order)
        X_win, y_win = M.create_windows(X_raw, y_raw, M.WINDOW_SIZE, M.STRIDE)
        nf = X_win.shape[2]
        X_win = StandardScaler().fit_transform(
            X_win.reshape(-1, nf)).reshape(X_win.shape).astype(np.float32)

        cs, ce = cache_dir / f'{ln}_softmax.npy', cache_dir / f'{ln}_embed.npy'
        if not (cs.exists() and ce.exists()):
            raise FileNotFoundError(f'missing Mamba cache for {ln} in {cache_dir}')
        softmax = np.load(cs)
        emb = np.load(ce)
        sub = emb[:min(50000, len(emb))]
        pca = PCA(n_components=16, random_state=M.SEED).fit(sub)
        emb_pca = pca.transform(emb).astype(np.float32)

        Xs.append(X_win); ys.append(y_win); sms.append(softmax); embs.append(emb_pca)
        print(f'  pooled {ln}: {len(y_win)} windows', flush=True)

    X = np.concatenate(Xs); y = np.concatenate(ys)
    sm = np.concatenate(sms); emb = np.concatenate(embs)
    del Xs, ys, sms, embs

    # stratified subsample for a tractable sweep
    rng = np.random.default_rng(M.SEED)
    if len(y) > POOL_CAP:
        idx = rng.choice(len(y), POOL_CAP, replace=False)
        idx.sort()
        X, y, sm, emb = X[idx], y[idx], sm[idx], emb[idx]
    print(f'  pooled total: {len(y)} windows (cap {POOL_CAP})', flush=True)
    return X, y, sm, emb, feature_order


def dirichlet_partition(y, k, alpha, rng):
    """Label-skewed Dirichlet(alpha) split of indices across k clients."""
    client_idx = [[] for _ in range(k)]
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        props = rng.dirichlet([alpha] * k)
        cuts = (np.cumsum(props) * len(c_idx)).astype(int)[:-1]
        for j, part in enumerate(np.split(c_idx, cuts)):
            client_idx[j].extend(part.tolist())
    return [np.array(sorted(ci), dtype=np.int64) for ci in client_idx]


def het(vals):
    return float(np.std(vals)), float(max(vals) - min(vals))


def run_alpha_seed(alpha, seed, X, y, sm, emb, feature_order, out_dir):
    # seed drives BOTH the client partition and the training stochasticity
    rng = np.random.default_rng(10_000 * seed + int(alpha * 100))
    shards = dirichlet_partition(y, K, alpha, rng)
    sizes = [len(s) for s in shards]
    print(f'\n--- alpha={alpha} seed={seed} | client sizes={sizes} ---', flush=True)

    envs = []
    for kk, sh in enumerate(shards):
        envs.append(M.SDNEnvironment(X[sh], y[sh], sm[sh], emb[sh],
                                     link_idx=kk, feature_order=feature_order,
                                     max_steps=SWEEP_MAX_STEPS))
    agents = [M.DQNAgent() for _ in range(K)]
    M.seed_everything(seed)
    phase_dir = out_dir / f'alpha_{alpha}_seed_{seed}'
    res = M.train_fl_dqn(agents, envs, SWEEP_ROUNDS, SWEEP_LOCAL_EPISODES,
                         algo='fedavg', phase_dir=phase_dir)

    per_msr = [res[ln]['msr'] for ln in M.LINK_NAMES]
    per_fpr = [res[ln]['fpr'] for ln in M.LINK_NAMES]
    msr_std, msr_gap = het(per_msr)
    fpr_std, fpr_gap = het(per_fpr)
    row = {
        'alpha': alpha, 'seed': seed, 'client_sizes': sizes,
        'per_client_msr': per_msr, 'per_client_fpr': per_fpr,
        'MSR_HetStd': msr_std, 'MSR_HetGap': msr_gap,
        'FPR_HetStd': fpr_std, 'FPR_HetGap': fpr_gap,
    }
    print(f'  alpha={alpha} seed={seed}: FPR HetStd={fpr_std:.4f} HetGap={fpr_gap:.4f}'
          f' | MSR HetStd={msr_std:.4f} HetGap={msr_gap:.4f}', flush=True)
    del envs, agents
    import gc; gc.collect()
    return row


def aggregate(seed_rows):
    """Mean +/- std over seeds for each heterogeneity metric."""
    keys = ('FPR_HetStd', 'FPR_HetGap', 'MSR_HetStd', 'MSR_HetGap')
    agg = {'alpha': seed_rows[0]['alpha'], 'n_seeds': len(seed_rows)}
    for k in keys:
        vals = [r[k] for r in seed_rows]
        agg[k + '_mean'] = float(np.mean(vals))
        agg[k + '_std'] = float(np.std(vals))
    agg['per_seed'] = seed_rows
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ciciot2023')
    ap.add_argument('--ds-dir', default=None)
    args = ap.parse_args()

    ds_dir = (Path(args.ds_dir).resolve() if args.ds_dir
              else Path('experiments/results_A5run').resolve() / args.dataset)
    out_dir = Path('experiments/results_A5run').resolve() / args.dataset / 'a6_dirichlet'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'A6 Dirichlet sweep | dataset={args.dataset} ds_dir={ds_dir}', flush=True)
    print(f'  alphas={ALPHAS} seeds={SEEDS} '
          f'(rounds={SWEEP_ROUNDS}, local_eps={SWEEP_LOCAL_EPISODES}, '
          f'max_steps={SWEEP_MAX_STEPS})', flush=True)
    X, y, sm, emb, fo = load_pooled(args.dataset, ds_dir)

    agg_rows = []
    for a in ALPHAS:
        seed_rows = [run_alpha_seed(a, s, X, y, sm, emb, fo, out_dir) for s in SEEDS]
        agg = aggregate(seed_rows)
        agg_rows.append(agg)
        print(f'== alpha={a}: FPR HetStd={agg["FPR_HetStd_mean"]:.4f}'
              f'+/-{agg["FPR_HetStd_std"]:.4f} | '
              f'MSR HetStd={agg["MSR_HetStd_mean"]:.4f}'
              f'+/-{agg["MSR_HetStd_std"]:.4f} (n={agg["n_seeds"]})', flush=True)

    with open(out_dir / 'a6_summary.json', 'w') as f:
        json.dump(agg_rows, f, indent=2)

    # figure: mean heterogeneity vs alpha with +/- std error bars over seeds
    a = [r['alpha'] for r in agg_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(a, [r['FPR_HetStd_mean'] for r in agg_rows],
                yerr=[r['FPR_HetStd_std'] for r in agg_rows],
                fmt='o-', color='#D1495B', capsize=4, label='FPR HetStd')
    ax.errorbar(a, [r['MSR_HetStd_mean'] for r in agg_rows],
                yerr=[r['MSR_HetStd_std'] for r in agg_rows],
                fmt='s-', color='#2C6FBB', capsize=4, label='MSR HetStd')
    ax.set_xscale('log')
    ax.set_xlabel(r'Dirichlet concentration $\alpha$  (smaller = more non-IID)')
    ax.set_ylabel('Client heterogeneity (std across clients)')
    ax.set_title(f'A6 client heterogeneity vs non-IID level '
                 f'({args.dataset}, mean$\\pm$std over {len(SEEDS)} seeds)',
                 fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'a6_dirichlet.png', dpi=150)
    fig.savefig(out_dir / 'a6_dirichlet.pdf')
    plt.close(fig)

    print('\nWrote:', out_dir / 'a6_summary.json', '+ figure', flush=True)
    print('A6_DIRICHLET_DONE', flush=True)


if __name__ == '__main__':
    main()
