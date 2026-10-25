# -*- coding: utf-8 -*-
"""
============================================================================
Test-Set Evaluator for Strategy B (leakage-free)
============================================================================
Loads a previously-trained results_StrategyB_*/{dataset}/ folder, rebuilds
the Mamba + PCA + fuzzy pipeline on the RECORD-LEVEL TEST split (produced
by scripts/data/record_level_split.py), reloads the final global DQN
weights for each federated phase (B2, B3, B4), and scores every eval
window once against its GROUND-TRUTH severity label (not the Mamba
prediction that drives the env): 3x3 confusion matrix, per-class
recall/precision/F1, macro-recall, macro-F1, binary FPR/FNR (attack =
High|Medium, flagged = BLOCK|RATE_LIMIT; None when undefined), bootstrap
95% CIs, and the same metrics for the "act on Mamba's prediction" baseline.
The legacy random-episode evaluation (labels = Mamba prediction) is kept
behind --n-episodes for reproducing previously reported numbers.

This is the metric a Q1 reviewer will demand: numbers on a split whose
records were never seen during scaler fitting, PCA fitting, AFRAS-FL
rule mining, or DQN training.

Usage:
    python scripts/training/evaluate_on_test.py \\
        --results-dir experiments/results_StrategyB_rls \\
        --dataset ciciot2023 [--n-episodes 50] [--max-windows 2000]

Output: <results-dir>/<dataset>/evaluation_on_<split>_groundtruth.json
============================================================================
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

# Reuse the training module's classes and constants
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / 'data'))

import train_rl_multiagent_StrategyB as B  # noqa: E402
from load_link_rls import load_link_rls  # noqa: E402


PROJECT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_ROOT = PROJECT / 'experiments'
DEFAULT_RLS_ROOT = PROJECT / 'data' / 'unified_14features_5links_rls'

# Which split to evaluate on. Overridden by --eval-split. Default 'test'.
EVAL_SPLIT = 'test'


def log(msg=''):
    print(msg, flush=True)


# ============================================================================
# Env building (one per link) — must be built as a set so neighbor_envs
# can be cross-linked, matching the training-time evaluation contract.
# ============================================================================
def _find_mamba_ckpt(dataset_key: str, link_name: str) -> Path:
    ds_data_dir = B.DATASETS[dataset_key]['data_dir']
    ckpt_candidates = [
        ds_data_dir.parent.parent / 'stage3_results' / dataset_key / link_name / f'{link_name}_Mamba_Only.pt',
        ds_data_dir / f'{link_name}_Mamba_Only.pt',
        PROJECT / 'experiments' / 'results' / dataset_key / 'checkpoints' / f'{link_name}_Mamba_Only.pt',
        PROJECT / 'experiments' / 'results' / 'checkpoints' / 'stage3' / f'{link_name}_Mamba_Only.pt',
    ]
    ckpt = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt is None:
        raise FileNotFoundError(
            f'Mamba checkpoint missing for {link_name}. Tried: '
            + ', '.join(str(p) for p in ckpt_candidates))
    return ckpt


def build_link_env(dataset_key: str, link_name: str, i: int,
                   feature_order, rls_root: Path,
                   fuzzy_engine, use_fuzzy_reward: bool):
    """Rebuild the SDNEnvironmentB for one link on the requested split."""
    log(f'    [{link_name}] loading {EVAL_SPLIT} split...')
    X_eval, y_eval = load_link_rls(
        dataset=dataset_key, link_name=link_name,
        feature_order=feature_order, split=EVAL_SPLIT, rls_root=rls_root)
    X_train, _ = load_link_rls(
        dataset=dataset_key, link_name=link_name,
        feature_order=feature_order, split='train', rls_root=rls_root)

    log(f'      creating windows W={B.WINDOW_SIZE} S={B.STRIDE}...')
    # Keep the ground-truth window labels (label of the window's last
    # record, as in create_windows). The env itself is still driven by the
    # Mamba prediction so the agents see the state/reward they were trained
    # on, but every metric below is scored against these true labels.
    X_win_eval, y_win_eval = B.create_windows(X_eval, y_eval, B.WINDOW_SIZE, B.STRIDE)
    X_win_train, _ = B.create_windows(
        X_train, np.zeros(len(X_train), dtype=np.int64), B.WINDOW_SIZE, B.STRIDE)
    del X_eval, y_eval, X_train
    gc.collect()

    ckpt = _find_mamba_ckpt(dataset_key, link_name)
    extractor = B.MambaEmbeddingExtractor(ckpt, n_features=len(feature_order))
    log(f'      Mamba embed (train {len(X_win_train):,}, {EVAL_SPLIT} {len(X_win_eval):,})...')
    _, embed_train = extractor.extract_all(X_win_train)
    softmax_eval, embed_eval = extractor.extract_all(X_win_eval)
    del extractor, X_win_train
    gc.collect()

    # PCA fit on TRAIN embeddings, transform EVAL embeddings.
    pca = PCA(n_components=16, random_state=B.SEED)
    pca.fit(embed_train)
    embed_eval_pca = pca.transform(embed_eval).astype(np.float32)
    log(f'      PCA fit(train) transform({EVAL_SPLIT})  '
        f'ev={pca.explained_variance_ratio_.sum():.3f}')
    del embed_train, embed_eval
    gc.collect()

    y_mamba = np.asarray(softmax_eval).argmax(axis=1).astype(np.int64)
    env = B.SDNEnvironmentB(
        X_windows=X_win_eval,
        y_windows=y_mamba,
        mamba_softmax=softmax_eval,
        mamba_embed_pca=embed_eval_pca,
        link_idx=i, feature_order=feature_order,
        fuzzy_engine=fuzzy_engine,
        use_mosar=True, use_fuzzy_reward=use_fuzzy_reward,
        max_steps=B.MAX_STEPS_PER_EPISODE,
    )
    env._n_eval_windows = int(len(X_win_eval))
    env._y_true = np.asarray(y_win_eval, dtype=np.int64)
    env._y_mamba = y_mamba
    return env


def _apply_global_weights(agent, global_weights):
    """Copy a saved [p.clone() ...] list into agent.q_net.parameters()."""
    for p, w in zip(agent.q_net.parameters(), global_weights):
        p.data.copy_(w.to(p.device))
    agent.target_net.load_state_dict(agent.q_net.state_dict())


def run_step_locked_episodes(envs, agents, n_episodes: int):
    """Run n_episodes across all 5 (env, agent) pairs in LOCK-STEP.

    Each step ticks every env once so neighbor_envs' recent_severities
    stay in sync with what the DQN was trained under. This is the
    contract used by every FL round in the training script; running
    envs sequentially (one full episode at a time) leaves the other
    envs' recent_severities empty and degrades the state vector.
    """
    n = len(envs)
    optimal_action = {0: 2, 1: 1, 2: 0}
    per_link_msr = [[] for _ in range(n)]
    per_link_fpr = [[] for _ in range(n)]
    per_link_fnr = [[] for _ in range(n)]
    per_link_reward = [[] for _ in range(n)]

    for ep in range(n_episodes):
        states = [envs[i].reset() for i in range(n)]
        done_flags = [False] * n
        correct = [0] * n
        total = [0] * n
        fp = [0] * n; benign = [0] * n
        fn = [0] * n; attack = [0] * n
        rsum = [0.0] * n
        while not all(done_flags):
            for i in range(n):
                if done_flags[i]:
                    continue
                a = agents[i].select_action(states[i], eval_mode=True)
                q = agents[i].get_q_values(states[i])
                nxt, r, d, info = envs[i].step(a, q_values=q)
                t = info['true_severity']
                correct[i] += int(a == optimal_action[t])
                if t == 2:
                    benign[i] += 1
                    if a == 2:
                        fp[i] += 1
                elif t == 0:
                    attack[i] += 1
                    if a == 0:
                        fn[i] += 1
                total[i] += 1
                rsum[i] += r
                states[i] = nxt
                done_flags[i] = d
        for i in range(n):
            per_link_msr[i].append(correct[i] / max(total[i], 1))
            per_link_fpr[i].append(fp[i] / max(benign[i], 1))
            per_link_fnr[i].append(fn[i] / max(attack[i], 1))
            per_link_reward[i].append(rsum[i] / max(total[i], 1))

    out = []
    for i, env in enumerate(envs):
        out.append({
            'link_name': B.LINK_NAMES[i],
            'n_episodes': n_episodes,
            'msr_mean': float(np.mean(per_link_msr[i])),
            'msr_std': float(np.std(per_link_msr[i])),
            'fpr_mean': float(np.mean(per_link_fpr[i])),
            'fpr_std': float(np.std(per_link_fpr[i])),
            'fnr_mean': float(np.mean(per_link_fnr[i])),
            'fnr_std': float(np.std(per_link_fnr[i])),
            'reward_mean': float(np.mean(per_link_reward[i])),
            'eval_windows': int(getattr(env, '_n_eval_windows', 0)),
        })
    return out


# ============================================================================
# Ground-truth metrics
# ----------------------------------------------------------------------------
# Severity: 0=High, 1=Medium, 2=Low.  Action: 0=ALLOW, 1=RATE_LIMIT, 2=BLOCK.
# The severity-optimal action is High->BLOCK, Medium->RATE_LIMIT, Low->ALLOW,
# so an action can be read as a predicted severity via ACT_TO_SEV.
# Binary view: attack = {High, Medium}; flagged = {BLOCK, RATE_LIMIT}.
# Rates whose denominator is empty are reported as None, not 0.
# ============================================================================
SEV_NAMES = ['High', 'Medium', 'Low']
ACT_NAMES = ['ALLOW', 'RATE_LIMIT', 'BLOCK']
OPT_ACTION = np.array([2, 1, 0], dtype=np.int64)   # indexed by severity
ACT_TO_SEV = np.array([2, 1, 0], dtype=np.int64)   # indexed by action


def _safe_mean(mask):
    return float(mask.mean()) if mask.size else None


def gt_metrics(y_true, actions):
    """Metrics of `actions` scored against ground-truth severities."""
    y = np.asarray(y_true, dtype=np.int64)
    a = np.asarray(actions, dtype=np.int64)
    pred = ACT_TO_SEV[a]

    cm = np.zeros((3, 3), dtype=np.int64)          # rows = true sev, cols = action
    np.add.at(cm, (y, a), 1)

    recall, precision, f1 = {}, {}, {}
    for k, name in enumerate(SEV_NAMES):
        r = _safe_mean(pred[y == k] == k)
        p = _safe_mean(y[pred == k] == k)
        recall[name], precision[name] = r, p
        f1[name] = (2 * p * r / (p + r)) if (r is not None and p is not None and p + r > 0) else (
            0.0 if (r is not None and p is not None) else None)
    defined_r = [v for v in recall.values() if v is not None]
    defined_f1 = [v for v in f1.values() if v is not None]

    attack = y != 2
    flagged = a != 0
    fpr = _safe_mean(flagged[~attack])             # benign that was rate-limited or blocked
    fnr = _safe_mean(~flagged[attack])             # attack that was allowed
    ba_bin = (1 - (fpr + fnr) / 2) if (fpr is not None and fnr is not None) else None

    return {
        'n': int(y.size),
        'class_counts': {n: int((y == k).sum()) for k, n in enumerate(SEV_NAMES)},
        'confusion_true_x_action': cm.tolist(),
        'msr_gt': _safe_mean(pred == y),           # severity-optimal action rate
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'macro_recall': float(np.mean(defined_r)) if defined_r else None,
        'macro_f1': float(np.mean(defined_f1)) if defined_f1 else None,
        'fpr_binary': fpr,
        'fnr_binary': fnr,
        'balanced_acc_binary': ba_bin,
        # Legacy definitions used in the thesis (BLOCK on Low / ALLOW on High
        # only; RATE_LIMIT and Medium never counted), kept for comparison.
        'legacy_fpr': _safe_mean(a[y == 2] == 2),
        'legacy_fnr': _safe_mean(a[y == 0] == 0),
    }


def bootstrap_ci(y_true, actions, keys, n_boot=500, seed=0):
    """Percentile 95% CIs over windows for scalar metrics in `keys`."""
    y = np.asarray(y_true, dtype=np.int64)
    a = np.asarray(actions, dtype=np.int64)
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, y.size)
        m = gt_metrics(y[idx], a[idx])
        for k in keys:
            if m[k] is not None:
                samples[k].append(m[k])
    return {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                if v else None) for k, v in samples.items()}


def run_full_pass(envs, agents, seed=0, max_windows=None):
    """Score every eval window exactly once, in lock-step across links.

    Each link's windows are visited in one fixed seeded random order (matching
    the random sampling used during training, so the history-dependent state
    features behave as in training), in chunks of the env's max_steps.
    Returns per-link arrays of true labels, agent actions and Mamba predictions.
    """
    n = len(envs)
    rng = np.random.default_rng(seed)
    orders = []
    for e in envs:
        order = rng.permutation(e.n_windows)
        if max_windows is not None:
            order = order[:max_windows]
        orders.append(order)
    recs = [{'y': [], 'a': [], 'pm': []} for _ in range(n)]
    n_chunks = max(int(np.ceil(len(o) / e.max_steps)) for o, e in zip(orders, envs))

    for c in range(n_chunks):
        states, active = [None] * n, []
        for i, e in enumerate(envs):
            chunk = orders[i][c * e.max_steps:(c + 1) * e.max_steps]
            if len(chunk) == 0:
                continue
            e.reset()
            e.episode_indices = chunk
            states[i] = e._get_state()
            active.append(i)
        done = {i: False for i in active}
        while not all(done.values()):
            for i in active:
                if done[i]:
                    continue
                e = envs[i]
                idx = e.episode_indices[e.step_idx]
                act = agents[i].select_action(states[i], eval_mode=True)
                q = agents[i].get_q_values(states[i])
                nxt, _, d, _ = e.step(act, q_values=q)
                recs[i]['y'].append(int(e._y_true[idx]))
                recs[i]['a'].append(int(act))
                recs[i]['pm'].append(int(e._y_mamba[idx]))
                states[i], done[i] = nxt, d
    return [{k: np.asarray(v, dtype=np.int64) for k, v in r.items()} for r in recs]


def summarise_full_pass(recs):
    """Per-link and pooled ground-truth metrics for the DQN and for the
    'act on Mamba's prediction' baseline, plus DQN-vs-Mamba agreement."""
    per_link, ys, acts, mamba_acts = [], [], [], []
    for i, r in enumerate(recs):
        m_act = OPT_ACTION[r['pm']]
        per_link.append({
            'link_name': B.LINK_NAMES[i],
            'dqn': gt_metrics(r['y'], r['a']),
            'mamba_baseline': gt_metrics(r['y'], m_act),
            'agreement_with_mamba': _safe_mean(r['a'] == m_act),
        })
        ys.append(r['y']); acts.append(r['a']); mamba_acts.append(m_act)
    y, a, ma = np.concatenate(ys), np.concatenate(acts), np.concatenate(mamba_acts)
    ci_keys = ['msr_gt', 'macro_recall', 'macro_f1', 'fpr_binary',
               'fnr_binary', 'balanced_acc_binary']
    link_mr = [pl['dqn']['macro_recall'] for pl in per_link
               if pl['dqn']['macro_recall'] is not None]
    jfi = ((np.sum(link_mr) ** 2) / (len(link_mr) * np.sum(np.square(link_mr)) + 1e-12)
           if link_mr else None)
    return {
        'per_link': per_link,
        'pooled': {
            'dqn': gt_metrics(y, a),
            'dqn_ci95': bootstrap_ci(y, a, ci_keys),
            'mamba_baseline': gt_metrics(y, ma),
            'mamba_baseline_ci95': bootstrap_ci(y, ma, ci_keys),
            'agreement_with_mamba': _safe_mean(a == ma),
        },
        'jfi_over_link_macro_recall': float(jfi) if jfi is not None else None,
    }


# ============================================================================
# Phase driver
# ============================================================================
def load_global_dqn(phase_dir: Path):
    """Load the final global DQN weights saved by _fedavg_or_prox."""
    path = phase_dir / 'global_dqn.pt'
    if not path.exists():
        raise FileNotFoundError(f'global_dqn.pt not found in {phase_dir}')
    return torch.load(path, weights_only=True, map_location=B.DEVICE)


def build_fuzzy_engines(afras_base: Path):
    """Reload the AFRAS-FL fuzzy engines from Phase B0 artefacts."""
    engines = {}
    for link_name in B.LINK_NAMES:
        pkg_file = afras_base / link_name / 'phase7_deployed.json'
        if pkg_file.exists():
            with open(pkg_file) as f:
                pkg = json.load(f)
            engines[link_name] = B.FuzzyInferenceEngine(
                rules=pkg.get('rules', []),
                membership_params=pkg.get('membership_params', {}),
                feature_names=pkg.get('feature_names', []),
            )
        else:
            log(f'    WARNING: {pkg_file} missing; using empty fuzzy engine.')
            engines[link_name] = B.FuzzyInferenceEngine([], {}, [])
    return engines


def evaluate_dataset(results_dir: Path, dataset_key: str,
                     rls_root: Path, n_episodes: int,
                     phases_to_eval: list, max_windows=None):
    log('=' * 72)
    log(f'  Test-set evaluation  dataset={dataset_key}')
    log(f'  Results dir: {results_dir}')
    log(f'  RLS root:    {rls_root}')
    log(f'  Episodes:    {n_episodes}')
    log('=' * 72)

    ds_dir = results_dir / dataset_key
    if not ds_dir.exists():
        raise FileNotFoundError(f'Dataset dir missing: {ds_dir}')

    # Feature order from the same JSON the training script uses
    ds_cfg = B.DATASETS[dataset_key]
    with open(ds_cfg['json_file']) as f:
        fd = json.load(f)
    feature_order = (fd.get('selected_features', list(fd.keys()))
                     if isinstance(fd, dict) else fd)
    log(f'  Features:    {len(feature_order)}')

    afras_base = ds_dir / 'afras_fl'
    fuzzy_engines = build_fuzzy_engines(afras_base)

    all_out = {'dataset': dataset_key,
               'n_episodes': n_episodes,
               'phases': {}}

    for phase_name, phase_folder in phases_to_eval:
        phase_dir = ds_dir / phase_folder
        if not (phase_dir / 'global_dqn.pt').exists():
            log(f'\n  [skip] {phase_name}: no global_dqn.pt in {phase_dir}')
            continue

        log(f'\n  ==== {phase_name} ({phase_folder}) ====')
        global_weights = load_global_dqn(phase_dir)
        use_fuzzy_reward = (phase_name == 'B4_Fuzzy')

        # Build all 5 envs, cross-link neighbor_envs (matches training),
        # load global DQN weights into 5 agents, then run in lock-step.
        envs, agents = [], []
        for i, link_name in enumerate(B.LINK_NAMES):
            env = build_link_env(dataset_key, link_name, i,
                                 feature_order, rls_root,
                                 fuzzy_engines[link_name], use_fuzzy_reward)
            envs.append(env)
            ag = B.DQNAgent()
            _apply_global_weights(ag, global_weights)
            agents.append(ag)
        for e in envs:
            e.neighbor_envs = envs

        phase_out = {}

        # Primary: every eval window once, scored against ground truth.
        recs = run_full_pass(envs, agents, seed=B.SEED, max_windows=max_windows)
        gt = summarise_full_pass(recs)
        phase_out['ground_truth'] = gt
        for pl in gt['per_link']:
            d = pl['dqn']
            log(f'      {pl["link_name"]}: n={d["n"]:,} {d["class_counts"]}  '
                f'MSR_gt={d["msr_gt"]:.4f}  macroRecall={d["macro_recall"]:.4f}  '
                f'FPR={_fmt(d["fpr_binary"])}  FNR={_fmt(d["fnr_binary"])}  '
                f'agree(Mamba)={pl["agreement_with_mamba"]:.4f}')
        p, ci = gt['pooled']['dqn'], gt['pooled']['dqn_ci95']
        mb = gt['pooled']['mamba_baseline']
        log(f'    [POOLED DQN]   MSR_gt={p["msr_gt"]:.4f} {ci["msr_gt"]}  '
            f'macroRecall={p["macro_recall"]:.4f} {ci["macro_recall"]}  '
            f'BA_bin={_fmt(p["balanced_acc_binary"])} {ci["balanced_acc_binary"]}  '
            f'FPR={_fmt(p["fpr_binary"])}  FNR={_fmt(p["fnr_binary"])}')
        log(f'    [POOLED MAMBA] MSR_gt={mb["msr_gt"]:.4f}  '
            f'macroRecall={mb["macro_recall"]:.4f}  '
            f'BA_bin={_fmt(mb["balanced_acc_binary"])}  '
            f'FPR={_fmt(mb["fpr_binary"])}  FNR={_fmt(mb["fnr_binary"])}')
        log(f'    DQN agreement with Mamba: {gt["pooled"]["agreement_with_mamba"]:.4f}')

        # Optional: the original random-episode evaluation (labels = Mamba
        # prediction). Kept only to reproduce the previously reported numbers.
        if n_episodes > 0:
            per_link = run_step_locked_episodes(envs, agents, n_episodes)
            msrs = np.array([r['msr_mean'] for r in per_link])
            jfi = (msrs.sum() ** 2) / (5 * (msrs ** 2).sum() + 1e-12)
            phase_out['legacy_vs_mamba_labels'] = {
                'per_link': per_link,
                'msr_mean_of_links': float(msrs.mean()),
                'fpr_mean_of_links': float(np.mean([r['fpr_mean'] for r in per_link])),
                'fnr_mean_of_links': float(np.mean([r['fnr_mean'] for r in per_link])),
                'jfi': float(jfi),
            }
            log(f'    [LEGACY vs Mamba labels] MSR={msrs.mean():.4f}  JFI={jfi:.4f}')

        all_out['phases'][phase_name] = phase_out

    out_file = ds_dir / f'evaluation_on_{EVAL_SPLIT}_groundtruth.json'
    with open(out_file, 'w') as f:
        json.dump(all_out, f, indent=2)
    log(f'\n  Wrote: {out_file}')
    return all_out


def _fmt(x):
    return 'n/a' if x is None else f'{x:.4f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True,
                    help='e.g. experiments/results_StrategyB_rls')
    ap.add_argument('--dataset', choices=['ciciot2023', 'diad2024', 'all'],
                    default='all')
    ap.add_argument('--rls-root', default=str(DEFAULT_RLS_ROOT))
    ap.add_argument('--n-episodes', type=int, default=0,
                    help='Also run the legacy random-episode evaluation '
                         '(labels = Mamba prediction) with this many episodes. '
                         '0 = skip (default).')
    ap.add_argument('--max-windows', type=int, default=None,
                    help='Cap eval windows per link (quick smoke test).')
    ap.add_argument('--phases', default='B2,B3,B4',
                    help='Comma list of phases to evaluate: B2,B3,B4')
    ap.add_argument('--eval-split', choices=['train', 'val', 'test'],
                    default='test',
                    help='Which split to run the 50-episode evaluation on. '
                         'Diagnostic: use --eval-split train to verify the '
                         'evaluator matches the training-time numbers.')
    args = ap.parse_args()

    global EVAL_SPLIT
    EVAL_SPLIT = args.eval_split

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT / results_dir
    rls_root = Path(args.rls_root)

    phase_map = {
        'B2': ('B2_FedAvg', 'mosar_fedavg'),
        'B3': ('B3_FedProx', 'mosar_fedprox'),
        'B4': ('B4_Fuzzy', 'fuzzy_retrain'),
    }
    phases_to_eval = [phase_map[p.strip()] for p in args.phases.split(',')
                      if p.strip() in phase_map]

    targets = ['ciciot2023', 'diad2024'] if args.dataset == 'all' else [args.dataset]

    t0 = time.time()
    for ds in targets:
        evaluate_dataset(results_dir, ds, rls_root,
                         args.n_episodes, phases_to_eval,
                         max_windows=args.max_windows)
    log(f'\n  Total time: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
