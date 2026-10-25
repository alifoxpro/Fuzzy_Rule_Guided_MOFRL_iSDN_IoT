# -*- coding: utf-8 -*-
"""
Stage 3: train the federated DQN phases B1 (Independent) -> B2 (FedAvg) ->
B3 (FedProx) -> B4 (Fuzzy, FedProx base) for one seed, reusing the thesis code
(train_rl_multiagent_StrategyB.py) unchanged, but with environments whose
reward is computed against GROUND-TRUTH window labels.

Also mines the AFRAS-FL rule bases from ground-truth labels on the SAME scaled
features the environment evaluates, so the fuzzy term can fire (the canonical
thesis run fitted rules on scaled features but evaluated them on raw ones).
The fuzzy-rule firing rate is logged.
"""
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch

TRAIN_DIR = Path(__file__).resolve().parents[1] / "training"
PROJECT = Path(__file__).resolve().parents[2]  # repository parent (holds experiments/, data/)
sys.path.insert(0, str(TRAIN_DIR)); sys.path.insert(0, str(TRAIN_DIR.parent / "data"))
import train_rl_multiagent_StrategyB as B          # noqa: E402

FEATURE_ORDER = None


def log(m):
    print(m, flush=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    B.SEED = s


def mine_rules(k, X, y, afras_dir):
    """AFRAS-FL on window means of the two highest-variance (scaled) features."""
    Xm = X.mean(axis=1)
    top2 = np.argsort(-Xm.var(axis=0))[:2]
    names = [FEATURE_ORDER[i] for i in top2]
    pipe = B.AFRASFLPipeline(Xm[:, top2].astype(np.float32), y.astype(np.int64), names, B.LINK_NAMES[k], afras_dir)
    pkg = pipe.run_all()
    return B.FuzzyInferenceEngine(rules=pkg.get("rules", []),
                                  membership_params=pkg.get("membership_params", {}),
                                  feature_names=pkg.get("feature_names", names))


def fire_rate(engine, env):
    """Share of windows on which at least one rule has non-negligible firing strength."""
    if env.fuzzy_features is None or not engine.rules:
        return 0.0
    hits = 0
    n = min(len(env.fuzzy_features), 2000)
    for fv in env.fuzzy_features[:n]:
        strength = 0.0
        for r in engine.rules:
            s = 1.0
            for fname, term in r["antecedents"]:
                mp = engine.membership_params.get(fname, {}).get(term)
                j = engine.feature_names.index(fname)
                s *= engine._gaussian_membership(fv[j], mp["center"], mp["width"]) if mp else 0.0
            strength = max(strength, s)
        hits += strength > 0.1
    return hits / max(n, 1)


def main():
    global FEATURE_ORDER
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT / "experiments" / "reeval_diad"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    manifest = json.load(open(root / "data" / "manifest.json"))
    FEATURE_ORDER = manifest["feature_order"]
    out = root / f"seed_{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    B.USE_RLS = True            # data already scaled on TRAIN; do not re-scale inside AFRAS-FL
    B.USE_PBRS = False
    if a.smoke:
        B.MAX_STEPS_PER_EPISODE = 200
    ind_eps = 4 if a.smoke else B.INDEPENDENT_EPISODES
    rounds = 2 if a.smoke else B.FL_ROUNDS
    local_eps = 1 if a.smoke else B.LOCAL_EPISODES_PER_ROUND

    envs, engines, fires = [], [], {}
    for k in range(5):
        d = np.load(root / "data" / f"link_{k}.npz", allow_pickle=True)
        fz = np.load(root / "detectors" / f"link_{k}_features.npz")
        eng = mine_rules(k, d["X_train"], d["y_train"], out / "afras_fl" / B.LINK_NAMES[k])
        env = B.SDNEnvironmentB(
            X_windows=d["X_train"], y_windows=d["y_train"],          # <- ground-truth labels
            mamba_softmax=fz["soft_train"], mamba_embed_pca=fz["pca_train"],
            link_idx=k, feature_order=FEATURE_ORDER, fuzzy_engine=eng,
            use_mosar=True, use_fuzzy_reward=False, max_steps=B.MAX_STEPS_PER_EPISODE)
        fires[B.LINK_NAMES[k]] = fire_rate(eng, env)
        log(f"  {B.LINK_NAMES[k]}: {len(eng.rules)} rules, firing on {fires[B.LINK_NAMES[k]]:.1%} of windows")
        envs.append(env); engines.append(eng)
    json.dump({"rule_firing_rate": fires, "n_rules": {B.LINK_NAMES[k]: len(engines[k].rules) for k in range(5)}},
              open(out / "rules_summary.json", "w"), indent=1)

    agents = [B.DQNAgent() for _ in range(5)]
    t0 = time.time()
    B.train_independent_mosar(agents, envs, ind_eps, out / "mosar_independent")
    log(f"  B1 done ({time.time()-t0:.0f}s)")
    B.train_fl_mosar(agents, envs, rounds, local_eps, algo="fedavg", phase_dir=out / "mosar_fedavg")
    log(f"  B2 done ({time.time()-t0:.0f}s)")
    B.train_fl_mosar(agents, envs, rounds, local_eps, algo="fedprox", mu=B.FEDPROX_MU, phase_dir=out / "mosar_fedprox")
    log(f"  B3 done ({time.time()-t0:.0f}s)")
    B.train_fuzzy_augmented(agents, envs, rounds, local_eps, phase_dir=out / "fuzzy_retrain")
    log(f"  B4 done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
