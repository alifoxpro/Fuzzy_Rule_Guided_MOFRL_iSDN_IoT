# -*- coding: utf-8 -*-
"""
Stage 4: evaluate every phase of one seed on the TEST captures against
ground-truth labels, with the corrected evaluator (full pass over every test
window, 3x3 confusion, macro-recall, binary FPR/FNR, bootstrap CIs, and the
'act on the detector' baseline). Writes <root>/seed_<s>/evaluation_test.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

TRAIN_DIR = Path(__file__).resolve().parents[1] / "training"
PROJECT = Path(__file__).resolve().parents[2]  # repository parent (holds experiments/, data/)
sys.path.insert(0, str(TRAIN_DIR)); sys.path.insert(0, str(TRAIN_DIR.parent / "data"))
import train_rl_multiagent_StrategyB as B          # noqa: E402
import evaluate_on_test as E                       # noqa: E402

PHASES = [("B1_Independent", "mosar_independent"), ("B2_FedAvg", "mosar_fedavg"),
          ("B3_FedProx", "mosar_fedprox"), ("B4_Fuzzy", "fuzzy_retrain")]


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT / "experiments" / "reeval_diad"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    root = Path(a.root); sd = root / f"seed_{a.seed}"
    feat_order = json.load(open(root / "data" / "manifest.json"))["feature_order"]
    B.USE_RLS = True
    envs = []
    for k in range(5):
        d = np.load(root / "data" / f"link_{k}.npz", allow_pickle=True)
        fz = np.load(root / "detectors" / f"link_{k}_features.npz")
        env = B.SDNEnvironmentB(
            X_windows=d[f"X_{a.split}"], y_windows=d[f"y_{a.split}"],
            mamba_softmax=fz[f"soft_{a.split}"], mamba_embed_pca=fz[f"pca_{a.split}"],
            link_idx=k, feature_order=feat_order, fuzzy_engine=None,
            use_mosar=True, use_fuzzy_reward=False, max_steps=B.MAX_STEPS_PER_EPISODE)
        env._y_true = d[f"y_{a.split}"].astype(np.int64)
        env._y_mamba = fz[f"soft_{a.split}"].argmax(1).astype(np.int64)
        envs.append(env)
    for e in envs:
        e.neighbor_envs = envs

    results = {"seed": a.seed, "split": a.split, "phases": {}}
    for name, folder in PHASES:
        agents = []
        for k in range(5):
            ag = B.DQNAgent()
            ck = sd / folder / B.LINK_NAMES[k] / "best_dqn.pt"
            ag.q_net.load_state_dict(torch.load(ck, weights_only=True, map_location=B.DEVICE))
            ag.target_net.load_state_dict(ag.q_net.state_dict())
            agents.append(ag)
        recs = E.run_full_pass(envs, agents, seed=a.seed)
        summ = E.summarise_full_pass(recs)
        results["phases"][name] = summ
        p, ci = summ["pooled"]["dqn"], summ["pooled"]["dqn_ci95"]
        log(f"  {name:15s} macro-recall={p['macro_recall']:.4f} {ci['macro_recall']}  "
            f"BA_bin={p['balanced_acc_binary']} FPR={p['fpr_binary']} FNR={p['fnr_binary']}  "
            f"agree(detector)={summ['pooled']['agreement_with_mamba']:.3f}")
    mb = results["phases"][PHASES[0][0]]["pooled"]["mamba_baseline"]
    log(f"  {'Detector only':15s} macro-recall={mb['macro_recall']:.4f}  BA_bin={mb['balanced_acc_binary']} "
        f"FPR={mb['fpr_binary']} FNR={mb['fnr_binary']}")
    json.dump(results, open(sd / f"evaluation_{a.split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
