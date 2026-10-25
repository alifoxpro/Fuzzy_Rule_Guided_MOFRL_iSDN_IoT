# -*- coding: utf-8 -*-
"""Aggregate evaluation_test.json across seeds: mean, SD and paired differences."""
import json, sys
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]  # repository parent (holds experiments/, data/)
root = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / "experiments" / "reeval_diad"
runs = sorted(root.glob("seed_*/evaluation_test.json"))
if not runs:
    sys.exit("no evaluation_test.json found")
keys = ["macro_recall", "macro_f1", "balanced_acc_binary", "fpr_binary", "fnr_binary"]
table = {}
for r in runs:
    d = json.load(open(r))
    for ph, s in d["phases"].items():
        for k in keys:
            table.setdefault(ph, {}).setdefault(k, []).append(s["pooled"]["dqn"][k])
    mb = d["phases"]["B1_Independent"]["pooled"]["mamba_baseline"]
    for k in keys:
        table.setdefault("Detector_only", {}).setdefault(k, []).append(mb[k])
out = {"n_seeds": len(runs), "seeds": [r.parent.name for r in runs], "mean_sd": {}}
print(f"seeds: {out['seeds']}")
print(f"{'phase':16s} " + " ".join(f"{k:>22s}" for k in keys))
for ph, m in table.items():
    out["mean_sd"][ph] = {k: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0] for k, v in m.items()}
    print(f"{ph:16s} " + " ".join(f"{np.mean(m[k]):>12.4f} ± {np.std(m[k], ddof=1) if len(m[k])>1 else 0:.4f}" for k in keys))
json.dump(out, open(root / "summary_all_seeds.json", "w"), indent=1)
print("wrote", root / "summary_all_seeds.json")
