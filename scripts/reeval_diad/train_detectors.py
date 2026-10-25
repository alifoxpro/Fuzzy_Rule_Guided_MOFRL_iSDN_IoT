# -*- coding: utf-8 -*-
"""
Stage 2: train one Mamba detector per link on the capture-grouped TRAIN windows,
early-stop on VAL, and score it on TEST against ground-truth labels.
Also exports detector softmax + PCA-16 embeddings for the DQN stage.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from sklearn.decomposition import PCA

TRAIN_DIR = Path(__file__).resolve().parents[1] / "training"
PROJECT = Path(__file__).resolve().parents[2]  # repository parent (holds experiments/, data/)
sys.path.insert(0, str(TRAIN_DIR)); sys.path.insert(0, str(TRAIN_DIR.parent / "data"))
from train_isolated import MambaOnlyModel          # noqa: E402  (same architecture as the thesis)
import evaluate_on_test as E                       # noqa: E402  (ground-truth metrics)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(m):
    print(m, flush=True)


def fit(model, Xtr, ytr, Xva, yva, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    counts = np.bincount(ytr, minlength=3).astype(float)
    w = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32, device=DEV)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, bad = -1.0, None, 0
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            xb, yb = Xtr_t[idx].to(DEV), ytr_t[idx].to(DEV)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        pred = predict(model, Xva).argmax(1)
        score = E.gt_metrics(yva, E.OPT_ACTION[pred])["macro_recall"] if len(yva) else 0.0
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"      epoch {ep+1:2d}  val macro-recall={score:.4f}")
        if bad >= 4:
            break
    model.load_state_dict(best_state)
    return best


@torch.no_grad()
def predict(model, X):
    model.eval(); out = []
    for i in range(0, len(X), 1024):
        out.append(torch.softmax(model(torch.tensor(X[i:i + 1024]).to(DEV)), -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


@torch.no_grad()
def embed(model, X):
    model.eval(); sm, em = [], []
    for i in range(0, len(X), 1024):
        x = model.mamba(model.input_proj(torch.tensor(X[i:i + 1024]).to(DEV)))[:, -1, :]
        em.append(x.cpu().numpy()); sm.append(torch.softmax(model.fc(x), -1).cpu().numpy())
    return np.concatenate(sm), np.concatenate(em)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT / "experiments" / "reeval_diad"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    root = Path(a.root); data = root / "data"; out = root / "detectors"; out.mkdir(parents=True, exist_ok=True)
    report, pooled_y, pooled_a = {}, [], []
    for k in range(5):
        t0 = time.time()
        d = np.load(data / f"link_{k}.npz", allow_pickle=True)
        log(f"  link_{k}: train={len(d['y_train'])} val={len(d['y_val'])} test={len(d['y_test'])}")
        model = MambaOnlyModel(n_features=d["X_train"].shape[2], n_classes=3).to(DEV)
        best = fit(model, d["X_train"], d["y_train"], d["X_val"], d["y_val"], a.epochs, a.seed)
        torch.save(model.state_dict(), out / f"link_{k}_mamba.pt")
        feats = {}
        for sp in ("train", "val", "test"):
            feats[f"soft_{sp}"], feats[f"emb_{sp}"] = embed(model, d[f"X_{sp}"])
        pca = PCA(n_components=16, random_state=a.seed).fit(feats["emb_train"])
        for sp in ("train", "val", "test"):
            feats[f"pca_{sp}"] = pca.transform(feats[f"emb_{sp}"]).astype(np.float32)
            del feats[f"emb_{sp}"]
        np.savez_compressed(out / f"link_{k}_features.npz", **feats)
        pred = feats["soft_test"].argmax(1)
        m = E.gt_metrics(d["y_test"], E.OPT_ACTION[pred])
        report[f"link_{k}"] = {"best_val_macro_recall": best, "test": m, "seconds": round(time.time() - t0)}
        pooled_y.append(d["y_test"]); pooled_a.append(E.OPT_ACTION[pred])
        log(f"    TEST vs ground truth: macro-recall={m['macro_recall']:.4f}  "
            f"BA_bin={m['balanced_acc_binary']}  FPR={m['fpr_binary']}  FNR={m['fnr_binary']}")
    y, act = np.concatenate(pooled_y), np.concatenate(pooled_a)
    report["pooled_test"] = E.gt_metrics(y, act)
    report["pooled_test_ci95"] = E.bootstrap_ci(y, act, ["macro_recall", "balanced_acc_binary", "fpr_binary", "fnr_binary"])
    json.dump(report, open(out / "detector_report.json", "w"), indent=1)
    p = report["pooled_test"]
    log(f"  POOLED detector TEST: macro-recall={p['macro_recall']:.4f} BA_bin={p['balanced_acc_binary']} "
        f"FPR={p['fpr_binary']} FNR={p['fnr_binary']}  CI={report['pooled_test_ci95']}")


if __name__ == "__main__":
    main()
