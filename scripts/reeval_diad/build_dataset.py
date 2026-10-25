# -*- coding: utf-8 -*-
"""
Stage 1 of the DIAD2024 corrected re-evaluation: build time-ordered windows.

What differs from the original pipeline (see thesis Sec. "Evidence Audit"):
  * Windows are 20 CONSECUTIVE flows of ONE capture file, sorted by Timestamp,
    so the Mamba detector sees a genuine temporal sequence.
  * Labels are the capture's ground-truth severity (from its attack family).
  * Train / val / test are split BY CAPTURE FILE, so no capture contributes
    to more than one split.
  * A fixed number of flows is taken from each file to balance the classes.

Severity map (as in the thesis): DDoS, Mirai -> High(0); DoS -> Medium(1);
Recon, Benign -> Low(2). BruteForce / Spoofing / Web-Based are excluded.

Output: <out>/link_<k>.npz with X_{train,val,test} (n,20,14) float32 scaled,
y_{train,val,test}, file_{train,val,test}; plus manifest.json.
"""
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]  # repository parent (holds experiments/, data/)
RAW = Path(os.environ.get("DIAD_RAW", PROJECT / "data" / "raw" / "cic_iot_diad_2024"))
FEATURES = {  # unified name -> DIAD raw column (same map as scripts/data/unify_features.py)
    "flow_iat_mean": "Flow IAT Mean", "syn_flag_count": "SYN Flag Count",
    "avg_pkt_size": "Average Packet Size", "pkt_len_max": "Packet Length Max",
    "fwd_header_length": "Fwd Header Length", "flow_duration": "Flow Duration",
    "ack_flag_count": "ACK Flag Count", "flow_packets_per_s": "Flow Packets/s",
    "rst_flag_count": "RST Flag Count", "pkt_len_variance": "Packet Length Variance",
    "fwd_packets_per_s": "Fwd Packets/s", "protocol_type": "Protocol",
    "pkt_len_min": "Packet Length Min", "psh_flag_count": "PSH Flag Count",
}
FEATURE_ORDER = list(FEATURES.keys())
SEVERITY = {"DDOS": 0, "Mirai": 0, "DOS": 1, "Recon": 2, "Benign": 2}
ROWS_PER_FILE = {0: 4000, 1: 14000, 2: 90000}      # balances the three classes
W = 20                                               # window length (flows)
# Per-link class-mix targets (High, Medium, Low), as in the thesis partition.
LINK_MIX = [(0.50, 0.30, 0.20), (0.40, 0.20, 0.40), (0.10, 0.60, 0.30),
            (0.05, 0.15, 0.80), (0.20, 0.30, 0.50)]
# Low class has few captures: fix its split explicitly (documented).
LOW_SPLIT = {"BenignTraffic.pcap_Flow.csv": "train", "BenignTraffic1.pcap_Flow.csv": "train",
             "VulnerabilityScan.pcap_Flow.csv": "train",
             "BenignTraffic3.pcap_Flow.csv": "val", "BenignTraffic2.pcap_Flow.csv": "test"}


def log(m):
    print(m, flush=True)


_REF_HEADER = None


def reference_header():
    """Column names from a capture file that has a header row (all files share one layout)."""
    global _REF_HEADER
    if _REF_HEADER is None:
        ref = next((RAW / "Benign").rglob("*.csv"))
        with open(ref, encoding="utf-8", errors="ignore") as fh:
            _REF_HEADER = [c.strip() for c in fh.readline().strip().split(",")]
    return _REF_HEADER


def read_capture(path, nrows):
    cols = ["Timestamp"] + list(FEATURES.values())
    with open(path, encoding="utf-8", errors="ignore") as fh:
        first = [c.strip() for c in fh.readline().strip().split(",")]
    if "Timestamp" in first:
        df = pd.read_csv(path, usecols=lambda c: c.strip() in cols, nrows=nrows, low_memory=False)
    else:  # a few DoS-TCP_Flood files have no header row; same column layout as the others
        hdr = reference_header()
        assert len(first) == len(hdr), (path.name, len(first), len(hdr))
        df = pd.read_csv(path, header=None, names=hdr, usecols=cols, nrows=nrows, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    ts = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True, format="mixed")
    df = df.assign(_ts=ts).sort_values("_ts", kind="stable")
    X = df[list(FEATURES.values())].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    X[~np.isfinite(X)] = 0.0
    return X


def windows(X):
    n = len(X) // W
    return X[: n * W].reshape(n, W, X.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT / "experiments" / "reeval_diad" / "data"))
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--smoke", action="store_true", help="tiny subset for a quick end-to-end test")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.split_seed)
    t0 = time.time()

    # 1. list capture files per class and assign each FILE to a split
    files = []
    for fam, sev in SEVERITY.items():
        fl = sorted((RAW / fam).rglob("*.csv"))
        if sev == 2:
            for f in fl:
                files.append((f, sev, LOW_SPLIT[f.name]))
            continue
        rng.shuffle(fl)
        files += [(f, sev, None) for f in fl]
    for sev in (0, 1):   # 70/15/15 of files per attack class
        fl = [x for x in files if x[1] == sev]
        n = len(fl); n_tr = round(0.70 * n); n_va = round(0.15 * n)
        for i, x in enumerate(fl):
            split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            files[files.index(x)] = (x[0], x[1], split)
    if a.smoke:
        keep = []
        for sev in (0, 1, 2):
            for sp in ("train", "val", "test"):
                keep += [x for x in files if x[1] == sev and x[2] == sp][:2]
        files = keep

    # 2. read captures -> windows (per split, per class, per file)
    buckets = {sp: {c: [] for c in (0, 1, 2)} for sp in ("train", "val", "test")}
    manifest = {"files": [], "rows_per_file": ROWS_PER_FILE, "window": W, "split_seed": a.split_seed}
    for k, (f, sev, sp) in enumerate(files, 1):
        nrows = ROWS_PER_FILE[sev] // (10 if a.smoke else 1)
        X = read_capture(f, nrows)
        Xw = windows(X)
        buckets[sp][sev].append((f.name, Xw))
        manifest["files"].append({"file": f.name, "family": f.parent.name if f.parent.name in SEVERITY else f.parts[-2],
                                  "severity": sev, "split": sp, "rows": int(len(X)), "windows": int(len(Xw))})
        if k % 20 == 0 or k == len(files):
            log(f"  read {k}/{len(files)} files  ({time.time()-t0:.0f}s)")

    # 3. transform: sign-preserving log1p, then StandardScaler fitted on TRAIN only
    tr_all = np.concatenate([w for c in (0, 1, 2) for _, w in buckets["train"][c]]).reshape(-1, len(FEATURES))
    f = lambda x: np.sign(x) * np.log1p(np.abs(x))
    mu, sd = f(tr_all).mean(0), f(tr_all).std(0) + 1e-8
    scale = lambda w: ((f(w) - mu) / sd).astype(np.float32)
    manifest["scaler"] = {"mean": mu.tolist(), "std": sd.tolist(), "transform": "sign*log1p then z-score (train only)"}

    # 4. allocate each class's windows to the 5 links in contiguous chunks
    links = [{sp: {"X": [], "y": [], "f": []} for sp in ("train", "val", "test")} for _ in range(5)]
    for sp in ("train", "val", "test"):
        for c in (0, 1, 2):
            items = [(name, i, w) for name, Xw in buckets[sp][c] for i, w in enumerate(Xw)]
            if not items:
                continue
            wts = np.array([LINK_MIX[k][c] for k in range(5)]); wts = wts / wts.sum()
            cuts = np.floor(np.cumsum(wts) * len(items)).astype(int); cuts[-1] = len(items)
            start = 0
            for k in range(5):
                chunk = items[start:cuts[k]]; start = cuts[k]
                for name, _, w in chunk:
                    links[k][sp]["X"].append(scale(w)); links[k][sp]["y"].append(c); links[k][sp]["f"].append(name)
    summary = {}
    for k in range(5):
        d = {}
        for sp in ("train", "val", "test"):
            X = np.stack(links[k][sp]["X"]) if links[k][sp]["X"] else np.zeros((0, W, len(FEATURES)), np.float32)
            y = np.array(links[k][sp]["y"], dtype=np.int64)
            d[f"X_{sp}"], d[f"y_{sp}"], d[f"file_{sp}"] = X, y, np.array(links[k][sp]["f"])
        np.savez_compressed(out / f"link_{k}.npz", **d)
        summary[f"link_{k}"] = {sp: np.bincount(d[f"y_{sp}"], minlength=3).tolist() for sp in ("train", "val", "test")}
        log(f"  link_{k}: " + "  ".join(f"{sp}={summary[f'link_{k}'][sp]}" for sp in ("train", "val", "test")))
    manifest["class_counts_per_link_[High,Medium,Low]"] = summary
    manifest["feature_order"] = FEATURE_ORDER
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    log(f"done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
