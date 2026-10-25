"""Recompute every results table of the paper from the released result files.

    python scripts/reproduce/verify_paper_tables.py            # all tables
    python scripts/reproduce/verify_paper_tables.py --table phase_msr

Each table is rebuilt from the JSON files under experiments/ and compared, cell by
cell, with the value printed in the manuscript (rounded to the manuscript's
precision). The script needs only numpy and scipy; it does not retrain anything.
Exit status is 0 when every cell matches.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
LINKS = ["Link_1_Core", "Link_2_Aggregation_L", "Link_3_Aggregation_R", "Link_4_Edge_L", "Link_5_Edge_R"]
SHORT = dict(zip(LINKS, ["L1-Core", "L2-AggrL", "L3-AggrR", "L4-EdgeL", "L5-EdgeR"]))
DS = {"ciciot2023": "CICIoT2023", "diad2024": "DIAD2024"}
N_EPISODES = 50  # evaluation episodes behind every per-run mean +/- std


def load(*parts):
    with open(EXP.joinpath(*parts), encoding="utf-8") as f:
        return json.load(f)


def phase(run, ds, ph):
    return load(run, ds, "results.json")[f"phase_{ph}"]


class Table:
    def __init__(self, name, title):
        self.name, self.title, self.rows, self.fail = name, title, [], 0

    def check(self, label, got, want, nd):
        ok = round(float(got), nd) == round(float(want), nd)
        self.fail += not ok
        self.rows.append(f"  {'ok ' if ok else 'XX '} {label:52s} computed {got:.{nd}f}   paper {want:.{nd}f}")

    def missing(self, label, want, why):
        self.fail += 1
        self.rows.append(f"  --  {label:52s} no result file ({why})   paper {want}")

    def report(self):
        status = "PASS" if self.fail == 0 else f"FAIL ({self.fail} cell(s))"
        print(f"\n== {self.name}: {self.title} -- {status}")
        print("\n".join(self.rows))
        return self.fail


def welch(m1, s1, m2, s2, n=N_EPISODES):
    t, p = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
    d = (m1 - m2) / math.sqrt((s1 ** 2 + s2 ** 2) / 2)
    return t, p, d


def binary_from_severity(run, ds, sub):
    """Pooled binary counts. Attack = High/Medium severity rows; positive = RATE_LIMIT or BLOCK."""
    r = load(run, ds, sub, "results.json")
    tp = fp = fn = tn = 0
    for ln in LINKS:
        m = r[ln]["severity_action_matrix"]  # rows: severity High/Medium/Low, cols: ALLOW/RATE_LIMIT/BLOCK
        for i in range(3):
            for j in range(3):
                att, pos = i in (0, 1), j in (1, 2)
                c = m[i][j]
                tp += c * (att and pos); fn += c * (att and not pos)
                fp += c * (not att and pos); tn += c * (not att and not pos)
    acc = (tp + tn) / (tp + fp + fn + tn)
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    pn, rn = tn / (tn + fn), tn / (tn + fp)
    f1n = 2 * pn * rn / (pn + rn)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, macro_f1=(f1 + f1n) / 2, mcc=mcc)


# --------------------------------------------------------------------------- tables
def t_phase_msr():
    T = Table("phase_msr", "Phase-by-phase results, canonical run (experiments/results_StrategyB)")
    want = {
        ("ciciot2023", "b1"): (0.9985, 0.0018, 1.0000, 0.0004, 0.0000, 0.566),
        ("ciciot2023", "b2"): (0.9817, 0.0147, 0.9998, 0.0199, 0.0008, 0.556),
        ("ciciot2023", "b3"): (0.9978, 0.0028, 1.0000, 0.0036, 0.0053, 0.565),
        ("ciciot2023", "b4"): (0.9990, 0.0010, 1.0000, 0.0026, 0.0001, 0.554),
        ("diad2024", "b1"): (0.9996, 0.0004, 1.0000, 0.0001, 0.0000, 0.506),
        ("diad2024", "b2"): (0.9318, 0.0565, 0.9963, 0.2000, 0.0002, 0.470),
        ("diad2024", "b3"): (0.9028, 0.1036, 0.9870, 0.0018, 0.0212, 0.454),
        ("diad2024", "b4"): (0.8474, 0.1505, 0.9694, 0.0421, 0.0415, 0.414),
    }
    keys = ["mean_msr", "std_msr", "jain_fairness", "mean_fpr", "mean_fnr", "mean_reward"]
    for (ds, ph), w in want.items():
        p = phase("results_StrategyB", ds, ph)
        for k, v, nd in zip(keys, w, (4, 4, 4, 4, 4, 3)):
            T.check(f"{DS[ds]} {ph.upper()} {k}", p[k], v, nd)
    return T


def t_balanced():
    T = Table("balanced_metrics", "Binary detection metrics from the severity-action counts (canonical run)")
    want = {
        ("ciciot2023", "mosar_fedavg"): (0.9967, 0.9973, 0.9992, 0.9983, 0.9861, 0.9723),
        ("ciciot2023", "mosar_fedprox"): (0.9985, 0.9998, 0.9985, 0.9992, 0.9935, 0.9870),
        ("ciciot2023", "fuzzy_retrain"): (0.9998, 0.9998, 1.0000, 0.9999, 0.9991, 0.9982),
        ("diad2024", "mosar_fedavg"): (0.9512, 0.9433, 0.9998, 0.9707, 0.9128, 0.8389),
        ("diad2024", "mosar_fedprox"): (0.9554, 0.9515, 0.9957, 0.9731, 0.9217, 0.8513),
        ("diad2024", "fuzzy_retrain"): (0.8539, 0.8999, 0.9217, 0.9107, 0.7552, 0.5119),
    }
    for (ds, sub), w in want.items():
        m = binary_from_severity("results_StrategyB", ds, sub)
        for k, v in zip(["acc", "prec", "rec", "f1", "macro_f1", "mcc"], w):
            T.check(f"{DS[ds]} {sub} {k}", m[k], v, 4)
    return T


def t_phase_b0():
    T = Table("phase_b0", "Phase-B0 rule extraction (results.json -> phase_b0)")
    want = {
        "ciciot2023": [(4, 3, 1, 0, 0.972), (9, 1, 0, 8, 0.794), (8, 1, 3, 4, 0.761), (9, 0, 0, 9, 1.000), (8, 1, 2, 5, 0.779)],
        "diad2024": [(6, 5, 1, 0, 0.999), (6, 5, 1, 0, 1.000), (7, 6, 1, 0, 1.000), (6, 5, 1, 0, 0.999), (6, 5, 1, 0, 1.000)],
    }
    for ds, rows in want.items():
        b0 = load("results_StrategyB", ds, "results.json")["phase_b0"]
        for ln, w in zip(LINKS, rows):
            g = b0[ln]
            for k, v, nd in zip(["total_rules", "block_rules", "limit_rules", "allow_rules", "avg_confidence"], w, (0, 0, 0, 0, 3)):
                T.check(f"{DS[ds]} {SHORT[ln]} {k}", g[k], v, nd)
    return T


def t_stats():
    T = Table("stats", "Episode-level Welch t-tests, B4 vs B2/B3 (n = 50 per phase)")
    want = {
        ("ciciot2023", "b2"): (8.35, 5.1e-11, 1.67, 14.6), ("ciciot2023", "b3"): (2.94, 4.7e-3, 0.59, 2.8),
        ("diad2024", "b2"): (-3.71, 4.4e-4, -0.74, 0.38), ("diad2024", "b3"): (-2.14, 3.5e-2, -0.43, 0.69),
    }
    for (ds, other), (wt, wp, wd, wr) in want.items():
        b4, o = phase("results_StrategyB", ds, "b4"), phase("results_StrategyB", ds, other)
        t, p, d = welch(b4["mean_msr"], b4["std_msr"], o["mean_msr"], o["std_msr"])
        lab = f"{DS[ds]} B4 vs {other.upper()}"
        T.check(lab + " t", t, wt, 2)
        T.check(lab + " p (2 significant digits)", float(f"{p:.1e}"), wp, 12)
        T.check(lab + " Cohen d", d, wd, 2)
        ratio = o["std_msr"] / b4["std_msr"]
        T.check(lab + " sigma ratio", ratio, wr, 1 if wr >= 1 else 2)
    return T


B4_KEYS = ["mean_msr", "jain_fairness", "mean_fpr", "mean_reward"]


def t_lambda():
    T = Table("lambda_ablation", "Lambda ablation on CICIoT2023, phase B4")
    T.missing("lambda=0.00 (MSR/std/JFI/FPR/Reward)", "0.9941+-0.0099 / 1.0000 / 0.0068 / 0.545",
              "results_StrategyB_lam0.00 holds only a mid-training checkpoint")
    rows = {"0.10": ("results_StrategyB_lam0.10", (0.9869, 0.0238, 0.9994, 0.0049, 0.553)),
            "0.25": ("results_StrategyB_mini_lam0.25", (0.9995, 0.0008, 1.0000, 0.0143, 0.573)),
            "0.50": ("results_StrategyB_lam0.50", (0.9993, 0.0011, 1.0000, 0.0019, 0.542))}
    for lam, (run, w) in rows.items():
        p = phase(run, "ciciot2023", "b4")
        for k, v, nd in zip(["mean_msr", "std_msr", "jain_fairness", "mean_fpr", "mean_reward"], w, (4, 4, 4, 4, 3)):
            T.check(f"lambda={lam} {k} [{run}]", p[k], v, nd)
    return T


def t_lambda_mu():
    T = Table("lambda_mu_sweep", "(lambda, mu) sweep, phase B4")
    rows = [  # (lambda, mu), CICIoT2023 run, paper CIC, DIAD2024 run, paper DIAD
        ("0.10,0.10", "results_StrategyB_lam0.10", (0.9869, 0.9994, 0.0049, 0.553), "results_StrategyB_lam0.10_mu0.10_diad", (0.9948, 1.0000, 0.0000, 0.529)),
        ("0.25,0.10", "results_StrategyB", (0.9990, 1.0000, 0.0026, 0.573), "results_StrategyB", (0.8474, 0.9694, 0.0421, 0.414)),
        ("0.50,0.10", "results_StrategyB_lam0.50", (0.9993, 1.0000, 0.0019, 0.542), "results_StrategyB_lam0.50_mu0.10_diad", (0.9964, 1.0000, 0.0080, 0.531)),
        ("0.25,0.05", "results_StrategyB_lam0.25_mu0.05", (0.9397, 0.9843, 0.0045, 0.549), "results_StrategyB_lam0.25_mu0.05_diad", (0.9976, 1.0000, 0.0000, 0.543)),
        ("0.25,0.20", "results_StrategyB_lam0.25_mu0.20", (0.9397, 0.9843, 0.0045, 0.549), "results_StrategyB_lam0.25_mu0.20_diad", (0.9959, 1.0000, 0.0480, 0.542)),
        ("0.10,0.20", "results_StrategyB_lam0.10_mu0.20", (0.9988, 1.0000, 0.0000, 0.578), "results_StrategyB_lam0.10_mu0.20_diad", (0.9989, 1.0000, 0.0080, 0.527)),
    ]
    for lm, rc, wc, rd, wd in rows:
        for ds, run, w in (("ciciot2023", rc, wc), ("diad2024", rd, wd)):
            p = phase(run, ds, "b4")
            for k, v, nd in zip(B4_KEYS, w, (4, 4, 4, 3)):
                T.check(f"({lm}) {DS[ds]} {k} [{run}]", p[k], v, nd)
    return T


def _topsis_report(ds):
    """D+/D- are written only to the text report; CC and rank are in the JSON."""
    rows, on = {}, False
    for line in (EXP / "results" / "fuzzy_topsis" / ds / "fuzzy_topsis_report.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("Model") and "D+" in line:
            on = True
            continue
        if on:
            parts = line.split()
            if len(parts) != 5:
                break
            rows[parts[0]] = tuple(float(x) for x in parts[1:4]) + (int(parts[4]),)
    return rows


def t_topsis():
    T = Table("topsis_cc", "Fuzzy-TOPSIS backbone audit (results/fuzzy_topsis/, regenerated by analysis/fuzzy_topsis_selection.py)")
    want = {
        "ciciot2023": {"Mamba-Only": (0.6432, 1.1893, 0.6489, 1), "CNN-Transformer": (1.1856, 0.6462, 0.3528, 2), "LSTM-Only": (1.2489, 0.5942, 0.3225, 3)},
        "diad2024": {"LSTM-Only": (0.2841, 2.2131, 0.8863, 1), "Mamba-Only": (0.9421, 1.5386, 0.6202, 2), "CNN-Transformer": (1.4507, 1.0432, 0.4183, 3)},
    }
    for ds, rows in want.items():
        rep = _topsis_report(ds)
        js = load("results", "fuzzy_topsis", ds, "fuzzy_topsis_results.json")
        for model, (dp, dm, cc, rank) in rows.items():
            T.check(f"{DS[ds]} {model} D+", rep[model][0], dp, 4)
            T.check(f"{DS[ds]} {model} D-", rep[model][1], dm, 4)
            T.check(f"{DS[ds]} {model} CC", js["closeness_coefficients"][model], cc, 4)
            T.check(f"{DS[ds]} {model} rank", js["ranking"].index(model) + 1, rank, 0)
    return T


def _rules(ds, ln):
    return load("results_StrategyB", ds, "afras_fl", ln, "phase5_rules.json")


def t_rules():
    T = Table("rule_stats / rule_stability / rule_dqn_agreement", "Rule-base statistics (phase5_rules.json + fuzzy_retrain/results.json)")
    want_stats = {
        "ciciot2023": [(6, 0.941, 49796, "4/2/0"), (11, 0.800, 16386, "1/1/9"), (10, 0.749, 9846, "1/4/5"), (11, 0.919, 10557, "1/1/9"), (11, 0.748, 10871, "1/3/7")],
        "diad2024": [(9, 0.937, 19467, "6/2/1"), (9, 0.927, 16508, "6/2/1"), (10, 0.974, 24394, "7/2/1"), (8, 0.913, 8979, "5/2/1"), (9, 0.950, 13677, "6/2/1")],
    }
    want_agree = {
        "ciciot2023": [(1.000, 0.667, 0.333), (0.982, 0.091, 0.891), (0.706, 0.100, 0.606), (0.255, 0.091, 0.164), (0.966, 0.091, 0.875)],
        "diad2024": [(0.494, 0.667, 0.173), (0.544, 0.667, 0.123), (0.142, 0.700, 0.558), (0.114, 0.625, 0.511), (0.278, 0.667, 0.389)],
    }
    want_stab = {"ciciot2023": (11, 9, 13, 5, 11), "diad2024": (6, 6, 7, 3, 6)}
    for ds in DS:
        b4 = load("results_StrategyB", ds, "fuzzy_retrain", "results.json")
        ante = {}
        for ln, ws, wa in zip(LINKS, want_stats[ds], want_agree[ds]):
            rules = _rules(ds, ln)
            acts = [r["consequent"] for r in rules]
            bla = f'{acts.count("BLOCK")}/{acts.count("RATE_LIMIT")}/{acts.count("ALLOW")}'
            lab = f"{DS[ds]} {SHORT[ln]}"
            T.check(lab + " #rules", len(rules), ws[0], 0)
            T.check(lab + " avg conf", np.mean([r["confidence"] for r in rules]), ws[1], 3)
            T.check(lab + " avg supp", np.mean([r["support"] for r in rules]), ws[2], 0)
            T.fail += bla != ws[3]
            T.rows.append(f"  {'ok ' if bla == ws[3] else 'XX '} {lab + ' B/L/A':52s} computed {bla}   paper {ws[3]}")
            ad = b4[ln]["action_distribution"]
            dqn_b = ad.get("BLOCK", 0) / sum(ad.values())
            rule_b = acts.count("BLOCK") / len(rules)
            T.check(lab + " DQN BLOCK frac", dqn_b, wa[0], 3)
            T.check(lab + " rule BLOCK frac", rule_b, wa[1], 3)
            T.check(lab + " |delta|", abs(dqn_b - rule_b), wa[2], 3)
            for r in rules:
                ante.setdefault(tuple(sorted((a[0], a[1]) for a in r["antecedents"])), []).append(r["consequent"])
        s2 = sum(len(v) >= 2 for v in ante.values()); s3 = sum(len(v) >= 3 for v in ante.values())
        same = sum(len(v) >= 2 and len(set(v)) == 1 for v in ante.values())
        w = want_stab[ds]
        for lab, g, v in [("shared>=2", s2, w[0]), ("shared>=3", s3, w[1]), ("unique antecedents", len(ante), w[2]),
                          ("consistent consequents", same, w[3]), ("shared antecedents (denominator)", s2, w[4])]:
            T.check(f"{DS[ds]} {lab}", g, v, 0)
    return T


def t_strategy_ab():
    T = Table("strategy_ab / strategy_ab_stats", "Strategy A (results_StrategyA) vs Strategy B (results_StrategyB)")
    pairs = [("a", "b1", "Indep."), ("b", "b2", "FedAvg"), ("c", "b3", "FedProx")]
    want = {
        "ciciot2023": [(0.8903, 0.1120, 0.9985, 0.0018), (0.6511, 0.2039, 0.9817, 0.0147), (0.8227, 0.1357, 0.9978, 0.0028)],
        "diad2024": [(0.9758, 0.0080, 0.9996, 0.0004), (0.8431, 0.1401, 0.9318, 0.0565), (0.9361, 0.0378, 0.9028, 0.1036)],
    }
    want_fpr = {"ciciot2023": (0.0750, 0.0036), "diad2024": (0.2698, 0.0018)}
    want_t = {("ciciot2023", "FedAvg"): (11.43, 1.8e-15, 2.29), ("ciciot2023", "FedProx"): (9.19, 3.1e-12, 1.84),
              ("diad2024", "FedAvg"): (4.15, 9.8e-5, 0.83), ("diad2024", "FedProx"): (-4.04, 1.7e-4, -0.81)}
    for ds in DS:
        A = load("results_StrategyA", ds, "results.json")
        for (pa, pb, lab), w in zip(pairs, want[ds]):
            a, b = A[f"phase_{pa}"], phase("results_StrategyB", ds, pb)
            T.check(f"{DS[ds]} {lab} A MSR", a["mean_msr"], w[0], 4); T.check(f"{DS[ds]} {lab} A std", a["std_msr"], w[1], 4)
            T.check(f"{DS[ds]} {lab} B MSR", b["mean_msr"], w[2], 4); T.check(f"{DS[ds]} {lab} B std", b["std_msr"], w[3], 4)
        T.check(f"{DS[ds]} FedProx FPR A", A["phase_c"]["mean_fpr"], want_fpr[ds][0], 4)
        T.check(f"{DS[ds]} FedProx FPR B", phase("results_StrategyB", ds, "b3")["mean_fpr"], want_fpr[ds][1], 4)
        for lab, (pa, pb) in [("FedAvg", ("b", "b2")), ("FedProx", ("c", "b4"))]:
            a, b = A[f"phase_{pa}"], phase("results_StrategyB", ds, pb)
            t, p, d = welch(b["mean_msr"], b["std_msr"], a["mean_msr"], a["std_msr"])
            wt, wp, wd = want_t[(ds, lab)]
            T.check(f"{DS[ds]} A-{lab} vs B-{pb.upper()} t", t, wt, 2)
            T.check(f"{DS[ds]} A-{lab} vs B-{pb.upper()} p", float(f"{p:.1e}"), wp, 12)
            T.check(f"{DS[ds]} A-{lab} vs B-{pb.upper()} Cohen d", d, wd, 2)
    return T


def t_multiseed():
    T = Table("multiseed", "Five-seed mean-of-run MSR (seeds 42, 123, 2026, 7, 314)")
    runs = {"ciciot2023": ["results_StrategyB", "results_StrategyB_s123", "results_StrategyB_s2026", "results_StrategyB_seed_7", "results_StrategyB_seed_314"],
            "diad2024": ["results_StrategyB", "results_StrategyB_s123", "results_StrategyB_seed_2026", "results_StrategyB_seed_7", "results_StrategyB_seed_314"]}
    want = {("ciciot2023", "b2"): (0.9389, 0.0477, 0.8797, 0.9981), ("ciciot2023", "b3"): (0.9835, 0.0206, 0.9579, 1.0091),
            ("ciciot2023", "b4"): (0.9847, 0.0234, 0.9556, 1.0138), ("diad2024", "b2"): (0.8904, 0.1216, 0.7394, 1.0414),
            ("diad2024", "b3"): (0.9591, 0.0408, 0.9084, 1.0098), ("diad2024", "b4"): (0.9604, 0.0645, 0.8803, 1.0405)}
    tq = stats.t.ppf(0.975, 4)
    for (ds, ph), w in want.items():
        x = np.array([phase(r, ds, ph)["mean_msr"] for r in runs[ds]])
        m, s = x.mean(), x.std(ddof=1)
        h = tq * s / math.sqrt(len(x))
        for lab, g, v in zip(["mean", "std", "CI low", "CI high"], [m, s, m - h, m + h], w):
            T.check(f"{DS[ds]} {ph.upper()} {lab}", g, v, 4)
    return T


def t_ablation():
    T = Table("ablation_6cell", "Per-component ablation cells")
    cells = {
        "(a) scalar reward": ("results_StrategyB_cell_a_scalar", {"ciciot2023": (0.9982, 0.9894, 0.0000), "diad2024": (0.9996, 0.9987, 0.0000)}),
        "(b) MOSAR only": ("results_StrategyB_cell_b_mosar_only", {"ciciot2023": (0.9909, 0.9994, 0.0000), "diad2024": (0.9989, 0.9978, 0.0000)}),
        "(d) MOSAR + AFRAS-FL (mu=0)": ("results_StrategyB_cell_d_mosar_afras", {"ciciot2023": (0.9909, 0.9397, 0.0045), "diad2024": (0.9989, 0.9959, 0.0480)}),
        "(e) complete R_total": ("results_StrategyB", {"ciciot2023": (0.9978, 0.9990, 0.0026), "diad2024": (0.9028, 0.8474, 0.0421)}),
        "(f) MOSAR + PBRS": ("results_StrategyB_pbrs", {"ciciot2023": (0.9978, 0.9930, 0.0000), "diad2024": (0.9959, 0.9956, 0.0000)}),
    }
    for lab, (run, rows) in cells.items():
        for ds, (b3, b4, fpr) in rows.items():
            T.check(f"{lab} {DS[ds]} B3 MSR", phase(run, ds, "b3")["mean_msr"], b3, 4)
            T.check(f"{lab} {DS[ds]} B4 MSR", phase(run, ds, "b4")["mean_msr"], b4, 4)
            T.check(f"{lab} {DS[ds]} B4 FPR", phase(run, ds, "b4")["mean_fpr"], fpr, 4)
    return T


def _rls_phases(ds):
    """Record-level TEST evaluation written by training/evaluate_on_test_legacy.py (50 episodes).
    A re-evaluation of the saved models, if present, is used only for phases the original file lacks."""
    base = EXP / "results_StrategyB_rls" / ds
    orig = load("results_StrategyB_rls", ds, "evaluation_on_test.json")["phases"]
    src = {k: "evaluation_on_test.json" for k in orig}
    re_file = base / "evaluation_on_test_reevaluated.json"
    if re_file.exists():
        for k, v in json.load(open(re_file, encoding="utf-8"))["phases"].items():
            if k not in orig:
                orig[k], src[k] = v, re_file.name
    return orig, src


def t_rls():
    T = Table("rls_test / rls_balanced / rls_l9_mitigation", "Record-level TEST split (results_StrategyB_rls)")
    paper = {
        "diad2024": {"B2_FedAvg": (0.7928, 0.9445, 0.066, 0.000), "B3_FedProx": (0.9103, 0.9915, 0.066, 0.000), "B4_Fuzzy": (0.9038, 0.9837, 0.046, 0.000)},
        "ciciot2023": {"B2_FedAvg": (0.8642, 0.9731, 0.800, 0.000), "B3_FedProx": (0.6292, 0.7906, 0.451, 0.000)},
    }
    for ds, rows in paper.items():
        ph, src = _rls_phases(ds)
        for name, (msr, jfi, fpr, fnr) in rows.items():
            lab = f"{DS[ds]} {name}"
            if name not in ph:
                T.missing(lab + " MSR/JFI/FPR/FNR", f"{msr}/{jfi}/{fpr}/{fnr}", "overwritten by the B4-only L9 re-run on 2026-08-29")
                continue
            p = ph[name]
            T.check(f"{lab} MSR [{src[name]}]", p["msr_mean_of_links"], msr, 4)
            T.check(f"{lab} JFI [{src[name]}]", p["jfi"], jfi, 4)
            T.check(f"{lab} FPR [{src[name]}]", p["fpr_mean_of_links"], fpr, 3)
            T.check(f"{lab} FNR [{src[name]}]", p["fnr_mean_of_links"], fnr, 3)
            T.check(f"{lab} specificity", 1 - p["fpr_mean_of_links"], 1 - fpr, 3)
            T.check(f"{lab} balanced acc.", 1 - (p["fpr_mean_of_links"] + p["fnr_mean_of_links"]) / 2, 1 - (fpr + fnr) / 2, 3)
    # CICIoT2023 B4: the paper reports the original (pre-mitigation) run and the L9-mitigated run.
    T.missing("CICIoT2023 B4 original MSR/JFI/FPR", "0.6959/0.9073/0.737",
              "the B4 model and its evaluation were replaced by the L9-mitigated re-run")
    ph, src = _rls_phases("ciciot2023")
    p = ph["B4_Fuzzy"]
    T.check("CICIoT2023 B4 L9-mitigated MSR", p["msr_mean_of_links"], 0.6786, 4)
    T.check("CICIoT2023 B4 L9-mitigated FPR", p["fpr_mean_of_links"], 0.279, 3)
    T.check("CICIoT2023 B4 L9-mitigated JFI", p["jfi"], 0.8145, 4)
    T.check("CICIoT2023 B4 L9-mitigated specificity", 1 - p["fpr_mean_of_links"], 0.721, 3)
    T.check("CICIoT2023 B4 L9-mitigated balanced acc.", 1 - (p["fpr_mean_of_links"] + p["fnr_mean_of_links"]) / 2, 0.860, 3)
    return T

TABLES = {"phase_msr": t_phase_msr, "balanced_metrics": t_balanced, "phase_b0": t_phase_b0, "stats": t_stats,
          "lambda_ablation": t_lambda, "lambda_mu_sweep": t_lambda_mu, "topsis_cc": t_topsis, "rules": t_rules,
          "strategy_ab": t_strategy_ab, "rls": t_rls, "multiseed": t_multiseed, "ablation_6cell": t_ablation}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", choices=sorted(TABLES), help="check one table only")
    args = ap.parse_args()
    names = [args.table] if args.table else list(TABLES)
    fails = sum(TABLES[n]().report() for n in names)
    print(f"\n{'ALL TABLES MATCH' if fails == 0 else f'{fails} cell(s) differ'} ({len(names)} table group(s) checked)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
