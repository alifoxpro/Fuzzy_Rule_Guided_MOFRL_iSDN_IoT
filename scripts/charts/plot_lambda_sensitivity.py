"""Lambda-sensitivity figure (phase B4 MSR, JFI, FPR) drawn from the same result files as
the manuscript's lambda tables.

CICIoT2023: the four runs of tab:lambda_ablation.
DIAD2024:   the mu = 0.10 runs of tab:lambda_mu_sweep (no lambda = 0 run exists).

    python scripts/charts/plot_lambda_sensitivity.py [output.png]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[2] / "experiments"
RUNS = {
    "CICIoT2023": ("ciciot2023", {0.00: "results_StrategyB_lam0.00", 0.10: "results_StrategyB_lam0.10",
                                  0.25: "results_StrategyB_mini_lam0.25", 0.50: "results_StrategyB_lam0.50"}),
    "DIAD2024": ("diad2024", {0.10: "results_StrategyB_lam0.10_mu0.10_diad", 0.25: "results_StrategyB",
                              0.50: "results_StrategyB_lam0.50_mu0.10_diad"}),
}
STYLE = {"CICIoT2023": dict(color="#d62728", marker="o"), "DIAD2024": dict(color="#1f77b4", marker="s")}


def b4(run, ds):
    with open(EXP / run / ds / "results.json", encoding="utf-8") as f:
        return json.load(f)["phase_b4"]


def main(out):
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for label, (ds, runs) in RUNS.items():
        lams = sorted(runs)
        p = [b4(runs[l], ds) for l in lams]
        st = STYLE[label]
        axes[0].errorbar(lams, [q["mean_msr"] for q in p], yerr=[q["std_msr"] for q in p], capsize=4, lw=2, ms=8, label=label, **st)
        axes[1].plot(lams, [q["jain_fairness"] for q in p], lw=2, ms=8, label=label, **st)
        axes[2].plot(lams, [q["mean_fpr"] for q in p], lw=2, ms=8, label=label, **st)
    titles = ["(a) MSR sensitivity", "(b) Fairness sensitivity", "(c) FPR sensitivity"]
    ylabels = ["Mean Success Rate (MSR)", "Jain Fairness Index (JFI)", "False-Positive Rate (FPR)"]
    for ax, ti, yl in zip(axes, titles, ylabels):
        ax.axvline(0.25, ls="--", color="grey", lw=1)
        ax.text(0.255, 0.02, r"$\lambda^{\star}=0.25$", transform=ax.get_xaxis_transform(), va="bottom", color="grey", fontsize=9)
        ax.set_title(ti, fontweight="bold")
        ax.set_xlabel(r"Fuzzy shaping weight $\lambda$")
        ax.set_ylabel(yl)
        ax.set_xticks([0.0, 0.1, 0.25, 0.5])
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fig_lambda_sensitivity.png")
