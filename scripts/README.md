# AFRAS-FL: Fuzzy-Augmented Federated Reinforcement Learning for SDN-IoT Intrusion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the source code, configuration files, and the
experimental harness used to produce all results, figures, and tables in
the paper:

> **Fadel, A. H.<sup>⭐</sup>, Karimpour, J., and Salmanian Kozehkonan, Z.**
> *"Fuzzy-Augmented Reward Shaping for Multi-Objective Deep Reinforcement
> Learning in Federated SDN-IoT Intrusion Detection."*
> Resubmitted to *IEEE Internet of Things Journal*, August 2026
> (original ID: IoT-66079-2026).
>
> <sup>⭐</sup> Corresponding author.

---

## Contents

```
.
├── analysis/                  # XAI feature selection & Fuzzy-TOPSIS evaluation
│   ├── analyze_state_vector.py
│   ├── extract_state_vectors.py
│   ├── fuzzy_topsis_selection.py
│   └── xai_unified_selection.py
│
├── data/                      # Data preprocessing
│   └── unify_features.py
│
├── training/                  # Main training scripts (Strategy A vs B)
│   ├── train_isolated.py
│   ├── train_rl_multiagent.py
│   ├── train_rl_multiagent_StrategyA.py     # Baseline: conventional 3-term reward (see REPRODUCIBILITY.md)
│   ├── train_rl_multiagent_StrategyB.py     # Full pipeline: MOSAR + AFRAS-FL
│   └── train_top3_models.py
│
├── charts/                    # Plot generation
│   ├── charts_comprehensive_comparison.py
│   ├── charts_state_vectors.py
│   ├── flowchart_strategy_comparison.py
│   └── regenerate_charts_*.py
│
└── ablation/                  # Lambda-ablation infrastructure
    ├── run_lambda_sweep.bat   # Production GPU sweep
    ├── run_b4_only_sweep.py   # Lightweight B4-only mini sweep
    ├── aggregate_lambda_results.py
    ├── compute_lambda_stats.py
    ├── generate_lambda_figure.py
    ├── inject_lambda_results.py
    ├── templates/             # LaTeX snippets for paper integration
    └── README_LAMBDA_ABLATION.md
```

## Key contributions

This repository implements two contributions introduced in the paper:

1. **MOSAR** — a five-component Multi-Objective Security-Aware Reward
   that decomposes agent feedback into detection, classification,
   latency, confidence, and impact terms.

2. **AFRAS-FL** — a seven-phase Adaptive Fuzzy Rule-Augmented Shaping
   pipeline that mines IF–THEN rules from on-policy agent experience
   and converts them into a potential-based shaping signal that is
   provably reward-preserving.

The combined reward is

```
R_total = R_MOSAR + lambda * R_fuzzy + mu * R_link
```

with default `lambda = 0.25` and `mu = 0.05`.

## Datasets

The paper evaluates the framework on two public IoT intrusion-detection
benchmarks:

* **CICIoT2023** — Canadian Institute for Cybersecurity, 33 attack types,
  millions of flow records.
* **DIAD2024** — used as an out-of-distribution stress test.

Both datasets are publicly available; download instructions are
documented in `data/unify_features.py`.

The processed CSVs and Mamba caches are **not** included in this
repository because of their size (>3 GB). They are regenerated
deterministically by `data/unify_features.py` from the raw datasets.

## Requirements

* Python 3.10 or newer
* PyTorch 2.0 or newer (CPU or CUDA)
* NumPy, pandas, scikit-learn, matplotlib, scipy
* Optional: seaborn (for some chart utilities)
* CUDA-capable GPU strongly recommended for `train_rl_multiagent_*`

```bash
pip install torch numpy pandas scikit-learn matplotlib scipy seaborn
```

## Reproducing the results in the paper

### 1. Strategy B (full pipeline, canonical λ = 0.25)

```bash
python training/train_rl_multiagent_StrategyB.py --dataset ciciot2023
python training/train_rl_multiagent_StrategyB.py --dataset diad2024
```

Each run takes ≈90 minutes on a single modern GPU and writes its
phase-by-phase results into
`experiments/results_StrategyB/{dataset}/results.json`.

### 2. Strategy A (baseline)

```bash
python training/train_rl_multiagent_StrategyA.py --dataset ciciot2023
```

### 3. λ-ablation (sensitivity analysis)

```bash
scripts\ablation\run_lambda_sweep.bat            # Windows
```

The sweep reuses Phases B0–B3 from the canonical Strategy B run and
re-runs only Phase B4 for `lambda ∈ {0.00, 0.10, 0.50}`. The
canonical `lambda = 0.25` cell is read from the existing
`results_StrategyB/` folder. Total wall-clock: ≈4.5 h on GPU.

### 4. Aggregate + plot the results

```bash
python ablation/aggregate_lambda_results.py
python ablation/compute_lambda_stats.py
python ablation/generate_lambda_figure.py
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{fadel2026afrasfl,
  title={Fuzzy-Augmented Federated Reinforcement Learning for Intrusion
         Detection in Software-Defined IoT Networks},
  author={Fadel, Ali Hussein and Karimpour, Jaber and
          Salmanian Kozehkonan, Zolfaghar},
  journal={IEEE Internet of Things Journal},
  year={2026},
  note={Manuscript submitted for publication}
}
```

## License

This code is released under the MIT License — see `LICENSE` for the
full text.

## Contact

* Ali Hussein Fadel (corresponding) — `ali_hussein_fadel@uodiyala.edu.iq` — ORCID: [0000-0003-1360-130X](https://orcid.org/0000-0003-1360-130X)
* Jaber Karimpour — `karimpour@tabrizu.ac.ir` — ORCID: [0000-0001-6193-3240](https://orcid.org/0000-0001-6193-3240)
* Zolfaghar Salmanian Kozehkonan — `salmanian@tabrizu.ac.ir` — ORCID: [0000-0002-1019-4123](https://orcid.org/0000-0002-1019-4123)

For questions about the experiments, please open an issue or contact
the corresponding authors.
