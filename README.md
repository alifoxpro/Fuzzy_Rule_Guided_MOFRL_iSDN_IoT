# AFRAS-FL: Fuzzy Rule-Guided Federated Deep Reinforcement Learning for SDN-IoT Intrusion Detection

Code and processed results for the manuscript

> Fadel, A. H., Karimpour, J., and Salmanian, Z.
> *Fuzzy Rule-Guided Exploration for Multi-Objective Federated Deep Reinforcement
> Learning in SDN-IoT Intrusion Detection.* Manuscript under review (MDPI), 2026.

```
scripts/       training, evaluation, preprocessing, ablation and analysis code
               (details: scripts/README.md)
  reproduce/   verify_paper_tables.py — recomputes every results table
experiments/   result files (JSON) written by the code, one folder per run
```

**To check the manuscript's tables against the result files**, see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md):

```bash
pip install numpy scipy
python scripts/reproduce/verify_paper_tables.py
```

**To retrain** (GPU recommended), install the requirements in
`scripts/README.md`, download CICIoT2023 and CIC IoT-DIAD 2024 from the
Canadian Institute for Cybersecurity, and run, for example,

```bash
python scripts/training/train_rl_multiagent_StrategyB.py --dataset ciciot2023 --seed 42
```

Released under the MIT license (see `LICENSE`).
