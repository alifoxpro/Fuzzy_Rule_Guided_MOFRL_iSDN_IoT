# Reproducibility guide

This repository accompanies the manuscript *"Fuzzy Rule-Guided Exploration for
Multi-Objective Federated Deep Reinforcement Learning in SDN-IoT Intrusion
Detection"*. It contains the training and evaluation code (`scripts/`) and the
processed result files written by that code (`experiments/`), from which every
results table of the manuscript can be recomputed.

## 1. Check the tables against the result files (no GPU, < 1 minute)

```bash
pip install numpy scipy
python scripts/reproduce/verify_paper_tables.py            # all tables
python scripts/reproduce/verify_paper_tables.py --table rls
```

The script rebuilds each table from `experiments/` and compares every cell with
the value printed in the manuscript, at the manuscript's precision. Its output for
this release is saved in
[`scripts/reproduce/verification_report.txt`](scripts/reproduce/verification_report.txt):
**401 cells reproduce exactly; 37 cells differ or have no result file** (listed in
Section 3).

## 2. Where each table comes from

| Manuscript table | Result files (`experiments/…`) | Produced by |
|---|---|---|
| Phase-by-phase MSR/JFI/FPR/FNR/Reward (`tab:phase_msr`) | `results_StrategyB/{dataset}/results.json` (seed 42) | `scripts/training/train_rl_multiagent_StrategyB.py` |
| Balanced metrics (`tab:balanced_metrics`) | `results_StrategyB/{dataset}/{mosar_fedavg,mosar_fedprox,fuzzy_retrain}/results.json` → `severity_action_matrix`, pooled over the 5 links; attack = High/Medium severity, positive = RATE_LIMIT or BLOCK | computed in `verify_paper_tables.py` |
| Phase-B0 rules (`tab:phase_b0`) | `results_StrategyB/{dataset}/results.json` → `phase_b0` | training script |
| Welch t-tests (`tab:stats`, `tab:strategy_ab_stats`) | stored mean/std of each phase, n = 50 evaluation episodes | computed in `verify_paper_tables.py` |
| λ ablation (`tab:lambda_ablation`) | `results_StrategyB_lam0.10`, `results_StrategyB_mini_lam0.25`, `results_StrategyB_lam0.50` | training script / `scripts/ablation/run_mini_lambda_sweep.py` |
| (λ, μ) sweep (`tab:lambda_mu_sweep`) | `results_StrategyB_lam*_mu*` (`_diad` = DIAD2024), `results_StrategyB_lam0.10`, `results_StrategyB_lam0.50`, `results_StrategyB` | `scripts/ablations/run_tier4_lammu_diad.py` |
| Fuzzy-TOPSIS (`tab:topsis_cc`) | `results/fuzzy_topsis/` (inputs: `results/{dataset}/results.json`) | `scripts/analysis/fuzzy_topsis_selection.py` |
| Rule statistics, stability, DQN agreement (`tab:rule_stats`, `tab:rule_stability`, `tab:rule_dqn_agreement`) | `results_StrategyB/{dataset}/afras_fl/*/phase5_rules.json`, `…/fuzzy_retrain/results.json` | `scripts/compute_rule_metrics_all.py` |
| Strategy A vs B (`tab:strategy_ab`) | `results_StrategyA/{dataset}/results.json`, `results_StrategyB/{dataset}/results.json` | `scripts/training/train_rl_multiagent_StrategyA.py` |
| Record-level test split (`tab:rls_test`, `tab:rls_balanced`, `tab:rls_l9_mitigation`) | `results_StrategyB_rls/{dataset}/evaluation_on_test.json` | `scripts/training/evaluate_on_test_legacy.py` (50-episode evaluator used for these tables) |
| Five-seed replication (`tab:multiseed`) | `results_StrategyB` (42), `_s123`, `_s2026` (CICIoT2023) / `_seed_2026` (DIAD2024), `_seed_7`, `_seed_314` | training script with `--seed` |
| Ablation cells (`tab:ablation_6cell`) | `results_StrategyB_cell_a_scalar`, `_cell_b_mosar_only`, `_cell_d_mosar_afras`, `results_StrategyB`, `_pbrs` | `scripts/ablations/run_major_revision_queue.py`, `run_tier3_queue.py` |

Model checkpoints (`*.pt`), Mamba embedding caches (`*.npy`) and the datasets are
not included. The datasets are public: CICIoT2023
(<https://www.unb.ca/cic/datasets/iotdataset-2023.html>) and CIC IoT-DIAD 2024
(<https://www.unb.ca/cic/datasets/iot-diad-2024.html>); `scripts/data/` contains
the preprocessing and record-level split scripts.

## 3. Cells that the result files do not reproduce

From `verification_report.txt`:

- **Fuzzy-TOPSIS, DIAD2024** — re-running `fuzzy_topsis_selection.py` reproduces the
  stored files exactly, which give LSTM-Only 0.8863 (rank 1), CNN-Transformer 0.7194
  (rank 2) and Mamba-Only 0.0846 (rank 3); the D⁺/D⁻ values of all three models also
  differ from the manuscript. CICIoT2023 D⁺/D⁻ differ in the fourth decimal.
- **Ablation cell (f), PBRS, CICIoT2023** — the result file gives B3 MSR 0.9334 and
  B4 FPR 0.0640.
- **(λ, μ) sweep** — five Reward cells differ (e.g. (0.25, 0.10) on CICIoT2023 is
  0.554 in `results_StrategyB`).
- **λ ablation, λ = 0.00 row** — `results_StrategyB_lam0.00` contains only a
  mid-training checkpoint; no result file exists for this row.
- **Record-level test, CICIoT2023 B2, B3 and the original (pre-mitigation) B4** —
  `results_StrategyB_rls/ciciot2023/evaluation_on_test.json` was overwritten by the
  B4-only L9-mitigation re-run and holds only that B4 result.
- **Rounding (≤ 0.001):** six 95 %-CI bounds of the five-seed table, three cells of
  the rule tables, and the mitigated CICIoT2023 FPR (0.2797).

## 4. Notes on the released code

These are properties of the code as released, stated so that the results can be
read correctly:

1. **Link term.** `SDNEnvironmentB._get_neighbor_actions()` (and the Strategy A
   equivalent) returns `None`, so the neighbour-consensus term `R_link` is 0 in every
   run and the weight `MU_LINK` has no effect. Runs that differ only in μ (e.g.
   `results_StrategyB_lam0.25_mu0.05` and `_mu0.20`) are byte-identical.
2. **Rule-agreement term.** In `results_StrategyB` (seed 42), `_lam0.10`, `_lam0.50`,
   `_mini_lam*`, `_s123` and `_s2026`, no deployed rule matched during training or
   evaluation: `history.fuzzy_reward_avg` is the constant no-match value −0.05 in every
   round. In `_cell_*`, `_pbrs`, `_rls`, `_seed_7`, `_seed_314` and `_seed_2026` the
   rules fire. This can be checked in each run's `fuzzy_retrain/results.json`.
3. **Strategy A reward.** `train_rl_multiagent_StrategyA.py` uses
   `0.4·R_correct + 0.3·R_security + 0.2·R_service + 0.1·R_neighbor` (with
   `R_neighbor = 0`, see note 1); it does not use the MOSAR components.
4. **Rule bases** are mined and deployed per client (phase P7 packages one rule set
   per link); they are not merged across clients.
