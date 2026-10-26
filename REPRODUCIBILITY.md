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
**all 460 cells of the 12 table groups reproduce** (`ALL TABLES MATCH`). Section 3
lists the result files that were completed or re-evaluated for this release.

## 2. Where each table comes from

| Manuscript table | Result files (`experiments/…`) | Produced by |
|---|---|---|
| Phase-by-phase MSR/JFI/FPR/FNR/Reward (`tab:phase_msr`) | `results_StrategyB/{dataset}/results.json` (seed 42) | `scripts/training/train_rl_multiagent_StrategyB.py` |
| Balanced metrics (`tab:balanced_metrics`) | `results_StrategyB/{dataset}/{mosar_fedavg,mosar_fedprox,fuzzy_retrain}/results.json` → `severity_action_matrix`, pooled over the 5 links; attack = High/Medium severity, positive = RATE_LIMIT or BLOCK | computed in `verify_paper_tables.py` |
| Phase-B0 rules (`tab:phase_b0`) | `results_StrategyB/{dataset}/results.json` → `phase_b0` | training script |
| Welch t-tests (`tab:stats`, `tab:strategy_ab_stats`) | stored mean/std of each phase, n = 50 evaluation episodes | computed in `verify_paper_tables.py` |
| λ ablation (`tab:lambda_ablation`) | `results_StrategyB_lam0.00`, `results_StrategyB_lam0.10`, `results_StrategyB_mini_lam0.25`, `results_StrategyB_lam0.50` | training script / `scripts/ablation/run_mini_lambda_sweep.py` |
| (λ, μ) sweep (`tab:lambda_mu_sweep`) | `results_StrategyB_lam*_mu*` (`_diad` = DIAD2024), `results_StrategyB_lam0.10`, `results_StrategyB_lam0.50`, `results_StrategyB` | `scripts/ablations/run_tier4_lammu_diad.py` |
| Fuzzy-TOPSIS (`tab:topsis_cc`) | `results/fuzzy_topsis/` (inputs: `results/{dataset}/results.json`) | `scripts/analysis/fuzzy_topsis_selection.py` |
| Rule statistics, stability, DQN agreement (`tab:rule_stats`, `tab:rule_stability`, `tab:rule_dqn_agreement`) | `results_StrategyB/{dataset}/afras_fl/*/phase5_rules.json`, `…/fuzzy_retrain/results.json` | `scripts/compute_rule_metrics_all.py` |
| Strategy A vs B (`tab:strategy_ab`) | `results_StrategyA/{dataset}/results.json`, `results_StrategyB/{dataset}/results.json` | `scripts/training/train_rl_multiagent_StrategyA.py` |
| Record-level test split (`tab:rls_test`, `tab:rls_balanced`, `tab:rls_l9_mitigation`) | `results_StrategyB_rls/{dataset}/evaluation_on_test.json`; CICIoT2023 B2/B3 from `evaluation_on_test_reevaluated.json`; CICIoT2023 pre-mitigation B4 from `results_StrategyB_rls_noL9_eps/ciciot2023/evaluation_on_test.json` | `scripts/training/evaluate_on_test_legacy.py` (50-episode evaluator used for these tables) |
| Five-seed replication (`tab:multiseed`) | `results_StrategyB` (42), `_s123`, `_s2026` (CICIoT2023) / `_seed_2026` (DIAD2024), `_seed_7`, `_seed_314` | training script with `--seed` |
| Ablation cells (`tab:ablation_6cell`) | `results_StrategyB_cell_a_scalar`, `_cell_b_mosar_only`, `_cell_d_mosar_afras`, `results_StrategyB`, `_pbrs` | `scripts/ablations/run_major_revision_queue.py`, `run_tier3_queue.py` |

Model checkpoints (`*.pt`), Mamba embedding caches (`*.npy`) and the datasets are
not included. The datasets are public: CICIoT2023
(<https://www.unb.ca/cic/datasets/iotdataset-2023.html>) and CIC IoT-DIAD 2024
(<https://www.unb.ca/cic/datasets/iot-diad-2024.html>); `scripts/data/` contains
the preprocessing and record-level split scripts.

## 3. Result files completed or re-evaluated for this release

- **λ ablation, λ = 0.00.** The original run (`results_StrategyB_lam0.00`) had stopped
  during phase B4 at round 15. It was resumed from its own checkpoint on 2026-10-25/26
  with the original command
  (`train_rl_multiagent_StrategyB.py --dataset ciciot2023 --lambda-fuzzy 0.00 --suffix _lam0.00`);
  the B1–B3 results are those of the original run.
- **Record-level test, CICIoT2023 B2/B3.** `results_StrategyB_rls/ciciot2023/evaluation_on_test.json`
  had been overwritten by the B4-only L9-mitigation re-run. The saved B2 and B3 models
  were re-evaluated on 2026-10-25 with the same evaluator
  (`evaluate_on_test_legacy.py --dataset ciciot2023 --phases B2,B3`, 50 episodes) into
  `evaluation_on_test_reevaluated.json`. The evaluator samples its episodes at random:
  re-evaluating the intact DIAD2024 models reproduced every MSR to four decimals, JFI
  within 0.0004 and FPR within 0.016.
- **Record-level test, CICIoT2023 pre-mitigation B4.** Its model had been replaced by
  the mitigated one, so B4 was retrained without the L9 rule-diversity step
  (`AFRAS_L9_DIVERSITY=0`) from the same B1–B3 checkpoints, seed 42, `--max-steps 500`,
  `--use-rls`, into `results_StrategyB_rls_noL9_eps`, and evaluated with the same
  evaluator. The manuscript reports this run.
- **Manuscript values corrected to the result files:** the Fuzzy-TOPSIS distances and
  DIAD2024 ranking, ablation cell (f) on CICIoT2023, five Reward cells of the (λ, μ)
  sweep, the λ = 0.00 row, the CICIoT2023 record-level rows, and last-digit rounding
  of six confidence-interval bounds and three rule-table cells.

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
5. **L9 switch.** The consequent-class diversity step (L9 mitigation) is on by
   default; `AFRAS_L9_DIVERSITY=0` mines the rule base without it.
6. **Exploration when phases are loaded from cache.** When B1–B3 are loaded from a
   previous run instead of trained, the code now applies the ε-decay those phases
   would have applied, so B4 starts at ε ≈ 0.448 as in an uninterrupted run. Before
   this fix, a run that loaded B1–B3 from cache started B4 at ε = 1.0; among the
   reported runs this affected the B4 of `results_StrategyB_s123/ciciot2023`
   (seed 123 of the five-seed table; ε 1.0 → 0.86 over B4). All other reported B4
   phases were trained after B1–B3 in the same process or resumed from a B4
   checkpoint, which stores ε (ε ends at 0.386).
7. **λ = 0.25 row of the λ ablation** comes from `results_StrategyB_mini_lam0.25`,
   a B4-only retraining (3 rounds × 2 local episodes) from the canonical B1–B3
   checkpoints; the other rows are full 30-round runs.
