# Order in which to run the reviewer-requested experiments

**Purpose:** deliver the numerical evidence for reviewer requests
R1-3/R2-5 (multi-seed), R2-4 (per-component ablation),
R2-6 (record-level split). Some cells are already done — those are
marked ✅. Everything else has a ready command line.

Hardware: NVIDIA RTX 5070 Laptop GPU is enough; each single run
(both phases B0–B5, one dataset) takes roughly 60–120 min. Budget
about **35–45 h** of wall-clock to complete the full plan below.

Recommended: run overnight in three batches (batch A, then B, then C),
resuming from the previous batch's results.

---

## Batch A — Multi-seed replication (R1-3, R2-5)

Two additional seeds on top of the existing `seed=42` run.

```bash
# seed 123
python scripts/training/train_rl_multiagent_StrategyB.py --dataset all --seed 123 --suffix _s123

# seed 2026
python scripts/training/train_rl_multiagent_StrategyB.py --dataset all --seed 2026 --suffix _s2026
```

Expected wall-clock: ~4 h per seed × 2 seeds = **~8 h**.

After it finishes:

```bash
python scripts/aggregate_multiseed.py \
  --dirs experiments/results_StrategyB experiments/results_StrategyB_s123 experiments/results_StrategyB_s2026 \
  --out results_reviewer/multiseed_summary.json
```

---

## Batch B — Per-component ablation (R2-4)

Existing runs already cover cells (b), (d) and (e) at seed=42:
- (b) MOSAR alone (λ=0, μ=0.10):    `results_StrategyB_lam0.00/`  ✅ (needs re-run with μ=0)
- (d) MOSAR+AFRAS-FL (λ=0.25, μ=0): NEW — must run
- (e) Complete (λ=0.25, μ=0.10):     `results_StrategyB/`          ✅
- (c) MOSAR+R_link (λ=0, μ=0.10):    ≈ same as `results_StrategyB_lam0.00/` ✅
- (f) PBRS baseline (Ng et al.):     NEW — see `train_pbrs_baseline.py`

Missing cells only:

```bash
# (b') MOSAR alone with μ=0 too (strict)
python scripts/training/train_rl_multiagent_StrategyB.py --dataset all --lambda-fuzzy 0.00 --mu-link 0.00 --seed 42 --suffix _R2-4_b_mosaronly

# (d) MOSAR+AFRAS-FL, μ=0
python scripts/training/train_rl_multiagent_StrategyB.py --dataset all --lambda-fuzzy 0.25 --mu-link 0.00 --seed 42 --suffix _R2-4_d_mosarfuzzy

# (f) PBRS baseline
python scripts/training/train_pbrs_baseline.py --dataset all --seed 42 --suffix _R2-4_f_pbrs
```

Expected wall-clock: ~3 × 4 h = **~12 h**.

---

## Batch C — Record-level train/val/test split (R2-6)

Requires the leak-free data pipeline in
`scripts/data/split_records_leakfree.py` (writes a `_split/` folder
containing per-client train/val/test CSVs partitioned BEFORE window
generation). Then re-run the canonical configuration on the split
data:

```bash
python scripts/data/split_records_leakfree.py --datasets ciciot2023 diad2024
python scripts/training/train_rl_multiagent_StrategyB.py --dataset all --seed 42 --suffix _R2-6_split --data-suffix _split
```

Expected wall-clock: ~30 min (split) + ~4 h (train) = **~5 h**.

---

## After all three batches

Regenerate the summary tables and populate the placeholders in the
manuscript:

```bash
python scripts/build_final_reviewer_tables.py
```

This writes `paper4-IEEE-IoT-J/reviewer_tables.tex` containing:
- Multi-seed mean-of-run table (replaces placeholder in Sec. V-C)
- 6-cell ablation table (replaces placeholder in Sec. V-D)
- Split-vs-no-split delta table (replaces placeholder in L8)

Then rebuild the PDF:

```bash
cd paper4-IEEE-IoT-J && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
