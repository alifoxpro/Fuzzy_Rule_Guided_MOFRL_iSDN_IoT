# GPU λ-sweep — Hand-over instructions

This document explains **exactly** how to replace the current
preview-quality (CPU mini-sweep) λ-ablation numbers in the paper with
publication-quality numbers from a full GPU run.

You will **NOT** need to touch the paper tex, the figures, the tables,
the statistical tests, or the captions. The aggregator, the figure
generator, the statistics script, and the injector have all been
updated so that the production numbers **override** the preview numbers
automatically.

---

## Current state (preview quality)

The paper currently contains λ-ablation results produced by a CPU mini
sweep with a reduced training config:

| Parameter | Canonical (full) | Mini sweep (current) |
|---|---|---|
| `MAX_ROWS` per link | `3,000,000` | `30,000` |
| `FL_ROUNDS` (B4) | `30` | `3` |
| `LOCAL_EPISODES_PER_ROUND` | `5` | `2` |
| Datasets | both | both |
| λ values | 4 | 4 (`0.00, 0.10, 0.25, 0.50`) |
| Wall-clock per run | ~90 min (GPU) | ~4 min (CPU) |

These numbers are stored in:
```
experiments/results_StrategyB_mini_lam{X.XX}/{dataset}/results.json
```

They are good enough to prove the plumbing works. They are **not**
good enough for Scientific Reports. Replace them with the full sweep
before submitting.

---

## Step-by-step: run the full sweep on GPU

### Prerequisites

1. **GPU working.**
   ```cmd
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```
   Must print `True`. If it prints `False`, stop — on CPU the full
   training takes ~15 h per run instead of ~90 min.

2. **Python packages.**
   All the packages used by the training script, plus `scipy` for
   the statistics step. The current environment already has
   everything we need (checked).

3. **Disk space.**
   Each full run writes ~300 MB (data + checkpoints + charts). Three
   new λ values × two datasets = six folders ≈ 1.8 GB.

### Step 1 — run the sweep

```cmd
cd /d "D:\My Project\FL-RLMAMS_stage3"
scripts\ablation\run_lambda_sweep.bat
```

This runs 3 training jobs sequentially on `ciciot2023` with
λ ∈ {0.00, 0.10, 0.50} (the 0.25 cell is reused from the canonical
`experiments/results_StrategyB/` folder, so we do not re-run it).

Expected time: **~4.5 hours** on a single GPU.

#### Where each run writes

```
experiments/results_StrategyB_lam0.00/ciciot2023/
experiments/results_StrategyB_lam0.10/ciciot2023/
experiments/results_StrategyB_lam0.50/ciciot2023/
```

#### To also run DIAD2024

Open `scripts\ablation\run_lambda_sweep.bat` in a text editor and
change:
```
set DATASET=ciciot2023
```
to:
```
set DATASET=all
```
or just run the batch file twice (once with `ciciot2023`, once with
`diad2024`). DIAD2024 adds another ~4.5 hours.

#### Resume if the sweep is interrupted

Each run writes a `progress_state.json` inside its own folder. If a
run fails or you kill it mid-way:

- The production script `train_rl_multiagent_StrategyB.py` reads
  `progress_state.json` at startup and skips any phase that is
  already marked complete (B0, B1, B2, B3, B4, B5).
- The Mamba cache (`mamba_cache/`) is also preserved, so data loading
  is fast on the second attempt.
- Just re-run `run_lambda_sweep.bat` — the completed lambdas will
  print `"Phase BX: COMPLETED (loading)"` and be skipped; the
  interrupted lambda will resume from its last checkpoint.

### Step 2 — collect the numbers

```cmd
python scripts\ablation\aggregate_lambda_results.py
```

The aggregator's priority order is:
1. `results_StrategyB_lam{X.XX}/` (full production run) — **highest**
2. `results_StrategyB_mini_lam{X.XX}/` (B4-only CPU mini sweep)
3. `results_StrategyB/` (canonical, only for λ=0.25) — **lowest**

So once the full sweep is done, the aggregator **automatically**
picks up the production numbers and ignores the mini numbers.

### Step 3 — re-inject into the paper

```cmd
python scripts\ablation\inject_lambda_results.py
```

This runs the full pipeline:
1. `compute_lambda_stats.py` (Welch t-tests)
2. `generate_lambda_figure.py` (matplotlib)
3. Re-renders the three LaTeX templates with the new numbers
4. Replaces the content between the `% <<<LAMBDA_R31_BEGIN>>>` /
   `% <<<LAMBDA_R31_END>>>` markers in `main.tex` (and the two
   other marker pairs for L2 and the abstract)

The injection is **idempotent** — running it a second time
overwrites the old block cleanly.

### Step 4 — recompile the paper

```cmd
cd paper4-Fuzzy-Augmented-Reward-Shaping
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Expected: 24 pages, 0 errors. The R3.1 section, the
`tab:lambda_ablation` table, and the `fig:lambda_sensitivity` figure
now contain the production numbers.

### Step 5 — clean up the mini sweep (optional)

Once you are satisfied with the production numbers, you can delete
the CPU mini-sweep folders to free disk space:

```cmd
rmdir /s /q experiments\results_StrategyB_mini_lam0.00
rmdir /s /q experiments\results_StrategyB_mini_lam0.10
rmdir /s /q experiments\results_StrategyB_mini_lam0.25
rmdir /s /q experiments\results_StrategyB_mini_lam0.50
```

You do NOT need to re-run the injection after deleting the mini
folders — the aggregator will still prefer the full production
folders on the next run.

---

## Troubleshooting

### "My GPU ran for 2 hours and crashed. What do I lose?"

- Phases B0, B1, B2, B3 that finished are preserved (written to
  `progress_state.json` as they complete).
- The Mamba cache is preserved.
- Only the currently-running phase is lost.

Just re-run `run_lambda_sweep.bat` — it will resume.

### "The injection overwrote my manual edits to the R3.1 section."

It will **only** touch the block between the marker comments
(`% <<<LAMBDA_R31_BEGIN>>>` ... `% <<<LAMBDA_R31_END>>>`). If you
edited outside those markers, your edits are safe.

If you edited inside those markers, your edits are lost. To preserve
edits inside the marker region, add them to the template file
`scripts/ablation/templates/r3_1_subsection.tex.tmpl` instead.

### "After the full sweep, the paper still shows the mini-sweep numbers."

The aggregator prioritises `results_StrategyB_lam{X.XX}/` over
`results_StrategyB_mini_lam{X.XX}/`. If you still see mini numbers:

1. Check that the full-run folders exist:
   ```cmd
   dir experiments\results_StrategyB_lam*
   ```
2. Re-run the aggregator with verbose output:
   ```cmd
   python scripts\ablation\aggregate_lambda_results.py
   ```
   It will print which path it loaded for each cell.
3. Re-run the injector:
   ```cmd
   python scripts\ablation\inject_lambda_results.py
   ```

### "I want to add a 5th lambda value (e.g. 0.35)."

1. Edit `scripts/ablation/aggregate_lambda_results.py`:
   ```python
   LAMBDAS = ['0.00', '0.10', '0.25', '0.35', '0.50']
   ```
2. Edit `scripts/ablation/run_lambda_sweep.bat`:
   ```
   set LAMBDAS=0.00 0.10 0.35 0.50
   ```
3. Edit the LaTeX template to add a row to the table:
   `scripts/ablation/templates/r3_1_subsection.tex.tmpl`
4. Run the sweep, aggregator, and injector.

---

## TL;DR

```cmd
cd /d "D:\My Project\FL-RLMAMS_stage3"

REM 1. Sweep (~4.5 h on GPU)
scripts\ablation\run_lambda_sweep.bat

REM 2. Collect
python scripts\ablation\aggregate_lambda_results.py

REM 3. Inject into paper (~5 s)
python scripts\ablation\inject_lambda_results.py

REM 4. Recompile
cd paper4-Fuzzy-Augmented-Reward-Shaping
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

That's it. The paper will now contain publication-quality numbers
with zero manual editing of tex, figures, tables, or captions.
