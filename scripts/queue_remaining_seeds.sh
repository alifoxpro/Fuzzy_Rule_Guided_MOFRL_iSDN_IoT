#!/bin/bash
# Trimmed queue (Option 3): 3 seeds on CIC, 2 seeds on DIAD, no R2-4(d).
# Skipped: DIAD-s2026, R2-4(d) on both datasets.
# Total remaining: ~30 hours instead of ~55.

cd "$(dirname "$0")/.."

WAIT_MARKER="experiments/results_StrategyB_s123/ciciot2023/fuzzy_retrain/results.json"

echo "$(date) queue trimmed: waiting for CIC-s123 completion"
until [ -f "$WAIT_MARKER" ]; do sleep 60; done
echo "$(date) CIC-s123 done, launching DIAD-s123"

python scripts/training/train_rl_multiagent_StrategyB.py --dataset diad2024 --seed 123 --max-steps 2500 --suffix _s123 > logs/train_s123_diad.log 2>&1
echo "$(date) DIAD-s123 rc=$?"

python scripts/training/train_rl_multiagent_StrategyB.py --dataset ciciot2023 --seed 2026 --max-steps 2500 --suffix _s2026 > logs/train_s2026_cic.log 2>&1
echo "$(date) CIC-s2026 rc=$?"

echo "$(date) QUEUE COMPLETE (trimmed) — run aggregate_multiseed.py to build the final table."
