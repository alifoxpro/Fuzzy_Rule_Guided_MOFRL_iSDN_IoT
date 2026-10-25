#!/bin/bash
cd "$(dirname "$0")/.."
WAIT="experiments/results_StrategyB_s123/diad2024/fuzzy_retrain/results.json"
echo "$(date) waiting for DIAD-s123..."
until [ -f "$WAIT" ]; do sleep 60; done
echo "$(date) DIAD-s123 done, launching CIC-s2026"
python scripts/training/train_rl_multiagent_StrategyB.py --dataset ciciot2023 --seed 2026 --max-steps 2500 --suffix _s2026 > logs/train_s2026_cic.log 2>&1
echo "$(date) CIC-s2026 rc=$?"
echo "$(date) QUEUE COMPLETE"
