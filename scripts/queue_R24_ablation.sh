#!/bin/bash
# R2-4 minimum viable ablation:
# Cell (b) DIAD → Cell (d) CIC → Cell (d) DIAD
# Cell (b) CIC already running.

cd "$(dirname "$0")/.."
mkdir -p logs

W1="experiments/results_StrategyB_R24_b/ciciot2023/fuzzy_retrain/results.json"
echo "$(date) queue_R24: waiting for cell(b) CIC → $W1"
until [ -f "$W1" ]; do sleep 60; done
echo "$(date) cell(b) CIC done — launching cell(b) DIAD"

python scripts/training/train_rl_multiagent_StrategyB.py --dataset diad2024 --lambda-fuzzy 0.0 --mu-link 0.0 --max-steps 2500 --seed 42 --suffix _R24_b > logs/train_R24_b_diad.log 2>&1
echo "$(date) cell(b) DIAD rc=$?"

python scripts/training/train_rl_multiagent_StrategyB.py --dataset ciciot2023 --lambda-fuzzy 0.25 --mu-link 0.0 --max-steps 2500 --seed 42 --suffix _R24_d > logs/train_R24_d_cic.log 2>&1
echo "$(date) cell(d) CIC rc=$?"

python scripts/training/train_rl_multiagent_StrategyB.py --dataset diad2024 --lambda-fuzzy 0.25 --mu-link 0.0 --max-steps 2500 --seed 42 --suffix _R24_d > logs/train_R24_d_diad.log 2>&1
echo "$(date) cell(d) DIAD rc=$?"

echo "$(date) R2-4 QUEUE COMPLETE — 4 cells ready for aggregation."
