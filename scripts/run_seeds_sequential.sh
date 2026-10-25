#!/bin/bash
# Sequential runner: multi-seed replication for R1-3 / R2-5
# Runs seeds 123, 2026 on both datasets (seed 42 is the canonical baseline)
# Each run ~90-120 min. Total: ~8 hours.

cd "$(dirname "$0")/.."
mkdir -p logs

for SEED in 123 2026; do
  for DS in ciciot2023 diad2024; do
    LOG="logs/train_s${SEED}_${DS}.log"
    echo "$(date) starting seed=$SEED dataset=$DS -> $LOG"
    python scripts/training/train_rl_multiagent_StrategyB.py \
      --dataset $DS --seed $SEED --suffix _s${SEED} \
      > "$LOG" 2>&1
    RC=$?
    echo "$(date) done seed=$SEED dataset=$DS rc=$RC"
  done
done

echo "$(date) ALL SEEDS COMPLETE"
