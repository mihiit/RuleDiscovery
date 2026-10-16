#!/bin/sh
# Trains the PPO baselines sequentially on one CPU core (mixed-difficulty distribution only; per-regime RL training
# did not fit the compute budget). Logs/curves in data/models/.
cd "$(dirname "$0")/.." || exit 1
echo "start $(date +%T)" > data/models/train_status.txt
scripts/train_one.sh gru main 0 1200000
scripts/train_one.sh ff main 0 1200000
echo "ALLDONE $(date +%T)" >> data/models/train_status.txt
