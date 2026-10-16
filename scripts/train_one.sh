#!/bin/sh
# usage: train_one.sh MODEL DIST SEED STEPS
cd "$(dirname "$0")/.." || exit 1
python -u rl/train_ppo.py --model "$1" --dist "$2" --seed "$3" --steps "$4" >> "data/models/log_$2_$1_s$3.txt" 2>&1
