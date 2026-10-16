#!/bin/sh
# Brings Macro-GRU+belief to 10 independent training runs (mixed difficulty and regime D). Resumable.
cd "$(dirname "$0")/.." || exit 1
run() { d="data/models/macro_$2_gruB_s$1.done"; [ -f "$d" ] && return
  python -u rl/train_macro.py --model gruB --dist $2 --seed $1 --episodes $3 >> data/models/macro_log.txt 2>&1 && touch "$d"; }
for s in 5 6 7 8 9; do run $s main 12000; done
for s in 5 6 7 8 9; do run $s D_rule_permutation 5000; done
echo "BELIEF10 $(date +%T)" >> data/models/macro_status.txt
