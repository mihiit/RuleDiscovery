#!/bin/sh
# Final compute queue (resumable): regime D to 10 seeds, then Macro-GRU+belief with 5 seeds on mixed difficulty and on regime D.
cd "$(dirname "$0")/.." || exit 1
run() { d="data/models/macro_$2_$3_s$1.done"; [ -f "$d" ] && return
  python -u rl/train_macro.py --model $3 --dist $2 --seed $1 --episodes $4 >> data/models/macro_log.txt 2>&1 && touch "$d"; }
for s in 3 4 5 6 7 8 9; do for m in gru ff; do run $s D_rule_permutation $m 5000; done; done
echo "D10 $(date +%T)" >> data/models/macro_status.txt
for s in 0 1 2 3 4; do run $s main gruB 12000; done
echo "BELIEF_MAIN5 $(date +%T)" >> data/models/macro_status.txt
for s in 0 1 2 3 4; do run $s D_rule_permutation gruB 5000; done
echo "BELIEF_D5 $(date +%T)" >> data/models/macro_status.txt
