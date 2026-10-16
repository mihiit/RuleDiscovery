#!/bin/sh
# Round 2 of training (resumable): key comparisons to 15 independent runs, remaining generalization regimes to 10.
cd "$(dirname "$0")/.." || exit 1
run() { d="data/models/macro_$2_$3_s$1.done"; [ -f "$d" ] && return
  python -u rl/train_macro.py --model $3 --dist $2 --seed $1 --episodes $4 >> data/models/macro_log.txt 2>&1 && touch "$d"; }
for s in 10 11 12 13 14; do for m in gru ff; do run $s main $m 12000; done; done
echo "R2_MAIN15 $(date +%T)" >> data/models/macro_status.txt
for s in 10 11 12 13 14; do for m in gru ff; do run $s D_rule_permutation $m 5000; done; done
echo "R2_D15 $(date +%T)" >> data/models/macro_status.txt
for s in 10 11 12 13 14; do run $s main gruB 12000; run $s D_rule_permutation gruB 5000; done
echo "R2_BELIEF15 $(date +%T)" >> data/models/macro_status.txt
for r in B_compositional E_novel_combination C_chain_length; do
  for s in 3 4 5 6 7 8 9; do for m in gru ff; do run $s $r $m 5000; done; done
  echo "R2_${r}10 $(date +%T)" >> data/models/macro_status.txt
done
