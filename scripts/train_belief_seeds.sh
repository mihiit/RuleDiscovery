#!/bin/sh
# Matched-information learned baseline (Macro-GRU+belief): same macro vocabulary AND the hypothesis tracker's expected value /
# information gain as per-candidate features. 10 training seeds on the mixed-difficulty distribution and on regime D.
cd "$(dirname "$0")/.." || exit 1
while ! grep -q MAIN10 data/models/macro_status.txt 2>/dev/null; do sleep 30; done
for dist in main D_rule_permutation; do
  ep=12000; [ "$dist" != main ] && ep=5000
  for s in 0 1 2 3 4 5 6 7 8 9; do
    d=data/models/macro_${dist}_gruB_s$s.done; [ -f "$d" ] && continue
    python -u rl/train_macro.py --model gruB --dist $dist --seed $s --episodes $ep >> data/models/macro_log.txt 2>&1 && touch "$d"
  done
done
echo "BELIEF10 $(date +%T)" >> data/models/macro_status.txt
