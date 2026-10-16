#!/bin/sh
# Generic-skill (go-to-object + INTERACT/PUSH) learned agents on the mixed-difficulty distribution: 2 seeds x {ff,gru}, 6k episodes.
# Waits for the macro-agent training to finish (single CPU core).
cd "$(dirname "$0")/.." || exit 1
while ! grep -q ALLDONE data/models/macro_status.txt 2>/dev/null; do sleep 20; done
for s in 0 1; do for m in gru ff; do
  d=data/models/generic_main_${m}_s$s.done; [ -f "$d" ] && continue
  python -u rl/train_generic.py --model $m --dist main --seed $s --episodes 6000 >> data/models/generic_log.txt 2>&1 && touch "$d"
done; done
echo "GENERIC_DONE $(date +%T)" >> data/models/macro_status.txt
