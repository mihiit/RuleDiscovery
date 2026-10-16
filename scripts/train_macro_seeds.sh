#!/bin/sh
# Macro-level PPO baselines, 3 training seeds each, uniform budget (curves plateau within ~3k episodes per regime, ~6k for the mixed distribution):
# 12k episodes for the mixed-difficulty models, 5k per generalization regime. Resumable: finished models leave a .done marker.
cd "$(dirname "$0")/.." || exit 1
mkdir -p data/models
run() { # seed dist model episodes
  d="data/models/macro_$2_$3_s$1.done"; [ -f "$d" ] && return
  python -u rl/train_macro.py --model $3 --dist $2 --seed $1 --episodes $4 >> data/models/macro_log.txt 2>&1 && touch "$d"
}
for seed in ${SEEDS:-0 1 2}; do
  for m in gru ff; do run $seed main $m 12000; done
  for r in A_interpolation B_compositional C_chain_length C2_size_only D_rule_permutation E_novel_combination; do
    for m in gru ff; do run $seed $r $m 5000; done
  done
  echo "seed $seed done $(date +%T)" >> data/models/macro_status.txt
done
echo "ALLDONE $(date +%T)" >> data/models/macro_status.txt
