#!/bin/sh
# Extends every principal learned agent to SEEDS_TOTAL independent training runs (default 10). Resumable (.done markers).
# Priority order: vault -> macro (mixed difficulty) -> macro regime D, B, E -> remaining regimes.
cd "$(dirname "$0")/.." || exit 1
mkdir -p data/models
N=${SEEDS_TOTAL:-10}
runm() { d="data/models/macro_$2_$3_s$1.done"; [ -f "$d" ] && return
  python -u rl/train_macro.py --model $3 --dist $2 --seed $1 --episodes $4 >> data/models/macro_log.txt 2>&1 && touch "$d"; }
runv() { d="data/models/vault_$2_s$1.done"; [ -f "$d" ] && return
  python -u rl/train_vault.py --model $2 --seed $1 --episodes 30000 --out data/models/vault_$2_s$1.pkl >> data/models/vault_log.txt 2>&1 && touch "$d"; }
for s in 0 1 2; do for m in ff gru; do [ -f data/models/vault_${m}_s$s.pkl ] && touch data/models/vault_${m}_s$s.done; done; done
s=3; while [ $s -lt $N ]; do for m in ff gru; do runv $s $m; done; s=$((s+1)); done
echo "VAULT10 $(date +%T)" >> data/models/macro_status.txt
s=3; while [ $s -lt $N ]; do for m in gru ff; do runm $s main $m 12000; done; s=$((s+1)); done
echo "MAIN10 $(date +%T)" >> data/models/macro_status.txt
for r in D_rule_permutation B_compositional E_novel_combination C_chain_length A_interpolation C2_size_only; do
  s=3; while [ $s -lt $N ]; do for m in gru ff; do runm $s $r $m 5000; done; s=$((s+1)); done
  echo "$r10 $(date +%T)" >> data/models/macro_status.txt
done
echo "ALL10 $(date +%T)" >> data/models/macro_status.txt
