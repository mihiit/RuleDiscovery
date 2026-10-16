#!/bin/sh
# Vault PPO baselines: 3 seeds x {ff, gru}, trained ONLY on small codes (L in {4,5}, k in {2,3}), 30k episodes each (~1-2 min).
cd "$(dirname "$0")/.." || exit 1
mkdir -p data/models
for s in 0 1 2; do for m in ff gru; do
  python -u rl/train_vault.py --model $m --seed $s --episodes 30000 --out data/models/vault_${m}_s$s.pkl >> data/models/vault_log.txt 2>&1
done; done
echo "VAULT_DONE $(date +%T)" >> data/models/macro_status.txt
