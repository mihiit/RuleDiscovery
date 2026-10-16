#!/bin/sh
# usage: run_stages.sh [nonrl|rl|all] -- runs experiment stages sequentially; progress in data/results/run_$1.log
cd "$(dirname "$0")/.." || exit 1
export AGENT_SET="${1:-all}"
STG="${STAGES:-run_baselines run_scaling run_generalization run_ablation run_factorial run_master_key run_theory run_rule_types run_recovery failure_analysis}"
[ "$AGENT_SET" = "rl" ] && [ -z "$STAGES" ] && STG="run_baselines run_scaling run_generalization"
for m in $STG; do
  echo "== $m $(date +%T)" >> data/results/run_$AGENT_SET.log
  (cd experiments && python -u -c "import $m; $m.main(seeds=30, envs=30)") >> data/results/run_$AGENT_SET.log 2>&1
done
echo ALLDONE >> data/results/run_$AGENT_SET.log
