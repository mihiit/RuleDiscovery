#!/bin/sh

set -u

run() {
    seed="$1"
    model="$2"
    done_file="data/models/macro_D_rule_permutation_${model}_s${seed}.done"

    if [ -f "$done_file" ]; then
        echo "SKIP D ${model} s${seed} (done)"
        return
    fi

    echo "START D ${model} s${seed}"

    python -u rl/train_macro.py \
        --model "$model" \
        --dist D_rule_permutation \
        --seed "$seed" \
        --episodes 5000 \
        >> data/models/macro_D_seed15_29.log 2>&1

    if [ "$?" -eq 0 ]; then
        touch "$done_file"
        echo "DONE D ${model} s${seed}"
    else
        echo "FAILED D ${model} s${seed}"
        return 1
    fi
}

for seed in $(seq 15 29); do
    run "$seed" ff || exit 1
    run "$seed" gru || exit 1
    run "$seed" gruB || exit 1
done
