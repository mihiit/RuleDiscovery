import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.common import (
    AGENTS,
    TEST_BASE,
    budget,
    eval_episodes,
    rl_factory,
)

SEEDS = 30
ENVS = 30
REGIME = "D_rule_permutation"
OUT = "data/results/generalization_D_30.csv"

# Keep the existing non-learned baselines plus all three newly expanded
# learned D baselines.
AGENT_NAMES = (
    "random",
    "reactive",
    "psrl",
    "hypothesis",
    "hyp_adaptive",
    "hyp_transfer",
    "macro_ff",
    "macro_gru",
    "macro_gruB",
)

def main():
    rows = []

    # Import the regime definition exactly as the existing experiment does.
    from regimes import REGIMES
    reg = REGIMES[REGIME]

    for name in AGENT_NAMES:
        print(f"\n=== {name} ===", flush=True)

        f = AGENTS.get(name) or rl_factory(name, REGIME)
        if f is None:
            print(f"SKIP {name}: no factory", flush=True)
            continue

        ns, ne = budget(name, SEEDS, ENVS)
        print(f"seeds={ns} envs={ne}", flush=True)

        for s in range(ns):
            ag = f(s)

            if name == "hyp_transfer":
                from experiments.common import train_agent
                train_agent(ag, reg["train"], s, 40)

            eps = eval_episodes(
                ag,
                reg["test"],
                TEST_BASE,
                s,
                ne,
            )

            mean_success = sum(int(e["success"]) for e in eps) / len(eps)

            for i, success in enumerate(eps):
                rows.append({
                    "regime": REGIME,
                    "split": "test",
                    "agent": name,
                    "seed": s,
                    "env": i,
                    "success": int(success["success"]),
                    "seed_mean": mean_success,
                })

            print(
                f"{name} s{s:02d}: mean={mean_success:.3f}",
                flush=True,
            )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    fields = [
        "regime", "split", "agent",
        "seed", "env", "success", "seed_mean"
    ]

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT}", flush=True)

if __name__ == "__main__":
    main()
