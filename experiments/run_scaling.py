"""Difficulty scaling 1..5."""
from common import *


def main(seeds=20, envs=30):
    rows = []
    for name in ("random", "reactive", "psrl", "hypothesis", "hyp_adaptive", "macro_ff", "macro_gru", "macro_gruB", "generic_ff", "generic_gru", "ppo_ff", "ppo_gru"):
        if not keep(name):
            continue
        f = AGENTS.get(name) or rl_factory(name, "main")
        if f is None:
            print("skip", name); continue
        ns, ne = budget(name, seeds, envs)
        for d in range(1, 6):
            for s in range(ns):
                eps = eval_episodes(f(s), [{}], EVAL_BASE + d * 100_000, s, ne, difficulty=d)
                rows.append(dict(experiment="scaling", agent=name, difficulty=d, seed=s, **aggregate(eps)))
            print(d, name, round(np.mean([r["success_rate"] for r in rows if r["agent"] == name and r["difficulty"] == d]), 3), flush=True)
    save("scaling", rows, dict(seeds=seeds, envs=envs, rl_seeds=RL_SEEDS, rl_envs=RL_ENVS, max_steps=200))


if __name__ == "__main__":
    main()
