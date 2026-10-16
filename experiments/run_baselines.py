"""Main comparison at difficulty 3 (key/door + switch/bridge + push block)."""
from common import *


def main(seeds=20, envs=30):
    rows = []
    names = ["random", "reactive", "psrl", "hypothesis", "hyp_adaptive", "macro_ff", "macro_gru", "macro_gruB", "generic_ff", "generic_gru", "ppo_ff", "ppo_gru"]
    for name in names:
        if not keep(name):
            continue
        f = AGENTS.get(name) or rl_factory(name, "main")
        if f is None:
            print("skip", name, "(no trained model)"); continue
        ns, ne = budget(name, seeds, envs)
        for s in range(ns):
            eps = eval_episodes(f(s), [{}], EVAL_BASE, s, ne, difficulty=3)
            rows.append(dict(experiment="baselines", agent=name, difficulty=3, seed=s, **aggregate(eps)))
        print(name, round(np.mean([r["success_rate"] for r in rows if r["agent"] == name]), 3), flush=True)
    save("baselines", rows, dict(seeds=seeds, envs=envs, rl_seeds=RL_SEEDS, rl_envs=RL_ENVS, max_steps=200))


if __name__ == "__main__":
    main()
