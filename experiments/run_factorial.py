"""2x2(+) factorial ablation separating (a) how the reward term is formed (expected value under the posterior vs a sampled binding)
from (b) the information-gain bonus. Difficulties 4 and 5."""
from common import *

AG = {
    "hyp_EV_l0": lambda s: HypothesisAgent(s, lam=0.0, label="hyp_EV_l0"),
    "hyp_EV_l0.5": lambda s: HypothesisAgent(s, lam=0.5, label="hyp_EV_l0.5"),
    "hyp_EV_l8": lambda s: HypothesisAgent(s, lam=8.0, label="hyp_EV_l8"),
    "psrl_S_l0": lambda s: PSRLAgent(s, label="psrl_S_l0"),
    "psrl_S_l0.5": lambda s: PSRLAgent(s, lam=0.5, ig_from_posterior=True, label="psrl_S_l0.5"),
    "psrl_S_l8": lambda s: PSRLAgent(s, lam=8.0, ig_from_posterior=True, label="psrl_S_l8"),
}


def main(seeds=20, envs=30):
    if os.environ.get("AGENT_SET") == "rl":
        return
    rows = []
    for d in (4, 5):
        for name, f in AG.items():
            for s in range(seeds):
                eps = eval_episodes(f(s), [{}], EVAL_BASE + 950_000 + d * 10_000, s, envs, d)
                rows.append(dict(experiment="factorial", condition=f"d{d}", agent=name, difficulty=d, seed=s, **aggregate(eps)))
            print(d, name, round(np.mean([r["success_rate"] for r in rows if r["agent"] == name and r["difficulty"] == d]), 3), flush=True)
    save("factorial", rows, dict(seeds=seeds, envs=envs))


if __name__ == "__main__":
    main()
