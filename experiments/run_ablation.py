"""Ablations (difficulty 4): memory, active exploration, revision, wide lambda sweep (also at d=5), mu sensitivity."""
from common import *

H = lambda **kw: (lambda s: HypothesisAgent(s, **kw))
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
ABL = {
    "1_memory": {"full_memory": H(label="full_memory"), "fixed_memory_3": H(memory_capacity=3, label="fixed_memory_3"),
                 "no_memory": H(memory_capacity=0, label="no_memory")},
    "2_active_exploration": {"info_driven": H(label="info_driven"), "random_explore": H(selector="random", label="random_explore")},
    "3_revision": {"revision_on": H(label="revision_on"), "revision_off": H(revision=False, label="revision_off")},
    "4_info_gain_weight": {f"lambda_{l}": H(lam=l, label=f"lambda_{l}") for l in LAMS},
    "5_action_cost": {f"mu_{m}": H(mu=m, label=f"mu_{m}") for m in (0.0, 0.05, 0.2)},
}


def main(seeds=20, envs=30, difficulty=4):
    if os.environ.get('AGENT_SET') == 'rl':
        return
    rows = []
    for abl, variants in ABL.items():
        for vname, f in variants.items():
            for s in range(seeds):
                eps = eval_episodes(f(s), [{}], EVAL_BASE + 500_000, s, envs, difficulty)
                rows.append(dict(experiment="ablation", condition=abl, agent=vname, difficulty=difficulty, seed=s, **aggregate(eps)))
            print(abl, vname, round(np.mean([r["success_rate"] for r in rows if r["agent"] == vname and r["condition"] == abl]), 3), flush=True)
    for l in LAMS:                      # same lambda sweep at difficulty 5 (more room for exploration to matter)
        for s in range(seeds):
            eps = eval_episodes(HypothesisAgent(s, lam=l), [{}], EVAL_BASE + 600_000, s, envs, 5)
            rows.append(dict(experiment="ablation", condition="4b_info_gain_weight_d5", agent=f"lambda_{l}_d5", difficulty=5, seed=s, **aggregate(eps)))
        print("d5 lambda", l, round(np.mean([r["success_rate"] for r in rows if r["agent"] == f"lambda_{l}_d5"]), 3), flush=True)
    for name, f in (("reactive", AGENTS["reactive"]), ("psrl", AGENTS["psrl"])):
        for s in range(seeds):
            eps = eval_episodes(f(s), [{}], EVAL_BASE + 500_000, s, envs, difficulty)
            rows.append(dict(experiment="ablation", condition="reference", agent=name, difficulty=difficulty, seed=s, **aggregate(eps)))
    save("ablation", rows, dict(seeds=seeds, envs=envs, difficulty=difficulty, lambdas=LAMS))


if __name__ == "__main__":
    main()
