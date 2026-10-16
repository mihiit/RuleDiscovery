"""Stochastic feedback: a correct key fails to open its door with probability `slip` (a jammed lock, indistinguishable from a wrong key).
Learned agents were trained on the noise-free distribution, so for them this is a test-time shift, not noise-aware training.
hyp_robust is the hypothesis agent that needs three contradictions (not two) before rejecting a hypothesis."""
from common import *

SLIPS = (0.0, 0.2, 0.4)
NONRL = {"reactive": AGENTS["reactive"], "psrl": lambda s: PSRLAgent(s, label="psrl"), "hypothesis": lambda s: HypothesisAgent(s, label="hypothesis"),
         "hyp_adaptive": lambda s: HypothesisAgent(s, exclusive="adaptive", label="hyp_adaptive"),
         "hyp_robust": lambda s: HypothesisAgent(s, reject_after=3, label="hyp_robust")}


def main(seeds=30, envs=20):
    rows = []
    names = list(NONRL) + ["macro_ff", "macro_gru", "macro_gruB"]
    for name in names:
        if not keep(name):
            continue
        f = NONRL.get(name) or rl_factory(name, "main")
        if f is None:
            print("skip", name); continue
        for d in (3, 4):
            for slip in SLIPS:
                for s in range(seeds):
                    eps = eval_episodes_slip(f(s), [{}], EVAL_BASE + 700_000, s, envs, d, slip)
                    rows.append(dict(experiment="noise", condition=f"slip{slip}", agent=name, difficulty=d, seed=s, slip=slip, **aggregate(eps)))
                print(name, "d", d, "slip", slip, round(np.mean([r["success_rate"] for r in rows if r["agent"] == name and r["difficulty"] == d and r["slip"] == slip]), 3), flush=True)
    save("noise", rows, dict(seeds=seeds, envs=envs, slips=SLIPS))


if __name__ == "__main__":
    main()
