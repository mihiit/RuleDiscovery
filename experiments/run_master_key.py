"""Assumption violation: one key opens two doors (exclusivity false). Compares exclusive vs non-exclusive hypothesis/PSRL agents."""
from common import *

AG = {"reactive": AGENTS["reactive"], "psrl": lambda s: PSRLAgent(s, label="psrl"), "psrl_nonexcl": lambda s: PSRLAgent(s, exclusive=False, label="psrl_nonexcl"),
      "hypothesis": lambda s: HypothesisAgent(s, label="hypothesis"), "hyp_nonexcl": lambda s: HypothesisAgent(s, exclusive=False, label="hyp_nonexcl"),
      "hyp_adaptive": lambda s: HypothesisAgent(s, exclusive="adaptive", label="hyp_adaptive")}
SETS = {"master_key_d3": dict(rules=(KD, KD, SB), shared_keys=True), "master_key_d4": dict(rules=(KD, KD, SB, KD), shared_keys=True, grid_size=12),
        "exclusive_control_d3": dict(rules=(KD, KD, SB)), "exclusive_control_d4": dict(rules=(KD, KD, SB, KD), grid_size=12)}


def main(seeds=20, envs=30):
    if os.environ.get("AGENT_SET") == "rl":
        return
    rows = []
    for cname, tm in SETS.items():
        for name, f in AG.items():
            for s in range(seeds):
                eps = eval_episodes(f(s), [tm], EVAL_BASE + 900_000, s, envs)
                rows.append(dict(experiment="master_key", condition=cname, agent=name, seed=s, **aggregate(eps)))
            print(cname, name, round(np.mean([r["success_rate"] for r in rows if r["condition"] == cname and r["agent"] == name]), 3), flush=True)
    save("master_key", rows, dict(seeds=seeds, envs=envs))


if __name__ == "__main__":
    main()
