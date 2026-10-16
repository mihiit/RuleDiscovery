"""Rule-TYPE discovery: some doors are opened by a SWITCH, not a key. An agent whose hypothesis vocabulary lacks that type
cannot solve them; an open-vocabulary agent must decide, per door, which rule type applies."""
from common import *
from regimes import SD

H = lambda **kw: (lambda s: HypothesisAgent(s, **kw))
SETS = {"typed_SD_d3": dict(rules=(KD, SD, SB)), "typed_SD_d4": dict(rules=(SD, SD, KD, SB), grid_size=12), "standard_d3": dict(rules=(KD, SB, PB))}
AG = {"reactive": AGENTS["reactive"], "psrl": AGENTS["psrl"], "psrl_open": lambda s: PSRLAgent(s, use_sd=True, label="psrl_open"),
      "hyp_closed_vocab": H(label="hyp_closed_vocab"), "hyp_open_vocab": H(use_sd=True, label="hyp_open_vocab")}


def main(seeds=20, envs=30):
    if os.environ.get('AGENT_SET') == 'rl':
        return
    rows = []
    for cname, tm in SETS.items():
        for name, f in AG.items():
            for s in range(seeds):
                eps = eval_episodes(f(s), [tm], EVAL_BASE + 800_000, s, envs)
                rows.append(dict(experiment="rule_types", condition=cname, agent=name, seed=s, **aggregate(eps)))
            print(cname, name, round(np.mean([r["success_rate"] for r in rows if r["condition"] == cname and r["agent"] == name]), 3), flush=True)
    save("rule_types", rows, dict(seeds=seeds, envs=envs))


if __name__ == "__main__":
    main()
