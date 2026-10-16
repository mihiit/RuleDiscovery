"""Contradiction-recovery: agents pre-trained on identity-mapped rules (key color == door color)
are tested on 'anti' mappings where that learned relation is always wrong."""
from common import *

C = ("red", "blue", "green", "yellow")
TRAIN = [dict(rules=(KD, KD, SB), colors=C, mapping="identity")]
TEST = [dict(rules=(KD, KD, SB), colors=C, mapping="anti")]
H = lambda **kw: (lambda s: HypothesisAgent(s, transfer=True, **kw))
CONDS = {
    "reactive": lambda s: ReactiveAgent(s),
    "hyp_no_revision": H(revision=False, label="hyp_no_revision"),
    "hyp_with_revision": H(label="hyp_with_revision"),
    "hyp_no_prior": lambda s: HypothesisAgent(s, label="hyp_no_prior"),
    "psrl": lambda s: PSRLAgent(s),
}


def main(seeds=20, envs=30, n_train=40):
    if os.environ.get('AGENT_SET') == 'rl':
        return
    rows, traces = [], {}
    for name, f in CONDS.items():
        for s in range(seeds):
            ag = f(s)
            if getattr(ag, "transfer", False):
                train_agent(ag, TRAIN, s, n_train)
            eps = eval_episodes(ag, TEST, TEST_BASE + 700_000, s, envs)
            rows.append(dict(experiment="recovery", agent=name, seed=s, **aggregate(eps)))
        print(name, round(np.mean([r["success_rate"] for r in rows if r["agent"] == name]), 3),
              "recovery", round(np.nanmean([r["recovery_rate"] for r in rows if r["agent"] == name]), 3))
    # trace episode: first held-out env where revision rescues a run that no-revision loses
    ar, an = CONDS["hyp_with_revision"](0), CONDS["hyp_no_revision"](0)
    for a_ in (ar, an):
        train_agent(a_, TRAIN, 0, n_train)
    for i in range(200):
        spec = generate(mk(TEST_BASE + 800_000 + i, TEST[0]))
        r1, r2 = run_episode(spec, ar, True), run_episode(spec, an, True)
        if r1["success"] and not r2["success"]:
            for nm, r in (("hyp_with_revision", r1), ("hyp_no_revision", r2)):
                traces[nm] = dict(trace=[(t, d) for t, d in r["trace"]], truth=r["truth_kd"], success=r["success"], steps=r["steps"], env_index=i)
            break
    ap = CONDS["hyp_no_prior"](0)
    r = run_episode(generate(mk(TEST_BASE + 800_000, TEST[0])), ap, True)
    traces["hyp_no_prior"] = dict(trace=[(t, d) for t, d in r["trace"]], truth=r["truth_kd"], success=r["success"], steps=r["steps"])
    save("recovery", rows, dict(seeds=seeds, envs=envs, n_train=n_train))
    json.dump(traces, open(os.path.join(OUT, "recovery_traces.json"), "w"))


if __name__ == "__main__":
    main()
