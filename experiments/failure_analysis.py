"""Failure analysis: step-budget sweep + rule-based categorisation of failed episodes."""
from common import *
from evaluation.metrics import run_episode
import json


def categorise(spec, agent, res):
    h = spec.hidden
    G = len([1 for o in spec.objects.values() if o.kind in ("door", "chasm", "block")])
    log, end = agent.macro_log, res["steps"]
    dist_steps = 0
    for i, (t0, typ, tag) in enumerate(log):
        t1 = log[i + 1][0] if i + 1 < len(log) else end
        if (typ in ("PICK", "TEST") and tag is not None and tag not in h.key_door) or \
           (typ == "SWITCH" and tag not in h.switch_bridge):
            dist_steps += t1 - t0
    repeated_refuted = res.get("redundant", 0) > 0
    frac = dist_steps / max(1, end)
    if repeated_refuted:
        return "repeated refuted hypothesis"
    if frac >= 0.25:
        return "distractor interference"
    if G >= 4:
        return "long-horizon dependency"
    return "insufficient exploration (budget)"


def main(seeds=10, envs=30):
    if os.environ.get('AGENT_SET') == 'rl':
        return
    out = {"budget_sweep": [], "categories": {}}
    for d in (4, 5):
        for ms in (100, 200, 400, 800):
            srs = []
            for s in range(seeds):
                eps = [run_episode(generate(EnvironmentConfig(seed=EVAL_BASE + 900_000 + s * 1000 + i, difficulty=d, max_steps=ms)), HypothesisAgent(s)) for i in range(envs)]
                srs.append(np.mean([e["success"] for e in eps]))
            out["budget_sweep"].append(dict(difficulty=d, max_steps=ms, success_rate=float(np.mean(srs)), sd=float(np.std(srs, ddof=1))))
            print("budget", d, ms, round(np.mean(srs), 3))
    cats, n_fail, n_tot, examples = {}, 0, 0, []
    for s in range(seeds):
        ag = HypothesisAgent(s)
        for i in range(envs):
            spec = generate(EnvironmentConfig(seed=EVAL_BASE + 900_000 + s * 1000 + i, difficulty=5, max_steps=200))
            r = run_episode(spec, ag); n_tot += 1
            if not r["success"]:
                n_fail += 1
                c = categorise(spec, ag, r)
                cats[c] = cats.get(c, 0) + 1
                if len(examples) < 3 and c not in [e["category"] for e in examples]:
                    examples.append(dict(env=spec.environment_id, category=c, steps=r["steps"], tests=r["tests"], redundant=r["redundant"]))
    out["categories"] = dict(counts=cats, n_failed=n_fail, n_episodes=n_tot, examples=examples)
    print(out["categories"])
    json.dump(out, open(os.path.join(OUT, "failure_analysis.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
