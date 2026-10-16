"""Empirical check of Proposition 1. Single key-door gate; n = number of keys REACHABLE from the start (distractor keys behind the
gate cannot be tried). Elimination: E[failed tests] = (n-1)/2.  Stateless lose-shift (never retries the key in hand): (n-1)^2/n.
Predictions are evaluated per episode with the episode's own n and averaged."""
from common import *
from environment.world import DIRS


def reachable_keys(spec):
    n, start, objs = spec.grid_size, spec.agent_start, spec.objects
    seen, st = {start}, [start]
    while st:
        r, c = st.pop()
        for dr, dc in DIRS.values():
            q = (r + dr, c + dc)
            if q in seen or not (0 <= q[0] < n and 0 <= q[1] < n) or q in spec.walls:
                continue
            o = objs.get(q)
            if o is not None and o.kind in ("door", "chasm", "block", "trap"):
                continue
            seen.add(q); st.append(q)
    return sum(1 for p, o in objs.items() if o.kind == "key" and p in seen)


def main(seeds=20, envs=60):
    if os.environ.get("AGENT_SET") == "rl":
        return
    rows = []
    for d in (3, 4, 5):
        for name in ("reactive", "hypothesis", "psrl"):
            for s in range(seeds):
                f, ag = [], AGENTS[name](s)
                for i in range(envs):
                    spec = generate(mk(EVAL_BASE + 970_000 + d * 10_000 + s * 1000 + i, dict(rules=(KD,)), d))
                    n = reachable_keys(spec)
                    r = run_episode(spec, ag)
                    f.append((n, r["n_locked"], (n - 1) / 2, (n - 1) ** 2 / n))
                a = np.array(f, float)
                rows.append(dict(experiment="theory", condition=f"d{d}", agent=name, difficulty=d, seed=s, mean_n=a[:, 0].mean(),
                                 failed_door_tests=a[:, 1].mean(), pred_elim=a[:, 2].mean(), pred_lose_shift=a[:, 3].mean()))
            g = [r for r in rows if r["agent"] == name and r["difficulty"] == d]
            print(f"d{d} {name}: n={np.mean([r['mean_n'] for r in g]):.2f} observed {np.mean([r['failed_door_tests'] for r in g]):.3f} "
                  f"elim-pred {np.mean([r['pred_elim'] for r in g]):.3f} lose-shift-pred {np.mean([r['pred_lose_shift'] for r in g]):.3f}", flush=True)
    save("theory", rows, dict(seeds=seeds, envs=envs))


if __name__ == "__main__":
    main()
