import os, sys, json, time, csv
try:
    import resource
except ImportError:
    resource = None
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from environment.generator import EnvironmentConfig, generate
from environment.rules import KEY_DOOR as KD, SWITCH_BRIDGE as SB, PUSH_BLOCK as PB
from agents.random_agent import RandomAgent
from agents.reactive_agent import ReactiveAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.psrl_agent import PSRLAgent
from agents.ppo_agent import PPOAgent
from evaluation.metrics import run_episode

from regimes import TRAIN_BASE, EVAL_BASE, TEST_BASE, mk, REGIMES
import glob
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results")

AGENTS = {
    "random": lambda s: RandomAgent(s),
    "reactive": lambda s: ReactiveAgent(s),
    "hypothesis": lambda s: HypothesisAgent(s, label="hypothesis"),
    "hyp_transfer": lambda s: HypothesisAgent(s, transfer=True, label="hyp_transfer"),
    "psrl": lambda s: PSRLAgent(s),
    "hyp_adaptive": lambda s: HypothesisAgent(s, exclusive="adaptive", label="hyp_adaptive"),
}
RL_SEEDS, RL_ENVS = 15, 20          # RL baselines are evaluated on fewer (seed, env) pairs: inference is ~10x slower


def finished(paths):
    """Only models whose training finished (a .done marker next to the .pkl) are evaluated; checkpoints of runs in progress are skipped."""
    return sorted(p for p in paths if os.path.exists(p[:-4] + ".done"))


def model_paths(dist, model):
    return sorted(p for p in glob.glob(os.path.join(os.path.dirname(OUT), "models", f"{dist}_{model}_s[0-9]*.pkl")))


def ppo_factory(dist, model):
    paths = model_paths(dist, model)
    if not paths:
        return None
    return lambda s: PPOAgent(s, model_path=paths[s % len(paths)], label=f"ppo_{model}")


def macro_factory(dist, model):
    paths = finished(glob.glob(os.path.join(os.path.dirname(OUT), "models", f"macro_{dist}_{model}_s[0-9]*.pkl")))
    if not paths:
        return None
    from agents.macro_ppo_agent import MacroPPOAgent
    return lambda s: MacroPPOAgent(s, model_path=paths[s % len(paths)], label=f"macro_{model}")


def generic_factory(dist, model):
    paths = finished(glob.glob(os.path.join(os.path.dirname(OUT), "models", f"generic_{dist}_{model}_s[0-9]*.pkl")))
    if not paths:
        return None
    from agents.generic_ppo_agent import GenericPPOAgent
    return lambda s: GenericPPOAgent(s, model_path=paths[s % len(paths)], label=f"generic_{model}")


def rl_factory(name, dist):
    kind, model = name.split("_")
    return {"ppo": ppo_factory, "macro": macro_factory, "generic": generic_factory}[kind](dist, model)


def budget(name, seeds, envs):
    return (min(seeds, RL_SEEDS), min(envs, RL_ENVS)) if name.startswith("ppo") else (seeds, envs)


def eval_episodes(agent, tmpls, base, seed, n, difficulty=3):
    return [run_episode(generate(mk(base + seed * 1000 + i, tmpls[i % len(tmpls)], difficulty)), agent) for i in range(n)]


def eval_episodes_slip(agent, tmpls, base, seed, n, difficulty, slip):
    import dataclasses
    return [run_episode(dataclasses.replace(generate(mk(base + seed * 1000 + i, tmpls[i % len(tmpls)], difficulty)), slip=slip), agent) for i in range(n)]


def train_agent(agent, tmpls, seed, n, difficulty=3):
    agent.learn = True
    for i in range(n):
        run_episode(generate(mk(TRAIN_BASE + seed * 1000 + i, tmpls[i % len(tmpls)], difficulty)), agent)
    agent.learn = False


def aggregate(eps):
    n = len(eps)
    succ = [e for e in eps if e["success"]]
    tot_steps = sum(e["steps"] for e in eps)
    r = {"n_episodes": n, "success_rate": len(succ) / n,
         "mean_steps_success": float(np.mean([e["steps"] for e in succ])) if succ else float("nan"),
         "mean_episode_len": tot_steps / n,
         "sample_efficiency": len(succ) / max(1, tot_steps),
         "cpu_time_per_episode": float(np.mean([e["cpu_time"] for e in eps])),
         "actions_per_sec": tot_steps / max(1e-9, sum(e["cpu_time"] for e in eps)),
         "peak_rss_mb": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024) if resource else None}
    r["failed_door_tests"] = float(np.mean([e.get("n_locked", 0) for e in eps]))
    fe, rc = sum(e["fail_events"] for e in eps), sum(e["recovered"] for e in eps)
    r["fail_events"], r["recovery_rate"] = fe / n, (rc / fe if fe else float("nan"))
    if "generated" in eps[0]:
        for k in ("generated", "rejected", "retained", "contradictions", "tests", "redundant"):
            r["hyp_" + k] = float(np.mean([e[k] for e in eps]))
        ex, xp = sum(e["explore_macros"] for e in eps), sum(e["exploit_macros"] for e in eps)
        r["explore_ratio"] = ex / max(1, ex + xp)
        ht = sum(e["hyp_tested"] for e in eps)
        r["hyp_accuracy"] = sum(e["hyp_correct"] for e in eps) / ht if ht else float("nan")
    return r


def keep(name):
    """AGENT_SET=nonrl|rl|all selects which agents an experiment script evaluates (RL models train in the background)."""
    only = os.environ.get("ONLY")
    if only and name not in only.split(","):
        return False
    f = os.environ.get("AGENT_SET", "all")
    return f == "all" or (f == "rl") == name.startswith(("ppo", "macro", "generic"))


def save(name, rows, meta):
    os.makedirs(OUT, exist_ok=True)
    if not rows and os.environ.get("AGENT_SET", "all") != "all":
        return   # an empty partial run must never overwrite existing results
    path = os.path.join(OUT, name + ".csv")
    if os.environ.get("AGENT_SET", "all") != "all" and os.path.exists(path) and os.path.getsize(path) > 10 and rows:
        import pandas as pd
        old = pd.read_csv(path)
        new = pd.DataFrame(rows)
        old = old[~old.agent.isin(set(new.agent))]
        rows = old.to_dict("records") + rows
    keys = sorted({k for r in rows for k in r})
    first = [k for k in ("experiment", "condition", "agent", "split", "difficulty", "seed") if k in keys]
    keys = first + [k for k in keys if k not in first]
    with open(os.path.join(OUT, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    meta = dict(meta, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), n_rows=len(rows))
    json.dump(meta, open(os.path.join(OUT, name + ".json"), "w"), indent=2, default=str)
