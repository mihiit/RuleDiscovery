"""Statistics that respect the structure of the experiments.

Two sources of variance exist and must not be confused:

* *evaluation variance* -- which environments an agent is run on (every agent);
* *training variance* -- which weights a learned agent ended up with (learned agents only).

Evaluating one trained network on hundreds of environments says nothing about how a *training procedure* behaves, so for a learned agent
the unit of analysis is the **trained model**: its score is the mean over all evaluation seeds assigned to it (evaluation seed ``s`` uses
trained model ``s mod m``). For agents without trained weights the unit is the evaluation seed. All intervals and tests below are
computed over these units, so a learned agent with ``m`` models contributes ``m`` observations, not 900.
"""
import glob
import os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEARNED_PREFIX = ("macro_", "generic_", "ppo_", "vault_")


def is_learned(agent):
    return agent.startswith(LEARNED_PREFIX)


def n_models(agent, dist="main"):
    """Number of finished trained models behind a learned agent (0 for agents without trained weights)."""
    if not is_learned(agent):
        return 0
    kind, model = agent.split("_", 1)
    pat = {"macro": f"macro_{dist}_{model}_s[0-9]*.pkl", "generic": f"generic_{dist}_{model}_s[0-9]*.pkl",
           "ppo": f"{dist}_{model}_s[0-9]*.pkl", "vault": f"vault_{model}_s[0-9]*.pkl"}[kind]
    paths = glob.glob(os.path.join(ROOT, "data", "models", pat))
    if kind != "ppo":
        paths = [p for p in paths if os.path.exists(p[:-4] + ".done")]
    return len(paths)


def units(df, agent, dist="main", col="success_rate"):
    """Per-unit scores of `agent` in the (already filtered) frame `df`: per trained model if learned, else per evaluation seed."""
    d = df[df.agent == agent]
    if d.empty:
        return np.array([])
    if is_learned(agent):
        m = max(1, n_models(agent, dist))
        d = d.assign(unit=d.seed % m)
        return d.groupby("unit")[col].mean().to_numpy(float)
    return d.groupby("seed")[col].mean().to_numpy(float)


def boot_mean_ci(x, n=10000, seed=0, conf=0.95):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"),) * 2
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return tuple(np.percentile(m, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100]))


def compare_units(a, b, n=10000, seed=0):
    """Difference of unit means (a - b) with a two-sample percentile bootstrap CI (units resampled independently inside each agent),
    Welch's t-test and Cohen's d computed on the unit scores."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = dict(n_a=len(a), n_b=len(b), mean_a=float(a.mean()) if len(a) else float("nan"), mean_b=float(b.mean()) if len(b) else float("nan"))
    if len(a) < 2 or len(b) < 2:
        return dict(out, diff=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), p=float("nan"), d=float("nan"))
    rng = np.random.default_rng(seed)
    diffs = a[rng.integers(0, len(a), (n, len(a)))].mean(1) - b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va == 0 and vb == 0:                      # both constant: a test is meaningless; treat equal constants as p = 1
        p = 1.0 if a.mean() == b.mean() else 0.0
        d = 0.0 if a.mean() == b.mean() else float("inf")
    else:
        p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
        d = float((a.mean() - b.mean()) / sp)
    return dict(out, diff=float(a.mean() - b.mean()), ci_lo=float(lo), ci_hi=float(hi), p=p, d=d)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()
