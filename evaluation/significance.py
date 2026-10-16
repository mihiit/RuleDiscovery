import numpy as np
from scipy import stats


def bootstrap_ci(x, n=5000, seed=0, conf=0.95):
    x = np.asarray([v for v in x if not np.isnan(v)], float)
    if len(x) == 0:
        return (float("nan"),) * 2
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100]))


def compare(a, b):
    """Welch t-test, Mann-Whitney U, Cohen's d between two per-seed samples (a - b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return dict(diff=float("nan"), welch_p=float("nan"), mwu_p=float("nan"), cohens_d=float("nan"))
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (a.mean() - b.mean()) / sp if sp > 0 else (0.0 if a.mean() == b.mean() else float("inf"))
    try:
        mw = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    except ValueError:
        mw = 1.0
    return dict(diff=float(a.mean() - b.mean()), welch_p=float(stats.ttest_ind(a, b, equal_var=False).pvalue),
                mwu_p=float(mw), cohens_d=float(d))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()
