import numpy as np
from scipy import stats


def mean_ci(x, conf=0.95):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, 0.0, (m, m)
    sd = x.std(ddof=1)
    h = stats.t.ppf((1 + conf) / 2, len(x) - 1) * sd / np.sqrt(len(x))
    return m, sd, (m - h, m + h)


def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = (np.mean(a) - np.mean(b)) / sp if sp > 0 else float("nan")
    return {"t": float(t), "p": float(p), "cohens_d": float(d)}
