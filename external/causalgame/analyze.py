"""Statistics and table for the CausalGame experiment. Unit of analysis for pooled comparisons: the scenario (mean over its sessions), so the 14
scenarios -- not the thousands of sessions -- set the sample size. Intervals: percentile bootstrap over scenarios; tests: Wilcoxon signed-rank
(paired by scenario); Holm correction over the pre-declared family below."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from evaluation.hierarchical import holm, boot_mean_ci

df = pd.read_csv(os.path.join(ROOT, "data", "results", "causalgame.csv"))
T = os.path.join(ROOT, "paper", "tables"); os.makedirs(T, exist_ok=True)
LAB = {"default": "Default design", "random": "Random search", "hillclimb": "Hill-climbing (reactive)", "psrl": "Posterior sampling", "hyp_ev": "Hypothesis (EV + info gain)", "hyp_ev_noig": "Hypothesis, no info bonus"}
ORDER = ["default", "random", "hillclimb", "psrl", "hyp_ev_noig", "hyp_ev"]
sc = df.groupby(["scenario", "agent"]).agg(survival=("survival", "mean"), win=("win", "mean"), n=("session", "count")).reset_index()
piv = sc.pivot(index="scenario", columns="agent", values="survival")
pw = sc.pivot(index="scenario", columns="agent", values="win")
rows = []
for a in ORDER:
    lo, hi = boot_mean_ci(piv[a].values); wl, wh = boot_mean_ci(pw[a].values)
    rows.append(dict(agent=LAB[a], survival=piv[a].mean(), s_lo=lo, s_hi=hi, win=pw[a].mean(), w_lo=wl, w_hi=wh, best_in=int((piv.idxmax(axis=1) == a).sum())))
S = pd.DataFrame(rows); S.round(4).to_csv(os.path.join(T, "causalgame_agents.csv"), index=False)
FAM = [("hyp_ev", "psrl"), ("hyp_ev", "hillclimb"), ("hyp_ev", "random"), ("hyp_ev", "default"), ("hyp_ev", "hyp_ev_noig")]
comp = []
for a, b in FAM:
    d = (piv[a] - piv[b]).values
    lo, hi = boot_mean_ci(d)
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    comp.append(dict(comparison=f"{LAB[a]} vs {LAB[b]}", diff=d.mean(), lo=lo, hi=hi, p=p, wins=int((d > 0).sum()), n=len(d)))
C = pd.DataFrame(comp); C["p_holm"] = holm(C.p.tolist()); C.round(4).to_csv(os.path.join(T, "causalgame_comparisons.csv"), index=False)
with open(os.path.join(T, "causalgame_agents.tex"), "w") as f:
    f.write("\\begin{tabular}{lccc}\n\\toprule\nAgent & Survival & Success rate & Best in \\\\\n\\midrule\n")
    for _, r in S.iterrows():
        f.write(f"{r.agent} & {r.survival:.3f} [{r.s_lo:.3f}, {r.s_hi:.3f}] & {r.win:.3f} [{r.w_lo:.3f}, {r.w_hi:.3f}] & {r.best_in} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
with open(os.path.join(T, "causalgame_comparisons.tex"), "w") as f:
    f.write("\\begin{tabular}{lcccc}\n\\toprule\nComparison & $\\Delta$ survival & 95\\% CI & scenarios won & $p_{\\text{Holm}}$ \\\\\n\\midrule\n")
    for _, r in C.iterrows():
        p = "$<$0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}"
        f.write(f"{r.comparison} & {r['diff']:+.3f} & [{r.lo:+.3f}, {r.hi:+.3f}] & {r.wins}/{r.n} & {p} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(S.round(3).to_string()); print(C.round(4).to_string())
print(piv.round(2).to_string())
