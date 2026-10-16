"""Primary statistical analysis (training-run aware).

Writes, to paper/tables/:
  primary_comparisons.{csv,tex}  every pre-declared pairwise claim of the paper: absolute difference, 95% CI, Holm-adjusted p, Cohen's d
  learned_spread.{csv,tex}       per learned agent: number of independently trained models, mean, SD, min, max, bootstrap CI over models
  gaps.csv                       train-minus-test gaps (regimes) with unit-level CIs
Holm's correction is applied once over the primary family declared in PRIMARY below (nothing is added after looking at results).
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
from evaluation.hierarchical import units, compare_units, holm, boot_mean_ci, n_models, is_learned

R, T = "data/results", "paper/tables"
rd = lambda n: pd.read_csv(f"{R}/{n}.csv")
sc, gen, mk, vt, nz = rd("scaling"), rd("generalization"), rd("master_key"), rd("vault"), rd("noise")
os.makedirs(T, exist_ok=True)

LABEL = {"hypothesis": "Hypothesis", "hyp_adaptive": "Hyp.-adaptive", "psrl": "PSRL", "reactive": "Reactive", "macro_ff": "Macro-FF",
         "macro_gru": "Macro-GRU", "macro_gruB": "Macro-GRU+belief", "generic_ff": "Generic-FF", "generic_gru": "Generic-GRU",
         "hyp_nonexcl": "Hyp.-nonexcl.", "vault_gru": "PPO-GRU", "vault_ff": "PPO-FF", "hyp_transfer": "Hyp.+transfer", "hyp_robust": "Hyp.-robust"}


def cell(df, agent, dist="main", col="success_rate", **flt):
    d = df
    for k, v in flt.items():
        d = d[d[k] == v]
    return units(d, agent, dist, col)


PRIMARY = []   # (family, label, a_units, b_units)


def add(fam, label, ua, ub):
    if len(ua) and len(ub):
        PRIMARY.append((fam, label, ua, ub))


# Generic-skill agents (2 trained models) are reported descriptively only: two units cannot support inference.
# 1. difficulty scaling: hypothesis agent against every other principal agent
for d in (3, 4, 5):
    h = cell(sc, "hypothesis", difficulty=d)
    for other in ("reactive", "psrl", "macro_gru", "macro_ff", "macro_gruB", "hyp_adaptive"):
        add("scaling", f"d={d}: Hypothesis vs {LABEL[other]}", h, cell(sc, other, difficulty=d))
    add("scaling", f"d={d}: Macro-GRU+belief vs Macro-GRU", cell(sc, "macro_gruB", difficulty=d), cell(sc, "macro_gru", difficulty=d))
    add("scaling", f"d={d}: PSRL vs Macro-GRU", cell(sc, "psrl", difficulty=d), cell(sc, "macro_gru", difficulty=d))
# 2. unseen colour symbols (regime D, held-out test split)
dg = gen[(gen.condition == "D_rule_permutation") & (gen.split == "test")]
add("regimeD", "D: Macro-GRU+belief vs Macro-GRU", cell(dg, "macro_gruB", "D_rule_permutation"), cell(dg, "macro_gru", "D_rule_permutation"))
for other in ("macro_gru", "macro_ff", "macro_gruB", "psrl", "reactive"):
    add("regimeD", f"D: Hypothesis vs {LABEL[other]}", cell(dg, "hypothesis", "D_rule_permutation"), cell(dg, other, "D_rule_permutation"))
# 3. structural assumption (master key)
for c in ("master_key_d3", "master_key_d4", "exclusive_control_d3", "exclusive_control_d4"):
    m = mk[mk.condition == c]
    for a, b in (("hyp_adaptive", "hypothesis"), ("hyp_adaptive", "psrl"), ("hyp_adaptive", "hyp_nonexcl")):
        add("assumption", f"{c}: {LABEL[a]} vs {LABEL[b]}", cell(m, a), cell(m, b))
# 4. vault extrapolation to larger codes
vl = vt[vt.condition == "larger"]
for other in ("vault_gru", "vault_ff", "reactive"):
    add("vault", f"vault larger codes: Hypothesis vs {LABEL.get(other, other)}", cell(vl, "hypothesis"), cell(vl, other))

# 5. noisy feedback (jammed locks); added in the second analysis round, before the full noise results were inspected
for d in (3, 4):
    for sl in (0.2, 0.4):
        m = nz[(nz.difficulty == d) & (nz.slip == sl)]
        for a, b in (("hypothesis", "psrl"), ("hyp_robust", "hypothesis"), ("hyp_adaptive", "hypothesis"), ("macro_gruB", "hypothesis"), ("macro_gru", "hypothesis")):
            add("noise", f"slip {sl}, d={d}: {LABEL[a]} vs {LABEL[b]}", cell(m, a), cell(m, b))

rows = []
for fam, label, ua, ub in PRIMARY:
    rows.append(dict(family=fam, comparison=label, **compare_units(ua, ub)))
P = pd.DataFrame(rows)
P["p_holm"] = holm(P.p.fillna(1.0))
P.round(4).to_csv(f"{T}/primary_comparisons.csv", index=False)


def tex_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


GROUPS = {"primary_scaling": ("scaling",), "primary_regimeD": ("regimeD",), "primary_assumption": ("assumption", "vault"), "primary_noise": ("noise",)}


def write_group(name, fams):
    with open(f"{T}/{name}.tex", "w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\nComparison & $\\Delta$ & 95\\% CI & $p_{\\text{Holm}}$ & $d$ \\\\\n\\midrule\n")
        last = None
        sel = P[P.family.isin(fams)]
        if name == "primary_noise":                    # table shows the strongest-noise rows; all rows are in primary_comparisons.csv
            sel = sel[sel.comparison.str.startswith("slip 0.4")]
        for _, r in sel.iterrows():
            if last and r.family != last:
                f.write("\\midrule\n")
            last = r.family
            d = "$\\infty$" if np.isinf(r.d) else f"{r.d:.1f}"
            f.write(f"{r.comparison.replace('_', ' ').replace('%', chr(92) + '%')} & {r['diff']:+.3f} & [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {tex_p(r.p_holm)} & {d} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


for g, fams in GROUPS.items():
    write_group(g, fams)

# ---- spread across independently trained models
sp = []
for dist, df in (("main", sc),):
    for a in sorted(set(df.agent)):
        if not is_learned(a) or a.startswith("ppo_"):
            continue
        for d in (3, 4, 5):
            u = units(df[df.difficulty == d], a, dist)
            if len(u):
                lo, hi = boot_mean_ci(u)
                sp.append(dict(agent=a, setting=f"d={d}", n_models=len(u), mean=u.mean(), sd=u.std(ddof=1) if len(u) > 1 else np.nan, min=u.min(), max=u.max(), ci_lo=lo, ci_hi=hi))
for a in ("macro_ff", "macro_gru", "macro_gruB"):
    for c in sorted(gen.condition.unique()):
        g = gen[(gen.condition == c) & (gen.split == "test")]
        u = units(g, a, c)
        if len(u):
            lo, hi = boot_mean_ci(u)
            sp.append(dict(agent=a, setting=f"regime {c.split('_')[0]} (test)", n_models=len(u), mean=u.mean(), sd=u.std(ddof=1) if len(u) > 1 else np.nan, min=u.min(), max=u.max(), ci_lo=lo, ci_hi=hi))
for a in ("vault_ff", "vault_gru"):
    for c in ("in_dist", "larger"):
        u = units(vt[vt.condition == c], a)
        if len(u):
            lo, hi = boot_mean_ci(u)
            sp.append(dict(agent=a, setting=f"vault {c}", n_models=len(u), mean=u.mean(), sd=u.std(ddof=1) if len(u) > 1 else np.nan, min=u.min(), max=u.max(), ci_lo=lo, ci_hi=hi))
S = pd.DataFrame(sp)
S.round(4).to_csv(f"{T}/learned_spread.csv", index=False)

# ---- train-test gaps per regime (unit level)
gp = []
for c in sorted(gen.condition.unique()):
    for a in sorted(set(gen.agent)):
        if a == "random":
            continue
        tr = units(gen[(gen.condition == c) & (gen.split == "train_heldout")], a, c)
        te = units(gen[(gen.condition == c) & (gen.split == "test")], a, c)
        if len(tr) and len(tr) == len(te):
            g = tr - te
            lo, hi = boot_mean_ci(g)
            gp.append(dict(regime=c, agent=a, n_units=len(g), train=tr.mean(), test=te.mean(), gap=g.mean(), gap_lo=lo, gap_hi=hi))
pd.DataFrame(gp).round(4).to_csv(f"{T}/gaps.csv", index=False)
print(P[["comparison", "diff", "ci_lo", "ci_hi", "p_holm", "n_a", "n_b"]].round(3).to_string())
print(S.round(3).to_string())
