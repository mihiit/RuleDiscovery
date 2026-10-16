"""Builds paper/main.tex from paper/head.tex + front.tex.in + results.tex.in + tail.tex.in.

Prose lives in the *.in files. Every number in them is a token  @{python expression}  or  @{expression!format}  that is evaluated here against
data/results/*.csv and paper/tables/primary_comparisons.csv, so the text cannot drift away from the data. Tables are generated below.
Run scripts/final_stats.py first (it writes primary_comparisons.csv).
"""
import os, re, sys, glob
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
from evaluation.hierarchical import units, boot_mean_ci, n_models, is_learned

R, T, P = "data/results", "paper/tables", "paper"
rd = lambda n: pd.read_csv(f"{R}/{n}.csv")
sc, gen, mk, vt, ab, fx, th, rt, rc, bl, nz = (rd(n) for n in ("scaling", "generalization", "master_key", "vault", "ablation", "factorial", "theory", "rule_types", "recovery", "baselines", "noise"))
prim = pd.read_csv(f"{T}/primary_comparisons.csv")
spread = pd.read_csv(f"{T}/learned_spread.csv")
LAB = {"random": "Random", "reactive": "Reactive", "psrl": "PSRL", "hypothesis": "Hypothesis", "hyp_adaptive": "Hypothesis-adaptive", "hyp_nonexcl": "Hyp.\\ (no exclusivity)",
       "macro_ff": "Macro-FF", "macro_gru": "Macro-GRU", "macro_gruB": "Macro-GRU+belief", "generic_ff": "Generic-FF", "generic_gru": "Generic-GRU",
       "ppo_ff": "PPO-FF (primitive)", "ppo_gru": "PPO-GRU (primitive)", "vault_ff": "PPO-FF", "vault_gru": "PPO-GRU"}


def U(df, a, dist="main", col="success_rate", **f):
    for k, v in f.items():
        df = df[df[k] == v]
    return units(df, a, dist, col)


def SR(a, d):
    u = U(sc, a, "main", difficulty=d); return float(u.mean())


def SRCI(a, d):
    u = U(sc, a, "main", difficulty=d); lo, hi = boot_mean_ci(u); return f"[{lo:.3f}, {hi:.3f}]"


def TE(a, r, split="test"):
    return float(U(gen[gen.split == split], a, r, condition=r).mean())


def GAP(a, r):
    return TE(a, r, "train_heldout") - TE(a, r, "test")


def MK(a, c):
    return float(U(mk, a, "main", condition=c).mean())


def VT(a, c, col="success_rate"):
    return float(U(vt, a, "main", col=col, condition=c).mean())


def VTC(a, c, col="success_rate"):
    lo, hi = boot_mean_ci(U(vt, a, "main", col=col, condition=c)); return f"[{lo:.3f}, {hi:.3f}]"


def A(name, col="success_rate"):
    return float(ab[ab.agent == name][col].mean())


def FX(name, d):
    return float(fx[(fx.agent == name) & (fx.difficulty == d)].success_rate.mean())


def TH(a, d, col="failed_door_tests"):
    return float(th[(th.agent == a) & (th.difficulty == d)][col].mean())


def row(label):
    r = prim[prim.comparison == label]
    if r.empty:
        raise KeyError(f"primary comparison not found: {label}")
    return r.iloc[0]


def PS(label):
    """compact: '$\\Delta=+0.172$ [+0.150, +0.194]' (Holm-adjusted p appended only when not significant at 0.05; all p are in the primary table)"""
    r = row(label)
    s = f"$\\Delta={r['diff']:+.3f}$ [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
    if r.p_holm >= 0.05:
        s += f", $p_{{\\mathrm{{Holm}}}}={r.p_holm:.2f}$"
    return s


def SIG(label):
    r = row(label)
    return 0 if r.p_holm >= 0.05 else (1 if r["diff"] > 0 else -1)


def WORD(label, up="higher", same="indistinguishable", down="lower"):
    return {1: up, 0: same, -1: down}[SIG(label)]


def NM(a, dist="main"):
    return n_models(a, dist)


NS = dict(nz=nz, U=U, SR=SR, SRCI=SRCI, TE=TE, GAP=GAP, MK=MK, VT=VT, VTC=VTC, A=A, FX=FX, TH=TH, PS=PS, SIG=SIG, WORD=WORD, NM=NM, np=np, row=row, prim=prim, spread=spread,
          sc=sc, gen=gen, mk=mk, vt=vt, ab=ab, fx=fx, th=th, rt=rt, rc=rc, bl=bl, LAB=LAB, boot_mean_ci=boot_mean_ci)
from evaluation.hierarchical import compare_units
def DIFF(df, a, b, **flt):
    """exploratory (unadjusted) difference of two agents' unit means with bootstrap CI"""
    for k, v in flt.items():
        df = df[df[k] == v]
    r = compare_units(units(df, a), units(df, b))
    return f"$\\Delta={r['diff']:+.3f}$ [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
def REC(a):
    return float(rc[rc.agent == a].success_rate.mean())
def RT(a, c):
    return float(rt[(rt.agent == a) & (rt.condition == c)].success_rate.mean())
import json
FA = json.load(open(f"{R}/failure_analysis.json"))
def BUD(d, steps):
    return [r["success_rate"] for r in FA["budget_sweep"] if r["difficulty"] == d and r["max_steps"] == steps][0]
NS.update(FA=FA, BUD=BUD, DIFF=DIFF, REC=REC, RT=RT)
NS["NMODELS"] = NM("macro_gru", "main")
NS["NGENERIC"] = NM("generic_gru", "main")
NS["NPRIMARY"] = len(prim)


def sub(text):
    def rep(m):
        expr, fmt = m.group(1), m.group(2)
        try:
            v = eval(expr, NS)
        except Exception as e:
            raise RuntimeError(f"token @{{{expr}}} failed: {e!r}")
        if fmt:
            return format(v, fmt).replace("%", "\\%")
        if isinstance(v, (float, np.floating)):
            return f"{v:.3f}"
        return str(v)
    return re.sub(r"@\{([^{}!]+?)(?:!([^{}]*))?\}", rep, text)


# ----------------------------------------------------------------------------------------------------------------- generated tables
def cellfmt(a, d):
    u = U(sc, a, "main", difficulty=d)
    if len(u) == 0:
        return "--"
    lo, hi = boot_mean_ci(u) if len(u) > 1 else (u.mean(), u.mean())
    return f"{u.mean():.3f} [{lo:.3f}, {hi:.3f}]"


def table_main():
    order = ["random", "reactive", "psrl", "hypothesis", "hyp_adaptive", "macro_ff", "macro_gru", "macro_gruB", "generic_ff", "generic_gru", "ppo_ff", "ppo_gru"]
    out = ["\\begin{table}[!htbp]\n\\caption{Success rate by difficulty: mean [95\\% bootstrap CI over units]. Units are evaluation seeds for agents without trained weights and trained models for learned agents; $n$ is the number of units. Agents that choose among testable macro-actions are separated from the reactive and primitive baselines.}\\label{tab:main}\n\\centering\\scriptsize\n"
           "\\begin{tabular}{lcccc}\n\\toprule\nAgent & $n$ & $d=3$ & $d=4$ & $d=5$ \\\\\n\\midrule"]
    for a in order:
        if a not in set(sc.agent):
            continue
        n = len(U(sc, a, "main", difficulty=3))
        out.append(f"{LAB[a]} & {n} & {cellfmt(a, 3)} & {cellfmt(a, 4)} & {cellfmt(a, 5)} \\\\")
    out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return "\n".join(out)


def table_regimes():
    regs = [("A_interpolation", "A"), ("B_compositional", "B"), ("C_chain_length", "C"), ("C2_size_only", "C2"), ("D_rule_permutation", "D"), ("E_novel_combination", "E")]
    order = [a for a in ("reactive", "psrl", "hypothesis", "hyp_adaptive", "macro_ff", "macro_gru", "macro_gruB") if a in set(gen.agent)]
    out = ["\\begin{table}[!htbp]\n\\caption{Held-out success per regime; in parentheses the train-minus-test gap (positive = worse on the held-out distribution). Learned agents: mean over trained models; the last row gives the number of models per regime for Macro-GRU.}\\label{tab:gen}\n\\centering\\scriptsize\n"
           "\\begin{tabular}{l" + "c" * len(regs) + "}\n\\toprule\nAgent & " + " & ".join(r[1] for r in regs) + " \\\\\n\\midrule"]
    for a in order:
        cells = []
        for rn, _ in regs:
            if len(U(gen[gen.split == "test"], a, rn, condition=rn)) == 0:
                cells.append("--"); continue
            cells.append(f"{TE(a, rn):.2f} ({GAP(a, rn):+.2f})")
        out.append(f"{LAB[a]} & " + " & ".join(cells) + " \\\\")
    out.append("\\midrule\nModels (Macro-GRU) & " + " & ".join(str(NM("macro_gru", rn)) for rn, _ in regs) + " \\\\")
    out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return "\n".join(out)


def _ptable(name, label, caption):
    body = open(f"{T}/{name}.tex").read()
    return f"\\begin{{table}}[!htbp]\n\\caption{{{caption}}}\\label{{{label}}}\n\\centering\\scriptsize\\setlength{{\\tabcolsep}}{{3pt}}\n" + body + "\\end{table}\n"


def table_p1():
    return _ptable("primary_scaling", "tab:p1", "Pre-declared comparisons, difficulty scaling. $\\Delta$: difference of mean success (first minus second) over units; CI: percentile bootstrap, units resampled within each agent; $p_{\\mathrm{Holm}}$: Welch $p$ adjusted over all @{NPRIMARY} comparisons of Tables~\\ref{tab:p1}--\\ref{tab:p4}; $d$: Cohen's $d$. Units are 30 evaluation seeds for agents without trained weights and one per trained model for learned agents.")


def table_p2():
    return _ptable("primary_regimeD", "tab:p2", "Pre-declared comparisons, regime D (unseen colors), same conventions as Table~\\ref{tab:p1}.")


def table_p3():
    return _ptable("primary_assumption", "tab:p3", "Pre-declared comparisons, assumption violation (master key and exclusive control) and vault extrapolation, same conventions as Table~\\ref{tab:p1}.")


def table_p4():
    return _ptable("primary_noise", "tab:p4", "Pre-declared comparisons, noisy feedback (jammed locks), same conventions as Table~\\ref{tab:p1}. The table shows the rows for $\\epsilon=0.4$; the $\\epsilon=0.2$ rows are in \\texttt{primary\\_comparisons.csv}. These comparisons were added in a second analysis round.")


def SRN(a, d, slip):
    return float(units(nz[(nz.difficulty == d) & (nz.slip == slip)], a, "main").mean())


def MDE(a, d, nb=30):
    """minimum detectable difference (80% power, two-sided 5%) against a 30-unit agent, from the observed between-unit SD of agent a at difficulty d"""
    u = U(sc, a, "main", difficulty=d)
    return float(2.8 * u.std(ddof=1) * np.sqrt(1 / len(u) + 1 / nb))


def RO3(n):
    """95% upper bound on a failure rate after 0 failures in n episodes ('rule of three')"""
    return 3.0 / n


NS.update(SRN=SRN, MDE=MDE, RO3=RO3)


def _cg():
    return pd.read_csv(f"{T}/causalgame_agents.csv"), pd.read_csv(f"{T}/causalgame_comparisons.csv")


def CGS(label, col):
    S, _ = _cg(); return float(S[S.agent == label][col].iloc[0])


def CGD(i, col):
    _, C = _cg(); return float(C.iloc[i][col])


NS.update(CGS=CGS, CGD=CGD)


def table_cg1():
    return ("\\begin{table}[!htbp]\n\\caption{CausalGame (survival part), 14 scenarios, 40 sessions per agent and scenario. Mean over scenarios [95\\% bootstrap CI over scenarios]; success rate is the share of sessions whose submitted design reached the scenario's survival threshold; last column: number of scenarios in which the agent had the highest mean survival.}\\label{tab:cg1}\n"
            "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n" + open(f"{T}/causalgame_agents.tex").read() + "\\end{table}\n")


def table_cg2():
    return ("\\begin{table}[!htbp]\n\\caption{Pre-declared CausalGame comparisons (five, Holm-adjusted together): difference in mean survival over the 14 scenarios, bootstrap CI over scenarios, scenarios won by the first agent, and Wilcoxon signed-rank $p$.}\\label{tab:cg2}\n"
            "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n" + open(f"{T}/causalgame_comparisons.tex").read() + "\\end{table}\n")


def table_spread():
    s = spread[(spread.setting.isin(["d=3", "d=4", "d=5"])) & (spread.agent.isin(["macro_ff", "macro_gru", "macro_gruB", "generic_ff", "generic_gru"]))]
    out = ["\\begin{table}[!htbp]\n\\caption{Spread across independently trained models (success rate; mean, SD, min--max over models).}\\label{tab:spread}\n\\centering\\scriptsize\n"
           "\\begin{tabular}{llcccc}\n\\toprule\nAgent & Level & Models & Mean & SD & Min--max \\\\\n\\midrule"]
    for _, r in s.iterrows():
        out.append(f"{LAB[r.agent]} & {r.setting.replace('=', '=')} & {int(r.n_models)} & {r['mean']:.3f} & {r.sd:.3f} & {r['min']:.3f}--{r['max']:.3f} \\\\")
    out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return "\n".join(out)


GEN_TABLES = dict(TABLE_MAIN=table_main, TABLE_REGIMES=table_regimes, TABLE_P1=table_p1, TABLE_P2=table_p2, TABLE_P3=table_p3, TABLE_P4=table_p4, TABLE_CG1=table_cg1, TABLE_CG2=table_cg2, TABLE_SPREAD=table_spread)


def to_ieee(txt):
    """Adapt the LNCS source to IEEEtran two-column: wide floats become starred, figures come from paper/figures_ieee, bibliography style changes."""
    # ---- structure and wording to follow the IEEE conference layout
    c = txt.index("\\textbf{Contributions.}"); d = txt.index("\\section{Related Work}")
    block = re.sub(r"\n\(\d\)~", "\n\\\\item ", txt[c:d].replace("\\textbf{Contributions.}", "The contributions of this paper are:\n\\begin{itemize}", 1).rstrip()) + "\n\\end{itemize}\n\n"
    txt = txt[:c] + block + txt[d:]
    txt = txt.replace("\\section{Benchmarks}", "\\section{Method}\n\\subsection{Benchmarks}", 1)
    txt = txt.replace("\\section{Agents}", "\\subsection{Agents}", 1)
    txt = txt.replace("\\section{Protocol and Statistics}", "\\section{Experimental Setup}", 1)
    txt = txt.replace("Sect.~", "Section~")
    l0 = txt.index("\\textbf{Limitations.}"); l1 = txt.index("\\section{Conclusion}")
    txt = txt[:l0] + sub(open(f"{P}/tail_ieee_limitations.tex.in").read()) + "\n" + txt[l1:]
    txt = txt.replace("\\bibliographystyle{splncs04}", "\\bibliographystyle{IEEEtran}")
    def fig(m):
        b = m.group(0)
        wide = any(k in b for k in ("fig_regimes", "fig_vault", "fig_infobonus"))
        rep = "width=\\textwidth]{figures_ieee/" if wide else "width=\\columnwidth]{figures_ieee/"
        b = re.sub(r"width=0\.\d+\\textwidth\]\{figures/", lambda m: rep, b)
        return b.replace("\\begin{figure}[!htbp]", "\\begin{figure*}[!t]").replace("\\end{figure}", "\\end{figure*}") if wide else b
    txt = re.sub(r"\\begin\{figure\}\[[^\]]*\].*?\\end\{figure\}", fig, txt, flags=re.S)
    wide = ("tab:main", "tab:gen", "tab:info")
    def tab(m):
        t = m.group(0)
        if any(f"\\label{{{w}}}" in t for w in wide):
            t = t.replace("\\begin{table}[!htbp]", "\\begin{table*}[!t]").replace("\\end{table}", "\\end{table*}")
            t = t.replace("\\resizebox{\\textwidth}{!}{%\n", "{%\n")          # natural size in the wide float (do not up-scale)
        else:                                                       # narrow tables: single column, scaled to fit if needed
            t = re.sub(r"(\\begin\{tabular\}.*?\\end\{tabular\})", lambda k: "\\resizebox{\\columnwidth}{!}{" + k.group(1) + "}", t, flags=re.S)
        return t
    txt = re.sub(r"\\begin\{table\}\[[^\]]*\].*?\\end\{table\}", tab, txt, flags=re.S)
    return txt


def build(ieee=False):
    parts = [open(f"{P}/{f}").read() for f in (("head_ieee.tex" if ieee else "head.tex"), "front.tex.in", "results.tex.in", "tail.tex.in")]
    if ieee:
        abs_ = open(f"{P}/abstract_ieee.tex.in").read()
        parts[0] = parts[0].replace("%%ABSTRACT%%", abs_)
    txt = "\n".join(parts)
    for k, f in GEN_TABLES.items():
        txt = txt.replace(f"@@{k}@@", f())
    txt = sub(txt)
    left = re.findall(r"@\{.*?\}|@@\w+@@", txt)
    assert not left, f"unresolved tokens: {left[:5]}"
    if ieee:
        txt = to_ieee(txt)
    out = "main_ieee.tex" if ieee else "main.tex"
    open(f"{P}/{out}", "w").write(txt)
    print("wrote paper/" + out, len(txt), "chars")




# ----------------------------------------------------------------------------------------------------------- 6-page IEEE version
def i6_info():
    rows = [("Reactive", "primitives", "lose-shift bit", "no", "--", "--"), ("PSRL", "macros", "outcome log", "yes", "hard", "--"),
            ("Hypothesis (tracker)", "macros", "log, confidences", "yes", "hard", "--"), ("Hypothesis-adaptive", "macros", "log, confidences", "yes", "soft prior", "--"),
            ("Macro-FF / GRU", "macros", "last outcome (+GRU state)", "implicit", "implicit", str(NM("macro_gru"))),
            ("Macro-GRU+belief", "macros", "tracker beliefs as inputs", "via tracker", "via tracker", str(NM("macro_gruB"))),
            ("Generic-FF / GRU", "generic skills", "last outcome (+GRU state)", "implicit", "implicit", str(NM("generic_gru"))),
            ("Primitive PPO", "primitives", "GRU state", "implicit", "implicit", "1")]
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    return ("\\begin{table*}[!t]\n\\caption{Information each agent is given. ``Templates'': hand-built hypothesis forms; $n$: independently trained models (main distribution).}\\label{tab:info}\n\\centering\\footnotesize\n"
            "\\begin{tabular}{llllll}\n\\toprule\nAgent & Actions & Evidence kept & Templates & Exclusivity & $n$ \\\\\n\\midrule\n" + body + " \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def i6_main():
    order = ["reactive", "psrl", "hypothesis", "hyp_adaptive", "macro_ff", "macro_gru", "macro_gruB", "generic_gru", "ppo_gru"]
    out = ["\\begin{table*}[!t]\n\\caption{Success by difficulty and on unseen color symbols (regime D, held out): mean [95\\% bootstrap CI over units]. Units: evaluation seeds, or trained models for learned agents ($n$).}\\label{tab:main}\n\\centering\\footnotesize\n"
           "\\begin{tabular}{lcccc}\n\\toprule\nAgent & $d=3$ & $d=4$ & $d=5$ & Unseen colors (D) \\\\\n\\midrule"]
    for a in order:
        if a not in set(sc.agent):
            continue
        n = len(U(sc, a, "main", difficulty=3))
        u = U(gen[gen.split == "test"], a, "D_rule_permutation", condition="D_rule_permutation")
        if len(u) > 1:
            lo, hi = boot_mean_ci(u); dcell = f"{u.mean():.3f} [{lo:.3f}, {hi:.3f}]"
        else:
            dcell = "--"
        lab = LAB[a] + (f" ($n={n}$)" if is_learned(a) else "")
        out.append(f"{lab} & {cellfmt(a, 3)} & {cellfmt(a, 4)} & {cellfmt(a, 5)} & {dcell} \\\\")
    out.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return "\n".join(out)


def i6_cg():
    S, _ = _cg()
    ren = {"Default design": "Default design", "Random search": "Random search", "Hill-climbing (reactive)": "Hill-climbing", "Posterior sampling": "Posterior sampling", "Hypothesis, no info bonus": "EV, no info bonus", "Hypothesis (EV + info gain)": "EV + info gain"}
    rows = " \\\\\n".join(f"{ren[r.agent]} & {r.survival:.3f} [{r.s_lo:.3f}, {r.s_hi:.3f}] & {r.win:.3f} \\" if False else f"{ren[r.agent]} & {r.survival:.3f} [{r.s_lo:.3f}, {r.s_hi:.3f}] & {r.win:.3f}" for _, r in S.iterrows())
    return ("\\begin{table}[!t]\n\\caption{CausalGame, survival part: mean over 14 scenarios [95\\% CI over scenarios]; success = share of sessions reaching the scenario threshold.}\\label{tab:cg}\n\\centering\\footnotesize\n"
            "\\begin{tabular}{lcc}\n\\toprule\nAgent & Survival & Success \\\\\n\\midrule\n" + rows + " \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


KEY = [("d=5: Hypothesis vs PSRL", "d5: tracker vs PSRL"), ("d=5: Hypothesis vs Reactive", "d5: tracker vs reactive"),
       ("master_key_d3: Hyp.-adaptive vs Hypothesis", "master key d3: adaptive vs hard"), ("master_key_d4: Hyp.-adaptive vs Hypothesis", "master key d4: adaptive vs hard"),
       ("exclusive_control_d4: Hyp.-adaptive vs Hypothesis", "control d4: adaptive vs hard"), ("D: Hypothesis vs PSRL", "D: tracker vs PSRL"),
       ("vault larger codes: Hypothesis vs PPO-GRU", "vault larger: tracker vs PPO-GRU"), ("slip 0.4, d=3: Hypothesis vs PSRL", "slip 0.4, d3: tracker vs PSRL"),
       ("slip 0.4, d=4: Hypothesis vs PSRL", "slip 0.4, d4: tracker vs PSRL")]


def i6_dec():
    def cell(lbl, sign=1):
        r = row(lbl); d, lo, hi = sign * r["diff"], (r.ci_lo if sign > 0 else -r.ci_hi), (r.ci_hi if sign > 0 else -r.ci_lo)
        return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    rows = []
    for k, name in (("d=3", "difficulty 3"), ("d=4", "difficulty 4"), ("d=5", "difficulty 5"), ("D", "unseen colors (D)")):
        rows.append(f"{name} & {cell(k + ': Hypothesis vs Macro-GRU')} & {cell(k + ': Macro-GRU+belief vs Macro-GRU')} & {cell(k + ': Hypothesis vs Macro-GRU+belief', -1) if False else cell(k + ': Hypothesis vs Macro-GRU+belief')}")
    return ("\\begin{table}[!t]\n\\caption{Decomposition of the tracker's advantage over Macro-GRU (Eq.~\\eqref{eq:dec}): total $S_H-S_L$, information gap $S_{L+b}-S_L$ and learning gap $S_H-S_{L+b}$, each as $\\Delta$ success [95\\% CI over units]; learned agents: @{NMODELS} trained models (belief agent: @{NM('macro_gruB')}).}\\label{tab:dec}\n\\centering\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\n"
            "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lccc}\n\\toprule\nCondition & Total & Information gap & Learning gap \\\\\n\\midrule\n" + " \\\\\n".join(rows) + " \\\\\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def i6_key():
    rows = []
    for lab, short in KEY:
        r = row(lab)
        p = "$<$0.001" if r.p_holm < 0.001 else f"{r.p_holm:.2f}"
        d = "$\\infty$" if np.isinf(r.d) else f"{r.d:.1f}"
        rows.append(f"{short} & {r['diff']:+.3f} & [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {p} & {d}")
    return ("\\begin{table}[!t]\n\\caption{Key pre-declared comparisons (first minus second) from the 65 Holm-corrected tests: $\\Delta$ success, 95\\% bootstrap CI over units, adjusted $p$, Cohen's $d$. Full list: \\texttt{paper/tables/primary\\_comparisons.csv}.}\\label{tab:key}\n\\centering\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\n"
            "\\begin{tabular}{lrrrr}\n\\toprule\nComparison & $\\Delta$ & 95\\% CI & $p_{\\text{Holm}}$ & $d$ \\\\\n\\midrule\n" + " \\\\\n".join(rows) + " \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")




def i6_all():
    """One full-width table with three panels: (a) decomposition, (b) key comparisons, (c) CausalGame."""
    def inner(s):
        m = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", s, flags=re.S)
        return m.group(0)
    a, k, c = inner(i6_dec()), inner(i6_key()), inner(i6_cg())
    cap = ("Results with intervals. (a) Decomposition of the tracker's advantage over Macro-GRU (Eq.~\\eqref{eq:dec}): total $S_H-S_L$, information gap $S_{L+b}-S_L$, learning gap $S_H-S_{L+b}$, as $\\Delta$ success [95\\% CI over units]; "
           "learned agents: @{NMODELS} trained models (belief agent @{NM('macro_gruB')}). (b) Key comparisons (first minus second) from the 65 Holm-corrected tests: $\\Delta$, 95\\% CI, adjusted $p$, Cohen's $d$; all tests are in \\texttt{paper/tables/primary\\_comparisons.csv}. "
           "(c) CausalGame, survival part: mean over 14 scenarios [95\\% CI over scenarios]; success = share of sessions reaching the scenario threshold.")
    return ("\\begin{table*}[!t]\n\\caption{" + cap + "}\\label{tab:res}\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{minipage}[t]{0.57\\textwidth}\\centering\\textbf{(a)}\\\\[2pt]\n" + a + "\\\\[6pt]\\textbf{(b)}\\\\[2pt]\n" + k + "\n\\end{minipage}\\hfill\n"
            "\\begin{minipage}[t]{0.40\\textwidth}\\centering\\textbf{(c)}\\\\[2pt]\n" + c + "\n\\end{minipage}\n\\end{table*}\n")


def build_ieee6():
    txt = open(f"{P}/ieee6.tex.in").read()
    for k, f in dict(TABLE_INFO=i6_info, TABLE_MAIN=i6_main, TABLE_CG=i6_cg, TABLE_KEY=i6_key, TABLE_DEC=i6_dec, TABLE_ALL=i6_all).items():
        txt = txt.replace(f"@@{k}@@", f())
    txt = sub(txt)
    left = re.findall(r"@\{.*?\}|@@\w+@@", txt)
    assert not left, f"unresolved tokens: {left[:5]}"
    open(f"{P}/main_ieee6.tex", "w").write(txt)
    print("wrote paper/main_ieee6.tex", len(txt), "chars")


if __name__ == "__main__":
    build(ieee=False); build_ieee6()
