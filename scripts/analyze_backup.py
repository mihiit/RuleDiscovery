"""Builds all tables (CSV + Markdown + LaTeX) and results_summary.json from data/results/*.csv."""

import os, sys, json, glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd

from evaluation.significance import bootstrap_ci, compare, holm

from environment.generator import DEFAULT_SIZE, DEFAULT_RULES



R, T = "data/results", "paper/tables"

os.makedirs(T, exist_ok=True)

ORDER = ["random", "reactive", "psrl", "hypothesis", "hyp_adaptive", "hyp_transfer", "macro_ff", "macro_gru", "macro_gruB", "generic_ff", "generic_gru", "ppo_ff", "ppo_gru"]





def fmt(x, ci=True, p=3):

    x = np.asarray(x, float); x = x[~np.isnan(x)]

    if len(x) == 0:

        return "n/a"

    if not ci:

        return f"{x.mean():.{p}f} ± {x.std(ddof=1) if len(x) > 1 else 0:.{p}f}"

    lo, hi = bootstrap_ci(x)

    return f"{x.mean():.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"





def write(name, df, caption):

    df.to_csv(f"{T}/{name}.csv", index=False)

    with open(f"{T}/{name}.md", "w") as f:

        f.write(f"**{caption}**\n\n" + df.to_markdown(index=False) + "\n")





def vals(df, **f):

    for k, v in f.items():

        df = df[df[k] == v]

    return df.sort_values("seed")





def agents_in(df):

    return [a for a in ORDER if a in set(df.agent)] + [a for a in df.agent.unique() if a not in ORDER]






def load_d30(path):
    """Load and normalize the 30-seed D rule-permutation re-evaluation.

    The evaluator may emit either per-environment ``success`` rows or already
    aggregated ``success_rate`` rows. Normalize both formats to one score per
    agent/seed (and split, if present), so the trained model/evaluation seed is
    the statistical unit.
    """
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    if "agent" not in df.columns or "seed" not in df.columns:
        raise ValueError(f"D30 file must contain 'agent' and 'seed' columns: {path}")

    if "split" not in df.columns:
        df["split"] = "test"

    if "success_rate" not in df.columns:
        if "success" not in df.columns:
            raise ValueError(
                f"D30 file must contain either 'success_rate' or 'success': {path}"
            )
        df["success_rate"] = pd.to_numeric(df["success"], errors="coerce")

    df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce")
    df = df.dropna(subset=["success_rate"]).copy()

    # Collapse per-environment rows to one seed-level score. If the file is
    # already seed-level, this groupby leaves the values unchanged.
    out = (
        df.groupby(["agent", "seed", "split"], as_index=False, sort=True)["success_rate"]
          .mean()
    )
    out["condition"] = "D_rule_permutation"
    return out


def d30_stats(d30):
    """Return descriptive statistics for each D30 agent."""
    rows = []
    for a in agents_in(d30):
        x = vals(d30[d30.agent == a], split="test").success_rate.values
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        if len(x) == 0:
            continue
        lo, hi = bootstrap_ci(x)
        rows.append({
            "Agent": a,
            "N seeds": len(x),
            "Mean": x.mean(),
            "SD": x.std(ddof=1) if len(x) > 1 else 0.0,
            "95% CI low": lo,
            "95% CI high": hi,
        })
    return pd.DataFrame(rows) if rows else None

def main():

    S, sig = {}, []



    def add(exp, label, a, b):

        sig.append(dict(Experiment=exp, Comparison=label, **compare(a, b)))



    # Table 1/2 ------------------------------------------------------------------------------------------------

    rows = [dict(Difficulty=d, Grid=DEFAULT_SIZE[d], Gates="+".join({"KEY_DOOR": "KD", "SWITCH_BRIDGE": "SB", "PUSH_BLOCK": "PB"}[r] for r in DEFAULT_RULES[d]),

                 Distractors=max(0, d - 2), Traps=max(0, d - 1)) for d in range(1, 6)]

    write("table1_environments", pd.DataFrame(rows), "Environment configurations per difficulty (KD key-door, SB switch-bridge, PB push-block).")

    write("table2_agents", pd.DataFrame([

        ["Random", "uniform over 7 actions", "-", "-"],

        ["Reactive", "BFS heuristics + 1 lose-shift bit", "none", "none"],

        ["PSRL", "posterior-sampled bindings, nearest-first macros", "evidence log", "sampled assignment"],

        ["Hypothesis", "macros chosen by U=R+lam*IG-mu*cost", "episodic", "KD/SB/PUSH(/SD) with alpha/beta updates"],

        ["Macro-FF", "PPO over macro-actions (no belief features)", "last macro+outcome", "implicit"],

        ["Macro-GRU", "PPO over macro-actions, GRU history", "recurrent state", "implicit"],

        ["PPO-FF", "CNN policy, PPO (JAX), shaped reward", "none (feedback input)", "implicit"],

        ["PPO-GRU", "CNN+GRU policy (RL^2-style), PPO, shaped reward", "recurrent state", "implicit"]],

        columns=["Agent", "Policy", "Memory", "Hypotheses"]), "Agent architectures.")

    # Main -----------------------------------------------------------------------------------------------------

    b = pd.read_csv(f"{R}/baselines.csv")

    rows = []

    for a in agents_in(b):

        g = b[b.agent == a]

        rows.append({"Agent": a, "Success": fmt(g.success_rate), "Steps to success": fmt(g.mean_steps_success, p=1),

                     "Succ./1k steps": fmt(g.sample_efficiency * 1000, p=2), "Actions/s": f"{g.actions_per_sec.mean():.0f}",

                     "Hyp. acc.": fmt(g.hyp_accuracy) if "hyp_accuracy" in g and g.hyp_accuracy.notna().any() else "-",

                     "Recovery": fmt(g.recovery_rate)})

    write("table3_main", pd.DataFrame(rows), "Main results, difficulty 3 (mean [95% bootstrap CI over seeds]).")

    # Scaling --------------------------------------------------------------------------------------------------

    sc = pd.read_csv(f"{R}/scaling.csv")

    rows = []

    for d in range(1, 6):

        for a in agents_in(sc):

            g = vals(sc, difficulty=d, agent=a)

            rows.append({"Difficulty": d, "Agent": a, "Success": fmt(g.success_rate), "Steps to success": fmt(g.mean_steps_success, p=1)})

        h = vals(sc, difficulty=d, agent="hypothesis")

        for o in ("reactive", "psrl", "macro_gru", "macro_ff", "generic_gru", "generic_ff", "ppo_gru", "ppo_ff"):

            if o in set(sc.agent):

                add("scaling", f"hypothesis vs {o}, d={d}, success", h.success_rate, vals(sc, difficulty=d, agent=o).success_rate)

        for g_ in ("generic_gru", "generic_ff"):

            if g_ in set(sc.agent) and "macro_gru" in set(sc.agent):

                add("scaling", f"macro_gru vs {g_}, d={d}, success", vals(sc, difficulty=d, agent="macro_gru").success_rate, vals(sc, difficulty=d, agent=g_).success_rate)

        add("scaling", f"hypothesis vs reactive, d={d}, steps-to-success", h.mean_steps_success, vals(sc, difficulty=d, agent="reactive").mean_steps_success)

    write("table_scaling", pd.DataFrame(rows), "Scaling results by difficulty.")

    # Generalization -------------------------------------------------------------------------------------------

    # The original file contains the broad six-regime evaluation.  For the
    # rule-permutation regime, use the new 30-seed re-evaluation when it is
    # available.  This replaces the old D test scores rather than mixing
    # checkpoints from the two experiments.
    gen = pd.read_csv(f"{R}/generalization.csv")
    d30 = load_d30(f"{R}/generalization_D_30.csv")

    # Keep the original generalization data for A/B/C/C2/E.  D is removed
    # entirely when the new 30-seed file is available so that old D train/test
    # scores are never combined with the new D checkpoints.
    if d30 is not None:
        gen = gen[gen.condition != "D_rule_permutation"].copy()

    rows = []

    for reg in gen.condition.unique():
        for a in agents_in(gen):
            q = gen[(gen.condition == reg) & (gen.agent == a)]
            if q.empty:
                continue

            tr = vals(q, split="train_heldout").success_rate.values
            te = vals(q, split="test").success_rate.values
            gp = tr - te if len(tr) == len(te) else np.array([np.nan])

            rows.append({
                "Regime": reg,
                "Agent": a,
                "Train SR": fmt(tr),
                "Test SR": fmt(te),
                "Gap": fmt(gp),
            })

        te = lambda a: vals(
            gen[(gen.condition == reg) & (gen.agent == a)],
            split="test"
        ).success_rate.values
        gp = lambda a: (
            vals(gen[(gen.condition == reg) & (gen.agent == a)],
                 split="train_heldout").success_rate.values - te(a)
        )

        for o in ("reactive", "psrl", "macro_gru", "macro_ff"):
            if o in set(gen[gen.condition == reg].agent):
                add(
                    "generalization",
                    f"{reg}: hypothesis vs {o}, test SR",
                    te("hypothesis"),
                    te(o),
                )

        add(
            "generalization",
            f"{reg}: hypothesis vs reactive, gap",
            gp("hypothesis"),
            gp("reactive"),
        )
        add(
            "generalization",
            f"{reg}: hyp_transfer vs hypothesis, test SR",
            te("hyp_transfer"),
            te("hypothesis"),
        )

    # New D: 30 independently trained learned models, 30 evaluation seeds,
    # 30 environments per seed.  The targeted evaluator contains test
    # performance only, so train-SR and train-test gaps are intentionally
    # reported as n/a rather than mixing old training-distribution scores with
    # the new checkpoints.
    if d30 is not None:
        d_agents = agents_in(d30)

        for a in d_agents:
            te = vals(d30[d30.agent == a], split="test").success_rate.values
            if len(te) == 0:
                continue

            rows.append({
                "Regime": "D_rule_permutation",
                "Agent": a,
                "Train SR": "n/a",
                "Test SR": fmt(te),
                "Gap": "n/a",
            })

        dte = lambda a: vals(
            d30[d30.agent == a], split="test"
        ).success_rate.values

        # Keep these new comparisons out of the original table6 significance
        # family because the 30-seed D run is a subsequent confirmatory
        # re-evaluation, not part of the original pre-declared family.
        d30_rows = []
        for o in ("macro_ff", "macro_gru", "psrl", "macro_gruB", "reactive"):
            if o in set(d30.agent):
                c = compare(dte("hypothesis"), dte(o))
                d30_rows.append({
                    "Comparison": f"D_rule_permutation: hypothesis vs {o}",
                    "N seeds": min(len(dte("hypothesis")), len(dte(o))),
                    "Delta": dte("hypothesis").mean() - dte(o).mean(),
                    "Welch p": c.get("welch_p", np.nan),
                    "MWU p": c.get("mwu_p", np.nan),
                })

        if d30_rows:
            dtab = pd.DataFrame(d30_rows)
            dtab["Welch p Holm within D"] = holm(dtab["Welch p"].fillna(1.0))
            dtab["MWU p Holm within D"] = holm(dtab["MWU p"].fillna(1.0))
            write(
                "table_D30_confirmatory",
                dtab.round(6),
                "Confirmatory 30-seed rule-permutation comparisons; separate from the original pre-declared significance family.",
            )

        dstats = d30_stats(d30)
        if dstats is not None:
            write(
                "table_D30_summary",
                dstats.round(6),
                "Rule-permutation re-evaluation over 30 independently trained learned models/evaluation seeds.",
            )

    write(
        "table4_generalization",
        pd.DataFrame(rows),
        "Generalization results (mean [95% bootstrap CI over seeds]).",
    )

    # Ablation -------------------------------------------------------------------------------------------------

    ab = pd.read_csv(f"{R}/ablation.csv")

    rows = []

    for (c, a), g in ab.groupby(["condition", "agent"], sort=False):

        rows.append({"Ablation": c, "Variant": a, "Success": fmt(g.success_rate), "Steps to success": fmt(g.mean_steps_success, p=1),

                     "Redundant": f"{g.hyp_redundant.mean():.2f}" if "hyp_redundant" in g and g.hyp_redundant.notna().any() else "-",

                     "Explore": f"{g.explore_ratio.mean():.3f}" if "explore_ratio" in g and g.explore_ratio.notna().any() else "-"})

    write("table5_ablation", pd.DataFrame(rows), "Ablations (difficulty 4 unless noted; mean [95% bootstrap CI]).")

    sg = lambda a: vals(ab, agent=a).success_rate.values

    for x, y in (("full_memory", "no_memory"), ("full_memory", "fixed_memory_3"), ("info_driven", "random_explore"), ("revision_on", "revision_off")):

        add("ablation", f"{x} vs {y}, success", sg(x), sg(y))

    for l in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):

        add("ablation", f"lambda_{l} vs lambda_0.0, success (d4)", sg(f"lambda_{l}"), sg("lambda_0.0"))

        add("ablation", f"lambda_{l}_d5 vs lambda_0.0_d5, success (d5)", sg(f"lambda_{l}_d5"), sg("lambda_0.0_d5"))

    for m in (0.0, 0.2):

        add("ablation", f"mu_{m} vs mu_0.05, success", sg(f"mu_{m}"), sg("mu_0.05"))

    # Rule types -----------------------------------------------------------------------------------------------

    rt = pd.read_csv(f"{R}/rule_types.csv")

    rows = []

    for c in rt.condition.unique():

        for a in rt[rt.condition == c].agent.unique():

            g = vals(rt, condition=c, agent=a)

            rows.append({"Environments": c, "Agent": a, "Success": fmt(g.success_rate), "Steps to success": fmt(g.mean_steps_success, p=1)})

        gv = lambda a: vals(rt, condition=c, agent=a).success_rate.values

        add("rule_types", f"{c}: hyp_open_vocab vs hyp_closed_vocab", gv("hyp_open_vocab"), gv("hyp_closed_vocab"))

        add("rule_types", f"{c}: hyp_open_vocab vs reactive", gv("hyp_open_vocab"), gv("reactive"))

    write("table_rule_types", pd.DataFrame(rows), "Rule-type discovery: doors opened by a switch instead of a key.")

    # Recovery -------------------------------------------------------------------------------------------------

    rc = pd.read_csv(f"{R}/recovery.csv")

    rows = []

    for a in rc.agent.unique():

        g = vals(rc, agent=a)

        rows.append({"Agent": a, "Success": fmt(g.success_rate), "Recovery": fmt(g.recovery_rate), "Failed tests/ep": f"{g.fail_events.mean():.2f}",

                     "Steps to success": fmt(g.mean_steps_success, p=1)})

    write("table_recovery", pd.DataFrame(rows), "Contradiction recovery on anti-mapped environments.")

    rv = lambda a, m: vals(rc, agent=a)[m].values

    for x, y in (("hyp_with_revision", "hyp_no_revision"), ("hyp_with_revision", "reactive"), ("hyp_no_prior", "hyp_with_revision"), ("hyp_with_revision", "psrl")):

        add("recovery", f"{x} vs {y}, success", rv(x, "success_rate"), rv(y, "success_rate"))

        add("recovery", f"{x} vs {y}, recovery", rv(x, "recovery_rate"), rv(y, "recovery_rate"))

    # Factorial ablation ---------------------------------------------------------------------------------------

    fx = pd.read_csv(f"{R}/factorial.csv")

    rows = []

    for d in (4, 5):

        for a in fx.agent.unique():

            g = vals(fx, difficulty=d, agent=a)

            rows.append({"Difficulty": d, "Variant": a, "Success": fmt(g.success_rate), "Redundant tests": f"{g.hyp_redundant.mean():.2f}"})

        fv = lambda a: vals(fx, difficulty=d, agent=a).success_rate.values

        add("factorial", f"d{d}: hyp_EV_l8 vs hyp_EV_l0", fv("hyp_EV_l8"), fv("hyp_EV_l0"))

        add("factorial", f"d{d}: psrl_S_l8 vs psrl_S_l0", fv("psrl_S_l8"), fv("psrl_S_l0"))

        add("factorial", f"d{d}: hyp_EV_l0 vs psrl_S_l0", fv("hyp_EV_l0"), fv("psrl_S_l0"))

        add("factorial", f"d{d}: hyp_EV_l8 vs psrl_S_l8", fv("hyp_EV_l8"), fv("psrl_S_l8"))

    write("table_factorial", pd.DataFrame(rows), "Factorial ablation: reward from expected value (EV) vs sampled binding (S) x information-gain weight.")

    # Master key (assumption violation) ------------------------------------------------------------------------

    mk = pd.read_csv(f"{R}/master_key.csv")

    rows = []

    for c in mk.condition.unique():

        for a in mk[mk.condition == c].agent.unique():

            g = vals(mk, condition=c, agent=a)

            rows.append({"Environments": c, "Agent": a, "Success": fmt(g.success_rate), "Steps to success": fmt(g.mean_steps_success, p=1)})

        gv = lambda a: vals(mk, condition=c, agent=a).success_rate.values

        add("master_key", f"{c}: hyp_nonexcl vs hypothesis", gv("hyp_nonexcl"), gv("hypothesis"))

        add("master_key", f"{c}: psrl_nonexcl vs psrl", gv("psrl_nonexcl"), gv("psrl"))

        add("master_key", f"{c}: hypothesis vs psrl", gv("hypothesis"), gv("psrl"))

    write("table_master_key", pd.DataFrame(rows), "Assumption violation: one key opens two doors.")

    # Theory check ---------------------------------------------------------------------------------------------

    th = pd.read_csv(f"{R}/theory.csv")

    rows = []

    for d in sorted(th.difficulty.unique()):

        for a in ("hypothesis", "psrl", "reactive"):

            g = th[(th.difficulty == d) & (th.agent == a)]

            rows.append({"Difficulty": int(d), "mean n": f"{g.mean_n.mean():.2f}", "Agent": a, "Observed failed tests": fmt(g.failed_door_tests),

                         "Elimination": f"{g.pred_elim.mean():.3f}", "Lose-shift": f"{g.pred_lose_shift.mean():.3f}"})

    write("table_theory", pd.DataFrame(rows), "Proposition 1: observed failed door tests vs predictions (n = reachable keys).")

    # Vault (second family) ------------------------------------------------------------------------------------

    vt = pd.read_csv(f"{R}/vault.csv")

    rows = []

    for c in vt.condition.unique():

        for a in agents_in(vt):

            g = vals(vt, condition=c, agent=a)

            if len(g):

                rows.append({"Codes": c, "Agent": a, "Success": fmt(g.success_rate), "Failed presses": fmt(g.resets, p=2), "Elimination prediction": f"{g.pred_elim_resets.mean():.2f}"})

        gv = lambda a: vals(vt, condition=c, agent=a).success_rate.values

        for o in ("vault_gru", "vault_ff", "reactive"):

            if o in set(vt.agent):

                add("vault", f"{c}: hypothesis vs {o}, success", gv("hypothesis"), gv(o))

    write("table_vault", pd.DataFrame(rows), "Combination vault: success and failed presses (train codes L in {4,5}, k in {2,3}).")

    # Training-seed spread of learned macro/generic agents ----------------------------------------------------

    rows = []

    for a in ("macro_ff", "macro_gru", "generic_ff", "generic_gru"):

        for d in (3, 4, 5):

            g = sc[(sc.agent == a) & (sc.difficulty == d)]

            if len(g):

                per = g.groupby(g.seed % (3 if a.startswith("macro") else 2)).success_rate.mean()

                rows.append({"Agent": a, "Difficulty": d, "Mean over training seeds": f"{per.mean():.3f}", "SD across training seeds": f"{per.std(ddof=1):.3f}", "Min": f"{per.min():.3f}", "Max": f"{per.max():.3f}"})

    if rows:

        write("table_trainseeds", pd.DataFrame(rows), "Spread across independently trained models of each learned agent.")

    # Training budgets -----------------------------------------------------------------------------------------

    rows = []

    for f in sorted(glob.glob("data/models/*_curve.json")):

        c = json.load(open(f)); name = os.path.basename(f).replace("_curve.json", "")

        if c:

            rows.append({"Model": name, "Steps/episodes": (f"{c[-1]['steps']:,} ep." if name.startswith("macro") else f"{c[-1]['steps']:,} steps"), "Final recent train success": f"{c[-1]['recent_success']:.3f}",

                         "Peak recent train success": f"{max(x['recent_success'] for x in c):.3f}", "Train time (min)": f"{c[-1]['time']/60:.0f}"})

    if rows:

        write("table_rl_training", pd.DataFrame(rows), "RL baseline training budgets and final training success (shaped training distribution).")

    t6 = pd.DataFrame(sig)

    t6["welch_p_holm"] = holm(t6.welch_p.fillna(1.0)); t6["mwu_p_holm"] = holm(t6.mwu_p.fillna(1.0))

    write("table6_significance", t6.round(4), "Pre-declared significance tests (diff = first minus second; Holm-corrected over the original test family).")

    # LaTeX versions -------------------------------------------------------------------------------------------

    for f in glob.glob(f"{T}/*.csv"):

        pd.read_csv(f).to_latex(f.replace(".csv", ".tex"), index=False, escape=True)

    json.dump({"n_tests": len(t6)}, open(f"{R}/results_summary.json", "w"))





if __name__ == "__main__":

    main()
