import os, sys, shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch

from evaluation.significance import bootstrap_ci

from environment.generator import EnvironmentConfig, generate

from environment.renderer import render

from visualization.trajectories import load_traces, pair_series



R, F = "data/results", "visualization/figures"

os.makedirs(F, exist_ok=True); os.makedirs("paper/figures", exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})

AGN = {"random": "Random", "reactive": "Reactive", "psrl": "PSRL", "hypothesis": "Hypothesis", "hyp_transfer": "Hyp.+transfer", "hyp_adaptive": "Hyp.+adaptive", "macro_ff": "Macro-FF", "macro_gru": "Macro-GRU", "macro_gruB": "Macro-GRU+belief", "generic_ff": "Generic-FF", "generic_gru": "Generic-GRU", "ppo_ff": "PPO-FF", "ppo_gru": "PPO-GRU", "vault_ff": "PPO-FF", "vault_gru": "PPO-GRU"}

COL = {"random": "#999999", "reactive": "#d95f02", "psrl": "#e7298a", "hypothesis": "#1b9e77", "hyp_transfer": "#7570b3", "ppo_ff": "#a6d854", "ppo_gru": "#a6cee3", "macro_ff": "#66a61e", "macro_gru": "#1f78b4", "generic_ff": "#e6ab02", "generic_gru": "#a6761d", "vault_ff": "#66a61e", "vault_gru": "#1f78b4"}





def save(fig, name):

    for ext in ("png", "pdf", "svg"):

        fig.savefig(f"{F}/{name}.{ext}", dpi=300, bbox_inches="tight")

        shutil.copy(f"{F}/{name}.{ext}", f"paper/figures/{name}.{ext}")

    plt.close(fig)





def ci(x):

    lo, hi = bootstrap_ci(x); m = np.nanmean(x)

    return m, m - lo, hi - m





def boxes(ax, items, y, w=2.2, h=0.9, gap=0.5, fc="#e8f1fa"):

    xs = []

    for i, t in enumerate(items):

        x = i * (w + gap)

        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="k"))

        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)

        xs.append(x)

        if i: ax.annotate("", (x, y + h / 2), (x - gap, y + h / 2), arrowprops=dict(arrowstyle="->"))

    return xs





def fig1_2():

    fig, ax = plt.subplots(figsize=(10, 2.6)); ax.axis("off"); ax.set_xlim(-.2, 15); ax.set_ylim(-0.5, 3)

    boxes(ax, ["Observation", "State\nrepresentation", "Hypothesis\nmanager", "Candidate\nbindings", "Utility\nselector", "Action"], 1.2)

    ax.text(7.2, 0.5, "episodic memory  ·  U(a) = R(a) + λ·IG(a) − μ·cost(a)  ·  α/β confidence updates  ·  rejection after repeated contradiction", ha="center", fontsize=8)

    ax.set_title("Hypothesis-driven agent architecture"); save(fig, "fig1_architecture")

    fig, ax = plt.subplots(figsize=(8.5, 2.4)); ax.axis("off"); ax.set_xlim(-.2, 13); ax.set_ylim(0, 2.6)

    boxes(ax, ["Environment", "Observation", "Agent +\nhypotheses", "Action", "Outcome +\nfeedback", "Hypothesis\nupdate"], 1.0, fc="#fdebd0")

    ax.annotate("", (0, 0.95), (12.9, 0.95), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.25", ls="--"))

    ax.text(6.4, 0.1, "no human in the loop; hidden rules are never exposed", ha="center", fontsize=8)

    ax.set_title("Autonomous interaction loop"); save(fig, "fig2_interaction_loop")





def fig3():

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))

    for ax, d in zip(axs, (1, 3, 5)):

        render(generate(EnvironmentConfig(seed=11, difficulty=d)), ax, f"difficulty {d}")

    fig.suptitle("Example generated environments (K key, coloured block door, hatched cell chasm, circle switch, brown block, x trap, * target)", fontsize=8)

    save(fig, "fig3_example_environment")





def fig45():

    sc = pd.read_csv(f"{R}/scaling.csv")

    for metric, name, yl, title in (("success_rate", "fig4_success_vs_difficulty", "Success rate", "Success rate vs difficulty"),

                                    ("mean_steps_success", "fig5_interactions_vs_difficulty", "Interactions to solve (successful episodes)", "Interactions required vs difficulty")):

        fig, ax = plt.subplots(figsize=(4.6, 3.2))

        for a in [x for x in ("random", "reactive", "psrl", "hypothesis", "macro_ff", "macro_gru", "ppo_ff", "ppo_gru") if x in set(sc.agent)]:

            pts = [ci(sc[(sc.agent == a) & (sc.difficulty == d)][metric].dropna().values) if sc[(sc.agent == a) & (sc.difficulty == d)][metric].notna().any() else (np.nan, 0, 0) for d in range(1, 6)]

            m, lo, hi = zip(*pts)

            ax.errorbar(range(1, 6), m, yerr=[lo, hi], marker="o", capsize=3, color=COL[a], label=AGN.get(a, a))

        ax.set_xlabel("difficulty"); ax.set_ylabel(yl); ax.set_title(title); ax.set_xticks(range(1, 6)); ax.legend(frameon=False)

        save(fig, name)





def fig6():
    g, d30 = generalization_with_d30()
    regs = list(g.condition.unique())
    ags = [x for x in ("reactive", "psrl", "hypothesis", "hyp_adaptive", "hyp_transfer",
                       "macro_ff", "macro_gru", "macro_gruB") if x in set(g.agent)]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.3))
    w = 0.8 / max(len(ags), 1)

    for j, a in enumerate(ags):
        vals = []
        for r in regs:
            q = g[(g.condition == r) & (g.agent == a) & (g.split == "test")]
            vals.append(ci(q.sort_values("seed").success_rate.values) if not q.empty else (np.nan, 0, 0))
        m, lo, hi = zip(*vals)
        axs[0].bar(np.arange(len(regs)) + j * w, m, w, yerr=[lo, hi],
                   capsize=2, color=COL.get(a), label=AGN.get(a, a))

    gap_regs = [r for r in regs if r != "D_rule_permutation"]
    gw = 0.8 / max(len(ags), 1)
    for j, a in enumerate(ags):
        vals = []
        for r in gap_regs:
            q = g[(g.condition == r) & (g.agent == a)]
            te = q[q.split == "test"].sort_values("seed").success_rate.values
            tr = q[q.split == "train_heldout"].sort_values("seed").success_rate.values
            vals.append(ci(tr - te) if len(te) and len(tr) == len(te) else (np.nan, 0, 0))
        m, lo, hi = zip(*vals)
        axs[1].bar(np.arange(len(gap_regs)) + j * gw, m, gw,
                   yerr=[lo, hi], capsize=2, color=COL.get(a),
                   label=AGN.get(a, a))

    axs[0].set_ylabel("success rate")
    axs[0].set_xticks(np.arange(len(regs)) + 0.4 - w / 2)
    axs[0].set_xticklabels(["D*" if r == "D_rule_permutation" else r.split("_")[0] for r in regs])
    axs[0].set_title("Held-out test success rate")
    axs[0].legend(frameon=False, fontsize=6, ncol=2)
    axs[1].set_xticks(np.arange(len(gap_regs)) + 0.4 - gw / 2)
    axs[1].set_xticklabels([r.split("_")[0] for r in gap_regs])
    axs[1].set_title("Generalization gap (train minus test)")
    axs[1].axhline(0, c="k", lw=.5)
    if d30 is not None:
        fig.text(0.5, 0.01, "* D = 30-model rule-permutation re-evaluation; test-only",
                 ha="center", fontsize=7)
    save(fig, "fig6_generalization")


def fig78():

    tr = load_traces()

    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    t = tr["hyp_no_prior"]

    for lab, (x, y) in pair_series(t["trace"], {k: v for k, v in t["truth"].items()}).items():

        ax.step(x, y, where="post", ls="-" if lab.startswith("TRUE") else "--", label=lab, lw=1.5)

    ax.set_xlabel("environment step"); ax.set_ylabel("P(key opens door)"); ax.set_title("Hypothesis confidence over time (no prior)")

    ax.legend(frameon=False, fontsize=6); save(fig, "fig7_hypothesis_confidence")

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)

    for ax, n in zip(axs, ("hyp_with_revision", "hyp_no_revision")):

        t = tr[n]

        for lab, (x, y) in pair_series(t["trace"], t["truth"]).items():

            ax.step(x, y, where="post", ls="-" if lab.startswith("TRUE") else "--", label=lab, lw=1.5)

        ax.set_title(f"{n} ({'solved' if t['success'] else 'failed'}, {t['steps']} steps)"); ax.set_xlabel("environment step")

    axs[0].set_ylabel("P(key opens door)"); axs[0].legend(frameon=False, fontsize=6)

    fig.suptitle("Contradiction and recovery: prior says same-colour, environment says otherwise", fontsize=9)

    save(fig, "fig8_contradiction_recovery")

    rc = pd.read_csv(f"{R}/recovery.csv")

    fig, axs = plt.subplots(1, 2, figsize=(7, 3))

    order = ["reactive", "psrl", "hyp_no_revision", "hyp_with_revision", "hyp_no_prior"]

    for ax, m, tt in ((axs[0], "success_rate", "Success rate"), (axs[1], "recovery_rate", "Recovery rate")):

        vals = [ci(rc[rc.agent == a][m].dropna().values) for a in order]

        mm, lo, hi = zip(*vals); ax.bar(range(5), mm, yerr=[lo, hi], capsize=3, color=["#d95f02", "#e7298a", "#cc4c02", "#1b9e77", "#66a61e"])

        ax.set_xticks(range(5)); ax.set_xticklabels([o.replace("hyp_", "hyp\n") for o in order], fontsize=7); ax.set_title(tt)

    save(fig, "fig8b_recovery_summary")





def fig9_10():

    ab = pd.read_csv(f"{R}/ablation.csv")

    names = ["full_memory", "fixed_memory_3", "no_memory", "info_driven", "random_explore", "revision_on", "revision_off", "reactive", "psrl"]

    fig, ax = plt.subplots(figsize=(7, 3.0))

    m, lo, hi = zip(*[ci(ab[ab.agent == n].success_rate.values) for n in names])

    grp = ["#1b9e77"] * 3 + ["#7570b3"] * 2 + ["#e7298a"] * 2 + ["#d95f02", "#e7298a"]

    ax.bar(range(len(names)), m, yerr=[lo, hi], capsize=2, color=grp)

    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7); ax.set_ylabel("success rate")

    ax.set_title("Ablations at difficulty 4"); save(fig, "fig9_ablation")

    lams = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    fig, axs = plt.subplots(1, 3, figsize=(10, 2.9))

    for sfx, lab, c in (("", "d=4", "#1b9e77"), ("_d5", "d=5", "#d95f02")):

        m, lo, hi = zip(*[ci(ab[ab.agent == f"lambda_{l}{sfx}"].success_rate.values) for l in lams])

        axs[0].errorbar(range(len(lams)), m, yerr=[lo, hi], marker="o", capsize=3, label=lab, color=c)

    axs[0].set_xticks(range(len(lams))); axs[0].set_xticklabels([str(l) for l in lams]); axs[0].set_xlabel("λ"); axs[0].set_ylabel("success rate"); axs[0].legend(frameon=False)

    axs[0].set_title("Success vs λ")

    axs[1].plot(range(len(lams)), [ab[ab.agent == f"lambda_{l}"].explore_ratio.mean() for l in lams], "o-")

    axs[1].set_xticks(range(len(lams))); axs[1].set_xticklabels([str(l) for l in lams]); axs[1].set_xlabel("λ"); axs[1].set_title("Exploration share")

    mus = [0.0, 0.05, 0.2]

    m, lo, hi = zip(*[ci(ab[ab.agent == f"mu_{u}"].success_rate.values) for u in mus])

    axs[2].errorbar(range(3), m, yerr=[lo, hi], marker="o", capsize=3); axs[2].set_xticks(range(3)); axs[2].set_xticklabels([str(u) for u in mus])

    axs[2].set_xlabel("μ (action cost)"); axs[2].set_title("Sensitivity to μ")

    save(fig, "fig10_explore_exploit")





def fig11_12():

    import glob, json

    cur = {}

    for f in sorted(glob.glob("data/models/*_curve.json")):

        c = json.load(open(f))

        if c: cur[os.path.basename(f).replace("_curve.json", "")] = c

    if cur:

        fig, ax = plt.subplots(figsize=(6, 3.2))

        for k, c in cur.items():

            ax.plot([x["steps"] / 1e6 for x in c], [x["recent_success"] for x in c], label=k, lw=1.2)

        ax.set_xlabel("training steps (millions)"); ax.set_ylabel("success (last 200 train episodes)"); ax.legend(frameon=False, fontsize=6, ncol=2)

        ax.set_title("PPO baselines: training curves (shaped training distribution)"); save(fig, "fig11_rl_training")

    rt = pd.read_csv(f"{R}/rule_types.csv")

    conds = list(rt.condition.unique()); ags = list(rt.agent.unique())

    fig, ax = plt.subplots(figsize=(7, 3))

    w = 0.8 / len(ags)

    for j, a in enumerate(ags):

        vs = [ci(rt[(rt.condition == c) & (rt.agent == a)].success_rate.values) for c in conds]

        m, lo, hi = zip(*vs); ax.bar(np.arange(len(conds)) + j * w, m, w, yerr=[lo, hi], capsize=2, label={"hyp_closed_vocab": "Hyp. (closed)", "hyp_open_vocab": "Hyp. (open)", "psrl": "PSRL", "psrl_open": "PSRL (open)", "reactive": "Reactive"}.get(a, a))

    ax.set_xticks(np.arange(len(conds)) + 0.4 - w / 2); ax.set_xticklabels(conds, fontsize=8); ax.set_ylabel("success rate"); ax.legend(frameon=False, fontsize=6, ncol=2)

    ax.set_title("Rule-type discovery (switch-operated doors)"); save(fig, "fig12_rule_types")





def fig13_14():

    fx = pd.read_csv(f"{R}/factorial.csv"); mk = pd.read_csv(f"{R}/master_key.csv")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.0))

    names = ["psrl_S_l0", "psrl_S_l0.5", "psrl_S_l8", "hyp_EV_l0", "hyp_EV_l0.5", "hyp_EV_l8"]

    for d, c, off in ((4, "#1b9e77", 0), (5, "#d95f02", 0.4)):

        vs_ = [ci(fx[(fx.agent == n) & (fx.difficulty == d)].success_rate.values) for n in names]

        m, lo, hi = zip(*vs_); axs[0].bar(np.arange(6) + off - 0.2, m, 0.4, yerr=[lo, hi], capsize=2, color=c, label=f"d={d}")

    axs[0].set_xticks(range(6)); axs[0].set_xticklabels(["PSRL\nλ=0", "PSRL\nλ=.5", "PSRL\nλ=8", "Hyp\nλ=0", "Hyp\nλ=.5", "Hyp\nλ=8"], fontsize=7)

    axs[0].set_ylabel("success rate"); axs[0].legend(frameon=False, fontsize=7); axs[0].set_title("Reward form x information bonus")

    conds = ["master_key_d3", "master_key_d4", "exclusive_control_d3", "exclusive_control_d4"]; ags = ["reactive", "psrl", "psrl_nonexcl", "hypothesis", "hyp_nonexcl"]

    w = 0.8 / len(ags)

    for j, a in enumerate(ags):

        vs_ = [ci(mk[(mk.condition == c) & (mk.agent == a)].success_rate.values) for c in conds]

        m, lo, hi = zip(*vs_); axs[1].bar(np.arange(4) + j * w, m, w, yerr=[lo, hi], capsize=1.5, label=a)

    axs[1].set_xticks(np.arange(4) + 0.4 - w / 2); axs[1].set_xticklabels(["master\nd3", "master\nd4", "control\nd3", "control\nd4"], fontsize=7)

    axs[1].legend(frameon=False, fontsize=6, ncol=2); axs[1].set_title("Exclusivity violated (master key) vs control")

    save(fig, "fig13_factorial_masterkey")

    th = pd.read_csv(f"{R}/theory.csv")

    fig, ax = plt.subplots(figsize=(5.2, 3.0))

    ds = sorted(th.difficulty.unique()); w = 0.16

    for j, (a, lab, c) in enumerate((("hypothesis", "Hypothesis", COL["hypothesis"]), ("psrl", "PSRL", COL["psrl"]), ("reactive", "Reactive", COL["reactive"]))):

        v = [th[(th.difficulty == d) & (th.agent == a)].failed_door_tests.mean() for d in ds]

        ax.bar(np.arange(len(ds)) + j * w, v, w, color=c, label=lab)

    ax.bar(np.arange(len(ds)) + 3 * w, [th[th.difficulty == d].pred_elim.mean() for d in ds], w, color="k", label="Elimination (theory)")

    ax.bar(np.arange(len(ds)) + 4 * w, [th[th.difficulty == d].pred_lose_shift.mean() for d in ds], w, color="#999999", label="Lose-shift (theory)")

    ax.set_xticks(np.arange(len(ds)) + 2 * w); ax.set_xticklabels([f"d={d}\nn={th[th.difficulty == d].mean_n.mean():.1f}" for d in ds], fontsize=8)

    ax.set_ylabel("failed door tests / episode"); ax.legend(frameon=False, fontsize=6); ax.set_title("Proposition 1: observed vs predicted")

    save(fig, "fig14_theory")





def fig15_16():
    sc = pd.read_csv(f"{R}/scaling.csv")
    g, d30 = generalization_with_d30()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.0))

    ags = [a for a in ("random", "reactive", "psrl", "hypothesis", "macro_ff", "macro_gru",
                       "generic_ff", "generic_gru", "ppo_gru") if a in set(sc.agent)]
    for a in ags:
        m, lo, hi = zip(*[ci(sc[(sc.agent == a) & (sc.difficulty == d)].success_rate.values)
                          for d in range(1, 6)])
        axs[0].errorbar(range(1, 6), m, yerr=[lo, hi], marker="o", ms=3, capsize=2,
                        color=COL.get(a), label=AGN.get(a, a), lw=1.2)
    axs[0].set_xlabel("difficulty"); axs[0].set_ylabel("success rate")
    axs[0].set_xticks(range(1, 6)); axs[0].legend(frameon=False, fontsize=6, ncol=2)
    axs[0].set_title("(a) Success vs difficulty")

    regs = list(g.condition.unique())
    ga = [a for a in ("reactive", "psrl", "hypothesis", "hyp_adaptive", "hyp_transfer",
                      "macro_ff", "macro_gru", "macro_gruB") if a in set(g.agent)]
    w = 0.8 / max(len(ga), 1)
    for j, a in enumerate(ga):
        vs_ = []
        for r in regs:
            q = g[(g.condition == r) & (g.agent == a) & (g.split == "test")]
            vs_.append(ci(q.sort_values("seed").success_rate.values)
                        if not q.empty else (np.nan, 0, 0))
        m, lo, hi = zip(*vs_)
        axs[1].bar(np.arange(len(regs)) + j * w, m, w, yerr=[lo, hi],
                   capsize=1.5, color=COL.get(a), label=AGN.get(a, a))
    axs[1].set_xticks(np.arange(len(regs)) + 0.4 - w / 2)
    axs[1].set_xticklabels(["D*" if r == "D_rule_permutation" else r.split("_")[0] for r in regs])
    axs[1].legend(frameon=False, fontsize=5.5, ncol=2)
    axs[1].set_title("(b) Held-out success by regime")
    if d30 is not None:
        fig.text(0.75, 0.01, "* D = 30-model rule-permutation re-evaluation",
                 ha="center", fontsize=7)
    save(fig, "fig15_main")

    fx = pd.read_csv(f"{R}/factorial.csv"); mk = pd.read_csv(f"{R}/master_key.csv"); vt = pd.read_csv(f"{R}/vault.csv")

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9))

    names = ["psrl_S_l0", "psrl_S_l8", "hyp_EV_l0", "hyp_EV_l8"]

    for d, c, off in ((4, "#1b9e77", -0.2), (5, "#d95f02", 0.2)):

        m, lo, hi = zip(*[ci(fx[(fx.agent == n) & (fx.difficulty == d)].success_rate.values) for n in names])

        axs[0].bar(np.arange(4) + off, m, 0.4, yerr=[lo, hi], capsize=2, color=c, label=f"d={d}")

    axs[0].set_xticks(range(4)); axs[0].set_xticklabels(["Sampled\nλ=0", "Sampled\nλ=8", "Exp.value\nλ=0", "Exp.value\nλ=8"], fontsize=7); axs[0].legend(frameon=False, fontsize=7)

    axs[0].set_ylabel("success rate"); axs[0].set_title("(a) Reward form x info bonus", fontsize=9)

    conds = ["master_key_d3", "master_key_d4", "exclusive_control_d4"]; ags = ["reactive", "psrl", "hypothesis", "hyp_nonexcl"]; w = 0.8 / len(ags)

    for j, a in enumerate(ags):

        m, lo, hi = zip(*[ci(mk[(mk.condition == c) & (mk.agent == a)].success_rate.values) for c in conds])

        axs[1].bar(np.arange(3) + j * w, m, w, yerr=[lo, hi], capsize=1.5, label={"hyp_nonexcl": "Hyp. (no excl.)"}.get(a, AGN.get(a, a)))

    axs[1].set_xticks(np.arange(3) + 0.4 - w / 2); axs[1].set_xticklabels(["master\nd3", "master\nd4", "control\nd4"], fontsize=7); axs[1].legend(frameon=False, fontsize=6)

    axs[1].set_title("(b) One key opens several doors", fontsize=9)

    va = [a for a in ("random", "reactive", "hypothesis", "vault_ff", "vault_gru") if a in set(vt.agent)]; w = 0.8 / len(va)

    for j, a in enumerate(va):

        m, lo, hi = zip(*[ci(vt[(vt.condition == c) & (vt.agent == a)].success_rate.values) for c in ("in_dist", "larger")])

        axs[2].bar(np.arange(2) + j * w, m, w, yerr=[lo, hi], capsize=1.5, color=COL.get(a), label=AGN.get(a, a))

    axs[2].set_xticks(np.arange(2) + 0.4 - w / 2); axs[2].set_xticklabels(["small codes\n(train)", "larger codes"], fontsize=8); axs[2].legend(frameon=False, fontsize=6)

    axs[2].set_title("(c) Second family: combination vault", fontsize=9); save(fig, "fig16_ablate")





if __name__ == "__main__":

    for f in (fig1_2, fig3, fig45, fig6, fig78, fig9_10, fig11_12, fig13_14, fig15_16):

        f()

    print("figures saved to", F)
