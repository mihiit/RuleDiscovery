"""Publication figures (vector PDF for LaTeX + 300-dpi PNG). Sized for the 12.2 cm LNCS text block, 8-9 pt text, colour-blind-safe palette,
distinct markers, legends outside the data area. Intervals are 95% percentile bootstraps over *units* (trained models for learned agents,
evaluation seeds otherwise); individual trained models are drawn as dots so that between-run variability is visible."""
import os, sys, json, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from evaluation.hierarchical import units, boot_mean_ci, is_learned
from visualization.trajectories import load_traces, pair_series

IEEE = os.environ.get("FIG_MODE") == "ieee"      # FIG_MODE=ieee: sizes for IEEEtran two-column (3.4 in column / 7.0 in page)
R, FIG = "data/results", ("paper/figures_ieee" if IEEE else "paper/figures")
os.makedirs(FIG, exist_ok=True); os.makedirs("visualization/figures", exist_ok=True)
W = 4.8   # inches = 12.2 cm
SZ = {"difficulty": (3.4, 2.55) if IEEE else (W, 2.55), "regimes": (7.0, 2.7) if IEEE else (W, 2.55), "assumption": (3.4, 2.4) if IEEE else (W, 2.35),
      "vault": (7.0, 2.4) if IEEE else (W, 2.35), "noise": (3.4, 3.0) if IEEE else (W, 2.45), "infobonus": (7.0, 2.3) if IEEE else (W * 1.12, 2.25)}
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "lines.linewidth": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
# Okabe-Ito palette
STYLE = {
    "random": ("Random", "#8c8c8c", "x", ":"), "reactive": ("Reactive", "#D55E00", "s", "-"), "psrl": ("PSRL", "#CC79A7", "^", "-"),
    "hypothesis": ("Hypothesis", "#009E73", "o", "-"), "hyp_adaptive": ("Hypothesis-adaptive", "#005A40", "D", "-"),
    "hyp_nonexcl": ("Hypothesis (no exclusivity)", "#56B4E9", "v", "-"), "hyp_robust": ("Hypothesis-robust", "#8FD14F", "h", "-"),
    "macro_ff": ("Macro-FF", "#E69F00", "P", "--"), "macro_gru": ("Macro-GRU", "#0072B2", "X", "--"), "macro_gruB": ("Macro-GRU+belief", "#6A3D9A", "*", "--"),
    "generic_gru": ("Generic-GRU", "#8B6914", "<", "-."), "generic_ff": ("Generic-FF", "#c9a227", ">", "-."),
    "ppo_gru": ("Primitive PPO (GRU)", "#7a7a7a", "1", ":"), "vault_ff": ("PPO-FF", "#E69F00", "P", "--"), "vault_gru": ("PPO-GRU", "#0072B2", "X", "--"),
}
rd = lambda n: pd.read_csv(f"{R}/{n}.csv")


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight")
    shutil.copy(f"{FIG}/{name}.pdf", f"visualization/figures/{name}.pdf"); shutil.copy(f"{FIG}/{name}.png", f"visualization/figures/{name}.png")
    plt.close(fig)


def stat(df, agent, dist="main", col="success_rate"):
    u = units(df, agent, dist, col)
    if len(u) == 0:
        return None
    lo, hi = boot_mean_ci(u) if len(u) > 1 else (u.mean(), u.mean())
    return u.mean(), lo, hi, u


def fig_difficulty():
    sc = rd("scaling")
    order = ["reactive", "psrl", "hypothesis", "hyp_adaptive", "macro_ff", "macro_gru", "macro_gruB", "generic_gru", "ppo_gru"]
    fig, ax = plt.subplots(figsize=SZ["difficulty"])
    for a in order:
        if a not in set(sc.agent):
            continue
        lab, c, mk, ls = STYLE[a]
        pts = [(d, stat(sc[sc.difficulty == d], a)) for d in range(1, 6)]
        pts = [(d, s) for d, s in pts if s]
        x = np.array([d for d, _ in pts], float) + (list(STYLE).index(a) - 6) * 0.012
        m = np.array([s[0] for _, s in pts]); lo = np.array([s[1] for _, s in pts]); hi = np.array([s[2] for _, s in pts])
        ax.errorbar(x, m, yerr=[m - lo, hi - m], color=c, marker=mk, ls=ls, ms=5, capsize=2, elinewidth=0.9, label=lab)
    ax.set_xlabel("difficulty level"); ax.set_ylabel("success rate"); ax.set_xticks(range(1, 6)); ax.set_ylim(-0.02, 1.04)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2 if IEEE else 3, frameon=False, columnspacing=1.2, handlelength=2.4)
    save(fig, "fig_difficulty")


def fig_regimes():
    g = rd("generalization"); g = g[g.split == "test"]
    regs = [("A_interpolation", "A"), ("B_compositional", "B"), ("C_chain_length", "C"), ("C2_size_only", "C2"), ("D_rule_permutation", "D"), ("E_novel_combination", "E")]
    order = ["reactive", "psrl", "hypothesis", "hyp_adaptive", "macro_ff", "macro_gru", "macro_gruB"]
    order = [a for a in order if a in set(g.agent)]
    fig, ax = plt.subplots(figsize=SZ["regimes"]); bw = 0.8 / len(order)
    for j, a in enumerate(order):
        lab, c, mk, ls = STYLE[a]; first = True
        for i, (rn, _) in enumerate(regs):
            s = stat(g[g.condition == rn], a, rn)
            if not s:
                continue
            x = i + (j - (len(order) - 1) / 2) * bw
            ax.bar(x, s[0], bw * 0.92, color=c, alpha=0.9, label=lab if first else None, edgecolor="white", linewidth=0.4); first = False
            ax.errorbar(x, s[0], yerr=[[s[0] - s[1]], [s[2] - s[0]]], color="k", capsize=1.5, lw=0.8)
            if is_learned(a) and len(s[3]) > 1:   # one dot per independently trained model
                ax.scatter(np.full(len(s[3]), x) + np.linspace(-bw * 0.25, bw * 0.25, len(s[3])), s[3], s=5, c="k", alpha=0.55, zorder=4, linewidths=0)
    ax.set_xticks(range(len(regs))); ax.set_xticklabels([r[1] for r in regs]); ax.set_xlabel("generalization regime, held-out test distribution (dots: individual trained models)")
    ax.set_ylabel("success rate"); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=7 if IEEE else 4, frameon=False, columnspacing=1.0, handlelength=1.2)
    save(fig, "fig_regimes")


def fig_assumption():
    mk = rd("master_key")
    conds = [("master_key_d3", "master key\n(d=3)" if IEEE else "one key, several doors\n(d=3)"), ("master_key_d4", "master key\n(d=4)" if IEEE else "one key, several doors\n(d=4)"),
             ("exclusive_control_d3", "control\n(d=3)" if IEEE else "exclusive control\n(d=3)"), ("exclusive_control_d4", "control\n(d=4)" if IEEE else "exclusive control\n(d=4)")]
    order = [a for a in ("hypothesis", "hyp_adaptive", "hyp_nonexcl", "psrl", "reactive") if a in set(mk.agent)]
    fig, ax = plt.subplots(figsize=SZ["assumption"]); bw = 0.8 / len(order)
    for j, a in enumerate(order):
        lab, c, _, _ = STYLE[a]
        for i, (cn, _) in enumerate(conds):
            s = stat(mk[mk.condition == cn], a)
            x = i + (j - (len(order) - 1) / 2) * bw
            ax.bar(x, s[0], bw * 0.92, color=c, label=lab if i == 0 else None, edgecolor="white", linewidth=0.4)
            ax.errorbar(x, s[0], yerr=[[s[0] - s[1]], [s[2] - s[0]]], color="k", capsize=1.5, lw=0.8)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([c[1] for c in conds], fontsize=7.5); ax.set_ylabel("success rate"); ax.set_ylim(0, 1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2 if IEEE else 3, frameon=False, columnspacing=1.0, handlelength=1.2)
    save(fig, "fig_assumption")


def fig_vault():
    vt = rd("vault")
    order = [a for a in ("random", "reactive", "hypothesis", "vault_ff", "vault_gru") if a in set(vt.agent)]
    fig, axs = plt.subplots(1, 2, figsize=SZ["vault"], gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]; bw = 0.8 / len(order)
    for j, a in enumerate(order):
        lab, c, _, _ = STYLE[a]
        for i, cn in enumerate(("in_dist", "larger")):
            s = stat(vt[vt.condition == cn], a)
            x = i + (j - (len(order) - 1) / 2) * bw
            ax.bar(x, s[0], bw * 0.92, color=c, label=lab if i == 0 else None, edgecolor="white", linewidth=0.4)
            ax.errorbar(x, s[0], yerr=[[s[0] - s[1]], [s[2] - s[0]]], color="k", capsize=1.5, lw=0.8)
            if is_learned(a) and len(s[3]) > 1:
                ax.scatter(np.full(len(s[3]), x) + np.linspace(-bw * 0.25, bw * 0.25, len(s[3])), s[3], s=5, c="k", alpha=0.55, zorder=4, linewidths=0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["small codes\n(training sizes)", "larger codes\n(unseen sizes)"]); ax.set_ylabel("success rate"); ax.set_ylim(0, 1.05)
    ax.set_title("(a) success", loc="left")
    ax = axs[1]; h = vt[vt.agent == "hypothesis"]
    obs = [h[h.condition == c].resets.mean() for c in ("in_dist", "larger")]; pred = [h[h.condition == c].pred_elim_resets.mean() for c in ("in_dist", "larger")]
    x = np.arange(2)
    ax.bar(x - 0.19, pred, 0.36, color="white", edgecolor="k", hatch="//", label="Proposition 1 (elimination)")
    ax.bar(x + 0.19, obs, 0.36, color=STYLE["hypothesis"][1], label="Hypothesis agent")
    ax.set_xticks(x); ax.set_xticklabels(["small", "larger"]); ax.set_ylabel("failed presses per episode"); ax.set_title("(b) cost of failed tests", loc="left"); ax.set_ylim(0, max(obs + pred) * 1.45)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.07), columnspacing=0.9, handlelength=1.1, fontsize=7.5)
    fig.tight_layout(); save(fig, "fig_vault")


def fig_infobonus():
    ab, fx = rd("ablation"), rd("factorial")
    fig, axs = plt.subplots(1, 3, figsize=SZ["infobonus"], gridspec_kw=dict(width_ratios=[1, 0.75, 1.35] if IEEE else [1.15, 1, 1]))
    ax = axs[0]
    lams = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    for d, suf, c, mk in ((4, "", "#009E73", "o"), (5, "_d5", "#D55E00", "s")):
        pts = [stat(ab, f"lambda_{l}{suf}") for l in lams]
        m = np.array([p[0] for p in pts]); lo = np.array([p[1] for p in pts]); hi = np.array([p[2] for p in pts])
        ax.errorbar(range(len(lams)), m, yerr=[m - lo, hi - m], color=c, marker=mk, ms=4.5, capsize=2, label=f"difficulty {d}")
    ax.set_xticks(range(len(lams))); ax.set_xticklabels([f"{l:g}" for l in lams], fontsize=7); ax.set_xlabel(r"information weight $\lambda$"); ax.set_ylabel("success rate")
    ax.set_title("(a)", loc="left"); ax.legend(frameon=False, loc="lower right")
    ax = axs[1]
    pts = [stat(ab, f"mu_{m_}") for m_ in (0.0, 0.05, 0.2)]
    m = np.array([p[0] for p in pts]); lo = np.array([p[1] for p in pts]); hi = np.array([p[2] for p in pts])
    ax.errorbar(range(3), m, yerr=[m - lo, hi - m], color="#0072B2", marker="o", ms=4.5, capsize=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["0", "0.05", "0.2"]); ax.set_xlabel(r"action cost $\mu$"); ax.set_title("(b) difficulty 4", loc="left"); ax.set_ylim(0, 1.05)
    ax = axs[2]
    cells = [("psrl_S_l0", "sampled\n$\\lambda{=}0$"), ("psrl_S_l8", "sampled\n$\\lambda{=}8$"), ("hyp_EV_l0", "expected\n$\\lambda{=}0$"), ("hyp_EV_l8", "expected\n$\\lambda{=}8$")]
    for i, (a, _) in enumerate(cells):
        for d, off, c in ((4, -0.19, "#009E73"), (5, 0.19, "#D55E00")):
            s = stat(fx[fx.difficulty == d], a)
            ax.bar(i + off, s[0], 0.36, color=c, label=f"difficulty {d}" if i == 0 else None)
            ax.errorbar(i + off, s[0], yerr=[[s[0] - s[1]], [s[2] - s[0]]], color="k", capsize=1.5, lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels([c[1] for c in cells], fontsize=6.5 if not IEEE else 7); ax.set_title("(c) reward form $\\times$ bonus", loc="left"); ax.set_ylim(0, 1.05)
    fig.tight_layout(w_pad=0.8); save(fig, "fig_infobonus")


def fig_noise():
    nz = rd("noise")
    order = [a for a in ("reactive", "psrl", "hypothesis", "hyp_robust", "macro_gru", "macro_gruB") if a in set(nz.agent)]
    slips = sorted(nz.slip.unique())
    fig, axs = plt.subplots(1, 2, figsize=SZ["noise"], sharey=True)
    for ax, d in zip(axs, (3, 4)):
        for a in order:
            lab, c, mk, ls = STYLE[a]
            pts = [stat(nz[(nz.difficulty == d) & (nz.slip == sl)], a) for sl in slips]
            m = np.array([p[0] for p in pts]); lo = np.array([p[1] for p in pts]); hi = np.array([p[2] for p in pts])
            ax.errorbar(slips, m, yerr=[m - lo, hi - m], color=c, marker=mk, ls=ls, ms=4.5, capsize=2, elinewidth=0.8, label=lab)
        ax.set_xlabel("lock-jam probability"); ax.set_xticks(slips); ax.set_title(f"difficulty {d}", loc="left"); ax.set_ylim(0, 1.04)
    axs[0].set_ylabel("success rate")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2 if IEEE else 3, frameon=False, bbox_to_anchor=(0.5, -0.12 if not IEEE else -0.2), fontsize=7.5)
    fig.tight_layout(); save(fig, "fig_noise")


def fig_recovery():
    tr = load_traces()
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    for ax, n, title in zip(axs, ("hyp_with_revision", "hyp_no_revision"), ("(a) with revision", "(b) without revision")):
        t = tr[n]
        for lab, (x, y) in pair_series(t["trace"], t["truth"]).items():
            true = lab.startswith("TRUE")
            ax.step(x, y, where="post", ls="-" if true else "--", color="#009E73" if true else "#D55E00", lw=1.5, alpha=0.95)
        ax.set_title(f"{title}: {'solved' if t['success'] else 'failed'} in {t['steps']} steps", loc="left", fontsize=8); ax.set_xlabel("environment step")
    axs[0].set_ylabel("P(key opens door)")
    axs[1].legend(handles=[Line2D([0], [0], color="#D55E00", ls="--", lw=1.5, label="same-colour prior (wrong)"), Line2D([0], [0], color="#009E73", lw=1.5, label="true binding")],
                  frameon=False, loc="center right", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig_recovery")


if __name__ == "__main__":
    for f in (fig_difficulty, fig_regimes, fig_assumption, fig_vault, fig_infobonus, fig_noise, fig_recovery):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, repr(e))
