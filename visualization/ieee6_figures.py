"""Two multi-panel, page-width figures for the 6-page IEEE version (7.1 in wide, 7 pt text). Intervals: 95% bootstrap over units
(trained models for learned agents, evaluation seeds otherwise); dots in panel (b) of Fig. 1 are the individual trained models."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from visualization import final_figures as F
from evaluation.hierarchical import is_learned

OUT = "paper/figures_ieee6"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.2})
STY, stat, rd = F.STYLE, F.stat, F.rd
W = 7.1


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def err(ax, x, s, color, mk, ls="-", label=None, ms=3.8):
    ax.errorbar(x, s[0], yerr=[[max(s[0] - s[1], 0)], [max(s[2] - s[0], 0)]] if np.isscalar(s[0]) else [s[0] - s[1], s[2] - s[0]], color=color, marker=mk, ls=ls, ms=ms, capsize=1.5, elinewidth=0.7, label=label)


def fig_main():
    sc, gen, nz = rd("scaling"), rd("generalization"), rd("noise")
    order = ["reactive", "psrl", "hypothesis", "macro_gru", "macro_gruB"]
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.95), gridspec_kw=dict(width_ratios=[1, 1.25, 1]))
    ax = axs[0]
    for a in order + ["generic_gru"]:
        lab, c, mk, ls = STY[a]
        pts = [stat(sc[sc.difficulty == d], a) for d in range(1, 6)]
        m = np.array([p[0] for p in pts]); lo = np.array([p[1] for p in pts]); hi = np.array([p[2] for p in pts])
        ax.errorbar(np.arange(1, 6) + (order.index(a) if a in order else 5) * 0.02 - 0.05, m, yerr=[m - lo, hi - m], color=c, marker=mk, ls=ls, ms=3.5, capsize=1.2, elinewidth=0.6)
    ax.set_xlabel("difficulty"); ax.set_ylabel("success rate"); ax.set_xticks(range(1, 6)); ax.set_ylim(0, 1.04); ax.set_title("(a) difficulty", loc="left")
    ax = axs[1]
    regs = [("A_interpolation", "A"), ("B_compositional", "B"), ("C_chain_length", "C"), ("C2_size_only", "C2"), ("D_rule_permutation", "D"), ("E_novel_combination", "E")]
    g = gen[gen.split == "test"]
    for j, a in enumerate(order):
        lab, c, mk, ls = STY[a]
        for i, (rn, _) in enumerate(regs):
            s = stat(g[g.condition == rn], a, rn)
            if not s:
                continue
            x = i + (j - 2) * 0.14
            ax.errorbar(x, s[0], yerr=[[s[0] - s[1]], [s[2] - s[0]]], color=c, marker=mk, ms=3.8, capsize=1.2, elinewidth=0.7, ls="none")
            if is_learned(a) and len(s[3]) > 1 and rn in ("D_rule_permutation", "B_compositional"):
                ax.scatter(np.full(len(s[3]), x), s[3], s=2, c=c, alpha=0.35, linewidths=0, zorder=1)
    ax.set_xticks(range(len(regs))); ax.set_xticklabels([r[1] for r in regs]); ax.set_xlabel("held-out regime"); ax.set_ylim(0, 1.04); ax.set_title("(b) generalization (dots: trained models)", loc="left")
    ax = axs[2]
    slips = sorted(nz.slip.unique())
    for a in ["reactive", "psrl", "hypothesis", "macro_gru", "macro_gruB"]:
        lab, c, mk, ls = STY[a]
        pts = [stat(nz[(nz.difficulty == 3) & (nz.slip == sl)], a) for sl in slips]
        m = np.array([p[0] for p in pts]); lo = np.array([p[1] for p in pts]); hi = np.array([p[2] for p in pts])
        ax.errorbar(slips, m, yerr=[m - lo, hi - m], color=c, marker=mk, ls=ls, ms=3.5, capsize=1.2, elinewidth=0.6)
    ax.set_xlabel("lock-jam probability"); ax.set_xticks(slips); ax.set_ylim(0, 1.04); ax.set_title("(c) noisy feedback, d=3", loc="left")
    h = [Line2D([0], [0], color=STY[a][1], marker=STY[a][2], ls=STY[a][3], ms=3.5, lw=1.1, label=STY[a][0]) for a in order + ["generic_gru"]]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.1), columnspacing=1.2, handlelength=2.0)
    fig.tight_layout(w_pad=0.8); save(fig, "fig6_main")


def fig_second():
    vt, mk, ab, fx = rd("vault"), rd("master_key"), rd("ablation"), rd("factorial")
    fig, axs = plt.subplots(1, 4, figsize=(W, 1.85), gridspec_kw=dict(width_ratios=[1.1, 0.8, 1.2, 1.1]))
    ax = axs[0]; ag = ["reactive", "hypothesis", "vault_ff", "vault_gru"]; bw = 0.2
    for j, a in enumerate(ag):
        lab, c, _, _ = STY[a]
        for i, cn in enumerate(("in_dist", "larger")):
            s = stat(vt[vt.condition == cn], a)
            x = i + (j - 1.5) * bw
            ax.bar(x, s[0], bw * 0.9, color=c, edgecolor="white", linewidth=0.3)
            ax.errorbar(x, s[0], yerr=[[s[0] - s[1]], [s[2] - s[0]]], color="k", capsize=1, lw=0.6)
            if is_learned(a) and len(s[3]) > 1:
                ax.scatter(np.full(len(s[3]), x), s[3], s=1.6, c="k", alpha=0.4, linewidths=0, zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["training sizes", "larger codes"]); ax.set_ylabel("success rate"); ax.set_ylim(0, 1.05); ax.set_title("(a) vault", loc="left")
    ax = axs[1]; h = vt[vt.agent == "hypothesis"]
    obs = [h[h.condition == c].resets.mean() for c in ("in_dist", "larger")]; pred = [h[h.condition == c].pred_elim_resets.mean() for c in ("in_dist", "larger")]
    ax.bar(np.arange(2) - 0.19, pred, 0.36, color="white", edgecolor="k", hatch="////", linewidth=0.5, label="Prop. 2")
    ax.bar(np.arange(2) + 0.19, obs, 0.36, color=STY["hypothesis"][1], label="agent")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["small", "larger"]); ax.set_ylabel("failed presses"); ax.set_ylim(0, max(obs + pred) * 1.5); ax.legend(frameon=False, loc="upper left"); ax.set_title("(b) cost of tests", loc="left")
    ax = axs[2]
    conds = [("master_key_d3", "master\nd=3"), ("master_key_d4", "master\nd=4"), ("exclusive_control_d4", "control\nd=4")]
    ag = ["hypothesis", "hyp_adaptive", "psrl"]; bw = 0.26
    for j, a in enumerate(ag):
        lab, c, _, _ = STY[a]
        for i, (cn, _) in enumerate(conds):
            s = stat(mk[mk.condition == cn], a)
            x = i + (j - 1) * bw
            ax.bar(x, s[0], bw * 0.9, color=c, edgecolor="white", linewidth=0.3)
            ax.errorbar(x, s[0], yerr=[[s[0] - s[1]], [s[2] - s[0]]], color="k", capsize=1, lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels([c[1] for c in conds]); ax.set_ylim(0, 1.08); ax.set_title("(c) false exclusivity", loc="left")
    ax = axs[3]
    lams = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    for suf, col, mkk, lab in (("", "#009E73", "o", "d=4"), ("_d5", "#D55E00", "s", "d=5")):
        pts = [stat(ab, f"lambda_{l}{suf}") for l in lams]
        m = np.array([p[0] for p in pts]); lo = np.array([p[1] for p in pts]); hi = np.array([p[2] for p in pts])
        ax.errorbar(range(len(lams)), m, yerr=[m - lo, hi - m], color=col, marker=mkk, ms=3.2, capsize=1.2, elinewidth=0.6, label=lab)
    ax.set_xticks(range(len(lams))); ax.set_xticklabels([f"{l:g}" for l in lams], fontsize=5.8); ax.set_xlabel(r"information weight $\lambda$"); ax.set_title("(d) information bonus", loc="left"); ax.legend(frameon=False, loc="lower right")
    h = [Line2D([0], [0], color=STY[a][1], lw=4, label=l) for a, l in (("reactive", "Reactive"), ("hypothesis", "Hypothesis (hard exclusivity)"), ("hyp_adaptive", "Hypothesis-adaptive"), ("psrl", "PSRL"), ("vault_ff", "PPO-FF"), ("vault_gru", "PPO-GRU"))]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.09), columnspacing=1.2, handlelength=1.2, fontsize=6.3)
    fig.tight_layout(w_pad=0.7); save(fig, "fig6_second")


if __name__ == "__main__":
    fig_main(); fig_second(); print("ok")
