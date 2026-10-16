import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COL = {"red": "#d62728", "blue": "#1f77b4", "green": "#2ca02c", "yellow": "#e6b800", "purple": "#9467bd", "orange": "#ff7f0e", "cyan": "#17becf", "pink": "#e377c2", "": "#7f7f7f"}


def render(spec, ax=None, title=None):
    n = spec.grid_size
    ax = ax or plt.subplots(figsize=(4.2, 4.2))[1]
    ax.set_xlim(0, n); ax.set_ylim(n, 0); ax.set_aspect("equal"); ax.set_xticks(range(n + 1)); ax.set_yticks(range(n + 1))
    ax.grid(True, lw=0.4, color="#cccccc"); ax.tick_params(labelbottom=False, labelleft=False, length=0)
    for (r, c) in spec.walls:
        ax.add_patch(Rectangle((c, r), 1, 1, color="#444444"))
    for (r, c), o in spec.objects.items():
        col = COL[o.color]
        if o.kind == "door":
            ax.add_patch(Rectangle((c + .1, r + .05), .8, .9, fc=col, ec="k"))
        elif o.kind == "key":
            ax.text(c + .5, r + .55, "K", ha="center", va="center", color=col, fontsize=11, weight="bold")
        elif o.kind == "switch":
            ax.add_patch(plt.Circle((c + .5, r + .5), .28, fc="white", ec=col, lw=2))
        elif o.kind == "chasm":
            ax.add_patch(Rectangle((c, r), 1, 1, fc="white", ec=col, lw=2, hatch="///"))
        elif o.kind == "block":
            ax.add_patch(Rectangle((c + .1, r + .1), .8, .8, fc="#a0522d", ec="k"))
        elif o.kind == "trap":
            ax.text(c + .5, r + .5, "x", ha="center", va="center", color="k", fontsize=12)
        elif o.kind == "target":
            ax.text(c + .5, r + .5, "*", ha="center", va="center", color="#c8a000", fontsize=20)
    r, c = spec.agent_start
    ax.add_patch(plt.Circle((c + .5, r + .5), .3, fc="#17becf", ec="k"))
    ax.set_title(title or spec.environment_id, fontsize=9)
    return ax
