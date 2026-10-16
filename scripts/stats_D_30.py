import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import numpy as np

f = "data/results/generalization_D_30.csv"
df = pd.read_csv(f)

x = df.groupby(["agent", "seed"])["success"].mean().unstack(0)

print()
print("=== D 30-SEED RESULTS ===")
for a in x.columns:
    print(f"{a:18s} mean={x[a].mean():.4f} sd={x[a].std(ddof=1):.4f}")

pairs = [
    ("hypothesis", "macro_ff"),
    ("hypothesis", "macro_gru"),
    ("hypothesis", "psrl"),
    ("hypothesis", "macro_gruB"),
    ("hypothesis", "reactive"),
]

print()
print("=== PAIRED TESTS ===")

for a, b in pairs:
    d = x[a] - x[b]
    t = ttest_rel(x[a], x[b])

    try:
        w = wilcoxon(d)
        wp = w.pvalue
    except Exception:
        wp = 1.0

    rng = np.random.default_rng(0)
    boot = rng.choice(
        d.to_numpy(),
        size=(20000, len(d)),
        replace=True
    ).mean(axis=1)

    ci = np.percentile(boot, [2.5, 97.5])

    print(
        f"{a} - {b}: "
        f"delta={d.mean():.4f}, "
        f"t_p={t.pvalue:.6g}, "
        f"wilcoxon_p={wp:.6g}, "
        f"CI=[{ci[0]:.4f},{ci[1]:.4f}]"
    )

print()
print("Rows:", len(df))

print("Seeds per agent:")
print(df.groupby("agent")["seed"].nunique())

print("Environments per seed:")
print(df.groupby(["agent", "seed"]).size().unique())
