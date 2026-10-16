import numpy as np
from evaluation.hierarchical import compare_units, holm, boot_mean_ci, is_learned


def test_holm_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert all(0 <= a <= 1 for a in adj)
    assert adj[0] == 0.04 and adj[3] == 0.5          # smallest p scaled by m=4; largest unchanged
    assert adj[2] >= adj[0]


def test_compare_units_detects_shift_and_handles_constants():
    r = compare_units([0.9, 0.92, 0.88, 0.91], [0.5, 0.52, 0.48, 0.51])
    assert r["diff"] > 0.3 and r["ci_lo"] > 0.3 and r["p"] < 0.001
    same = compare_units([1.0] * 5, [1.0] * 5)
    assert same["diff"] == 0 and same["p"] == 1.0


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(0).uniform(0.4, 0.6, 20)
    lo, hi = boot_mean_ci(x)
    assert lo <= x.mean() <= hi


def test_learned_prefixes():
    assert is_learned("macro_gru") and is_learned("vault_ff") and not is_learned("hypothesis") and not is_learned("hyp_adaptive")
