"""Shared environment-distribution definitions (used by experiments and RL training)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.generator import EnvironmentConfig
from environment.rules import (KEY_DOOR as KD, SWITCH_BRIDGE as SB, PUSH_BLOCK as PB, SWITCH_DOOR as SD, ALL_COLORS)

TRAIN_BASE, EVAL_BASE, TEST_BASE = 1_000_000, 2_000_000, 3_000_000
CA, CB = ALL_COLORS[:4], ALL_COLORS[4:]          # disjoint colour sets for regime D (red,blue,green,yellow | purple,orange,cyan,pink)

REGIMES = {
    "A_interpolation": dict(train=[dict(rules=(KD,)), dict(rules=(SB,)), dict(rules=(PB,))],
                            test=[dict(rules=(KD, SB)), dict(rules=(KD, PB)), dict(rules=(SB, PB))]),
    "B_compositional": dict(train=[dict(rules=(KD, SB)), dict(rules=(KD, PB)), dict(rules=(SB, PB))],
                            test=[dict(rules=(KD, SB, PB))]),
    # C: longer dependency chains (2 gates, 10x10 -> 4 gates, 12x12) -- complexity, not just size
    "C_chain_length": dict(train=[dict(rules=(KD, SB), grid_size=10)],
                           test=[dict(rules=(KD, SB, KD, SB), grid_size=12)]),
    # C2: size only (kept to expose the size/difficulty confound found in review)
    "C2_size_only": dict(train=[dict(rules=(KD, SB, PB), grid_size=10)],
                         test=[dict(rules=(KD, SB, PB), grid_size=14)]),
    # D: harder, disjoint colours (unseen colour symbols), identity relation preserved
    "D_rule_permutation": dict(train=[dict(rules=(KD, KD, SB, KD), colors=CA, mapping="identity", grid_size=12)],
                               test=[dict(rules=(KD, KD, SB, KD), colors=CB, mapping="identity", grid_size=12)]),
    "E_novel_combination": dict(train=[dict(rules=(KD,)), dict(rules=(SB, PB))],
                                test=[dict(rules=(KD, SB, PB))]),
}


def mk(seed, tmpl, difficulty=3):
    t = dict(tmpl)
    t.setdefault("difficulty", difficulty)
    return EnvironmentConfig(seed=seed, **t)


def train_config(dist, rng, max_difficulty=5):
    """Sample one training-environment config for distribution `dist` ('main' = difficulties 1-5, random bindings)."""
    seed = TRAIN_BASE + int(rng.integers(0, 900_000))
    if dist == "d1":
        return mk(seed, {}, difficulty=1)
    if dist == "main":
        return mk(seed, {}, difficulty=int(rng.integers(1, max_difficulty + 1)))
    tm = REGIMES[dist]["train"]
    return mk(seed, tm[int(rng.integers(0, len(tm)))])
