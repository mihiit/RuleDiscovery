import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from regimes import REGIMES, TRAIN_BASE, EVAL_BASE, TEST_BASE


def test_seed_ranges_disjoint():
    assert TRAIN_BASE + 29_999 < EVAL_BASE and EVAL_BASE + 29_999 < TEST_BASE


def test_test_compositions_not_in_train():
    for name, reg in REGIMES.items():
        tr = {tuple(sorted(t.get("rules", ()))) + (t.get("grid_size"), t.get("colors"), t.get("mapping")) for t in reg["train"]}
        te = {tuple(sorted(t.get("rules", ()))) + (t.get("grid_size"), t.get("colors"), t.get("mapping")) for t in reg["test"]}
        assert not (tr & te), name
