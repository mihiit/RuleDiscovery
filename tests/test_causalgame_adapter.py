import numpy as np
import pytest
cg = pytest.importorskip("cg_adapter", reason="needs external/causalgame on the path")


def _spec():
    s = cg.Spec.__new__(cg.Spec)
    s.name = "toy"; s.num = {"a": (0, 50, 10), "b": (0, 50, 20), "c": (0, 50, 30)}
    s.disc = {"mode": ["x", "y"]}; s.disc_default = {"mode": "x"}; s.boo = {"flag": False}; s.budget = 60; s.names = list(s.num)
    return s


def test_projection_respects_budget_and_bounds():
    s = _spec()
    d = s.project(dict(num={"a": 50, "b": 50, "c": 50}, disc={"mode": "x"}, boo={"flag": True}))
    assert sum(d["num"].values()) <= 60 and all(0 <= v <= 50 for v in d["num"].values())


def test_features_and_posterior_shapes():
    from cg_agents import Posterior
    s = _spec(); p = Posterior(s); d = s.default()
    p.add(d, 12, 20); p.fit()
    assert len(s.features(d)) == len(s.prior_precision()) and np.isfinite(p.mean(d)) and p.info_gain(d) > 0
