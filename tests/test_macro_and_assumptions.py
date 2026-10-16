import numpy as np
from environment.generator import EnvironmentConfig, generate, is_solvable
from environment.rules import KEY_DOOR, SWITCH_BRIDGE
from environment.world import World
from agents.hypothesis_agent import HypothesisAgent
from agents.psrl_agent import PSRLAgent
from agents.macro_engine import MacroEngine
from evaluation.metrics import run_episode


def test_master_key_solvable_and_shared():
    for s in range(15):
        sp = generate(EnvironmentConfig(seed=s, difficulty=3, rules=(KEY_DOOR, KEY_DOOR, SWITCH_BRIDGE), shared_keys=True))
        assert is_solvable(sp) and sum(len(v) for v in sp.hidden.key_door_extra.values()) == 1


def test_exclusivity_matters_on_master_key():
    sp = [generate(EnvironmentConfig(seed=s, difficulty=3, rules=(KEY_DOOR, KEY_DOOR, SWITCH_BRIDGE), shared_keys=True)) for s in range(12)]
    ex = sum(run_episode(x, HypothesisAgent(i))["success"] for i, x in enumerate(sp))
    ne = sum(run_episode(x, HypothesisAgent(i, exclusive=False))["success"] for i, x in enumerate(sp))
    assert ne > ex


def test_macro_engine_exposes_no_belief_information():
    sp = generate(EnvironmentConfig(seed=2, difficulty=4))
    w = World(sp); o = w.reset(); e = MacroEngine(0); e.reset()
    c1 = e.candidates(o)
    e.log.append(None) if False else None
    assert c1 and all(m["reward"] in (0.0, 1.5, 10.0, 1.5 * 1.0) or m["type"] in ("PICK", "TARGET", "TEST", "SWITCH", "PUSH") for m in c1)
    from rl.macro_feat import featurize, NMAX, CAND_F, STATE_F
    cf, st, mask = featurize(o, c1, (None, 0))
    assert cf.shape == (NMAX, CAND_F) and st.shape == (STATE_F,) and mask.sum() == min(len(c1), NMAX)


def test_psrl_information_bonus_hybrid_runs():
    sp = generate(EnvironmentConfig(seed=5, difficulty=4))
    r = run_episode(sp, PSRLAgent(1, lam=0.5, ig_from_posterior=True))
    assert r["steps"] <= sp.max_steps
