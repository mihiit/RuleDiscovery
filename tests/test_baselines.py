import numpy as np
from environment.generator import EnvironmentConfig, generate, is_solvable
from environment.rules import KEY_DOOR, SWITCH_DOOR, SWITCH_BRIDGE, ALL_COLORS
from environment.world import World, Obj, EnvironmentSpec
from environment.rules import HiddenRules
from agents.psrl_agent import PSRLAgent
from agents.hypothesis_agent import HypothesisAgent
from evaluation.metrics import run_episode


def test_switch_door_rule_and_solvable():
    for s in range(15):
        assert is_solvable(generate(EnvironmentConfig(seed=s, difficulty=3, rules=(KEY_DOOR, SWITCH_DOOR, SWITCH_BRIDGE))))
    h = HiddenRules(("SWITCH_DOOR",), {}, {}, {"red": "blue"})
    w = World(EnvironmentSpec("t", 0, 1, 5, frozenset(), {(1, 0): Obj("switch", "red", "off"), (1, 2): Obj("door", "blue", "closed")}, (1, 0), h, 50))
    assert w.step("INTERACT").feedback == "switch_toggled" and w.objs[(1, 2)].state == "open"


def test_psrl_deterministic_and_terminates():
    spec = generate(EnvironmentConfig(seed=7, difficulty=4))
    a, b = run_episode(spec, PSRLAgent(3)), run_episode(spec, PSRLAgent(3))
    assert a["steps"] == b["steps"] <= spec.max_steps


def test_open_vocab_agent_solves_switch_door():
    ok = sum(run_episode(generate(EnvironmentConfig(seed=s, difficulty=3, rules=(KEY_DOOR, SWITCH_DOOR))), HypothesisAgent(s, use_sd=True))["success"] for s in range(10))
    assert ok >= 8


def test_extended_colour_pool_and_encoding_shapes():
    from rl.encode import encode, G, N_VEC
    spec = generate(EnvironmentConfig(seed=1, difficulty=3, colors=ALL_COLORS[4:], mapping="identity"))
    obs = World(spec).reset()
    c, a, v = encode(obs)
    assert c.shape == (G, G) and a.shape == (G, G) and v.shape == (N_VEC,) and c.max() < 40
    assert all(k == d for k, d in spec.hidden.key_door.items())
