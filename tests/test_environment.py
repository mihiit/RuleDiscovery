import numpy as np
from environment.generator import EnvironmentConfig, generate, is_solvable
from environment.world import World, ACTIONS, Obj, EnvironmentSpec
from environment.rules import HiddenRules


def test_determinism():
    a, b = generate(EnvironmentConfig(seed=5, difficulty=3)), generate(EnvironmentConfig(seed=5, difficulty=3))
    assert a.agent_start == b.agent_start and a.walls == b.walls
    assert {p: (o.kind, o.color) for p, o in a.objects.items()} == {p: (o.kind, o.color) for p, o in b.objects.items()}


def test_all_difficulties_solvable():
    for d in range(1, 6):
        for s in range(15):
            assert is_solvable(generate(EnvironmentConfig(seed=s, difficulty=d)))


def test_reset_and_actions():
    w = World(generate(EnvironmentConfig(seed=1, difficulty=2)))
    o1 = w.reset(); w.step("WAIT"); o2 = w.reset()
    assert np.array_equal(o1.grid, o2.grid) and o2.step == 0
    for a in ACTIONS:
        w.step(a)


def test_hidden_isolation():
    w = World(generate(EnvironmentConfig(seed=1, difficulty=3)))
    o = w.reset()
    assert not any("hidden" in f or "rule" in f for f in o.__dataclass_fields__)
    assert not hasattr(o, "hidden")


def _mini(hidden, objs, start=(1, 0)):
    return World(EnvironmentSpec("t", 0, 1, 5, frozenset(), objs, start, hidden, 50))


def test_key_door_rule():
    h = HiddenRules(("KEY_DOOR",), {"red": "blue"}, {})
    w = _mini(h, {(1, 0): Obj("key", "red"), (1, 2): Obj("door", "blue", "closed")})
    w.step("INTERACT"); assert "red" in w.inventory
    w.step("RIGHT"); assert w.step("RIGHT").feedback == "blocked_door"
    assert w.step("INTERACT").feedback == "door_opened"
    assert w.step("RIGHT").feedback == "moved"


def test_wrong_key_does_not_open():
    h = HiddenRules(("KEY_DOOR",), {"red": "blue"}, {})
    w = _mini(h, {(1, 0): Obj("key", "green"), (1, 2): Obj("door", "blue", "closed")})
    w.step("INTERACT"); w.step("RIGHT"); w.step("RIGHT")
    assert w.step("INTERACT").feedback == "door_locked"


def test_switch_bridge_and_push():
    h = HiddenRules(("SWITCH_BRIDGE",), {}, {"green": "yellow"})
    w = _mini(h, {(1, 0): Obj("switch", "green", "off"), (1, 2): Obj("chasm", "yellow", "inactive"),
                  (3, 0): Obj("block"), })
    w.step("RIGHT"); assert w.step("RIGHT").feedback == "blocked_chasm"
    w.step("LEFT"); w.step("INTERACT"); assert w.objs[(1, 2)].state == "active"
    w.pos = (2, 0); w.facing = "DOWN"
    assert w.step("PUSH").feedback == "pushed" and (4, 0) in w.objs


def test_agents_terminate():
    from agents.random_agent import RandomAgent
    from agents.reactive_agent import ReactiveAgent
    from evaluation.metrics import run_episode
    spec = generate(EnvironmentConfig(seed=3, difficulty=3, max_steps=60))
    for A in (RandomAgent, ReactiveAgent):
        assert run_episode(spec, A(0))["steps"] <= 60
