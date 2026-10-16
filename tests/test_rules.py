from environment.generator import EnvironmentConfig, generate
from environment.rules import KEY_DOOR, SWITCH_BRIDGE, PUSH_BLOCK


def test_rule_composition_present():
    for rules in [(KEY_DOOR,), (SWITCH_BRIDGE,), (PUSH_BLOCK,), (KEY_DOOR, SWITCH_BRIDGE, PUSH_BLOCK)]:
        spec = generate(EnvironmentConfig(seed=2, difficulty=3, rules=rules))
        kinds = {o.kind for o in spec.objects.values()}
        assert ("door" in kinds) == (KEY_DOOR in rules)
        assert ("chasm" in kinds) == (SWITCH_BRIDGE in rules)
        assert ("block" in kinds) == (PUSH_BLOCK in rules)


def test_mapping_modes():
    for s in range(20):
        i = generate(EnvironmentConfig(seed=s, difficulty=3, rules=(KEY_DOOR, KEY_DOOR), mapping="identity", colors=("red", "blue", "green")))
        assert all(k == d for k, d in i.hidden.key_door.items())
        a = generate(EnvironmentConfig(seed=s, difficulty=3, rules=(KEY_DOOR, KEY_DOOR), mapping="anti"))
        assert all(k != d for k, d in a.hidden.key_door.items())


def test_key_capacity_swap():
    from environment.world import World
    spec = generate(EnvironmentConfig(seed=4, difficulty=3, rules=(KEY_DOOR,)))
    w = World(spec)
    keys = [p for p, o in w.objs.items() if o.kind == "key"]
    assert len(keys) >= 2
    for k in keys[:2]:
        w.pos = k; w.step("INTERACT")
    assert len(w.inventory) == 1
