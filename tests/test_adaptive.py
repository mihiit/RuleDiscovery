from agents.hypothesis_manager import HypothesisManager


def _confirm(m, key, door):
    h = m.get(("KD", key, door)); h.confirmed = True; h.confidence = 1.0


def test_hard_exclusivity_excludes_used_key():
    m = HypothesisManager(prior=None, exclusive=True)
    _confirm(m, "red", "blue")
    assert m.bind_prob("KD", "red", "green", ["red", "yellow"]) == 0.0


def test_adaptive_keeps_small_weight_then_drops_assumption():
    m = HypothesisManager(prior=None, exclusive="adaptive")
    _confirm(m, "red", "blue")
    p_soft = m.bind_prob("KD", "red", "green", ["red", "yellow"])
    assert 0.0 < p_soft < 0.2                      # tried last, but not excluded
    _confirm(m, "red", "green")                    # violation observed: one key, two doors
    assert m.violated("KD")
    p_free = m.bind_prob("KD", "red", "purple", ["red", "yellow"])
    assert p_free > p_soft                         # assumption dropped, red is a normal candidate again
