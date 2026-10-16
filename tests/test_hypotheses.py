from agents.hypothesis_manager import HypothesisManager, RelationPrior, Hypothesis
from agents.memory import EpisodicMemory, MemoryEntry
from agents.hypothesis_agent import HypothesisAgent
from environment.generator import EnvironmentConfig, generate
from evaluation.metrics import run_episode


def mgr(**kw):
    return HypothesisManager(RelationPrior(), **kw)


def entry(hid, error, step=1):
    return MemoryEntry(step, (), "TEST", "x", hid, True, error)


def test_creation_and_confidence_update():
    m = mgr()
    h = m.get(("KD", "red", "blue"))
    assert h.confidence == 0.5
    m.update_hypothesis(h, True)
    assert h.confidence > 0.5 and h.confirmed


def test_contradiction_and_rejection():
    m = mgr(reject_after=2)
    h = m.get(("KD", "red", "blue"))
    m.update_hypothesis(h, False)
    assert h.confidence < 0.5 and not h.rejected and h.contradiction_count == 1
    m.update_hypothesis(h, False)
    assert h.rejected


def test_revision_disabled_never_rejects():
    m = mgr(revision=False)
    h = m.get(("KD", "red", "blue"))
    for _ in range(5):
        m.update_hypothesis(h, False)
    assert not h.rejected and h.confidence == 0.5 and h.contradiction_count == 5


def test_elimination_after_confirmation():
    mem = EpisodicMemory()
    mem.add(entry(("KD", "red", "blue"), False))
    m = mgr(); m.rebuild(mem)
    assert m.bind_prob("KD", "red", "blue", {"red", "green"}) == 1.0
    assert m.bind_prob("KD", "green", "blue", {"red", "green"}) == 0.0
    assert m.bind_prob("KD", "red", "green", {"red", "green"}) == 0.0


def test_failed_pair_lowers_probability():
    mem = EpisodicMemory(); mem.add(entry(("KD", "red", "blue"), True))
    m = mgr(); m.rebuild(mem)
    assert m.bind_prob("KD", "red", "blue", {"red", "green"}) < 0.01


def test_memory_capacity():
    for cap, expect in ((None, 5), (3, 3), (0, 0)):
        mem = EpisodicMemory(cap)
        for i in range(5):
            mem.add(entry(("PUSH", "block"), False, i))
        assert len(mem) == expect


def test_transfer_prior_learns_relation():
    ag = HypothesisAgent(0, transfer=True)
    for s in range(15):
        run_episode(generate(EnvironmentConfig(seed=s, difficulty=3, rules=("KEY_DOOR",), mapping="identity")), ag)
    assert ag.prior.p_same("KD") > 0.8


def test_hypothesis_agent_deterministic_and_terminates():
    spec = generate(EnvironmentConfig(seed=9, difficulty=4))
    a, b = run_episode(spec, HypothesisAgent(1)), run_episode(spec, HypothesisAgent(1))
    assert a["steps"] == b["steps"] and a["success"] == b["success"] and a["steps"] <= spec.max_steps
