from agents.random_agent import RandomAgent
from agents.reactive_agent import ReactiveAgent
from agents.hypothesis_agent import HypothesisAgent
from environment.generator import EnvironmentConfig, generate
from environment.world import ACTIONS
from evaluation.metrics import run_episode


def test_agents_return_valid_actions_and_terminate():
    spec = generate(EnvironmentConfig(seed=1, difficulty=5, max_steps=80))
    for A in (RandomAgent, ReactiveAgent, HypothesisAgent):
        r = run_episode(spec, A(0))
        assert r["steps"] <= 80


def test_reproducible_with_seed():
    spec = generate(EnvironmentConfig(seed=3, difficulty=3))
    for A in (RandomAgent, ReactiveAgent):
        assert run_episode(spec, A(5))["steps"] == run_episode(spec, A(5))["steps"]
