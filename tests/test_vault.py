import numpy as np
from environment.vault import Vault
from agents.vault_agents import HypothesisVault, ReactiveVault, vec, VEC


def run(agent, L, k, seed, T=150):
    env = Vault(L, k, T, seed); o = env.reset(); agent.reset(o)
    while not o["done"]:
        o = env.step(agent.act(o))
    return o["success"], env.resets


def test_code_hidden_and_injective():
    env = Vault(6, 4, 100, 3)
    assert not any("code" in k for k in env.obs()) and len(set(env._code)) == 4 and vec(env.obs()).shape == (VEC,)


def test_wrong_press_resets_and_correct_sequence_opens():
    env = Vault(5, 3, 100, 1); env.reset()
    code = env._code
    wrong = next(i for i in range(5) if i != code[0])
    assert env.step(wrong)["fb"] == 2 and env.p == 0
    for i, c in enumerate(code):
        o = env.step(c)
    assert o["success"] and o["done"]


def test_elimination_matches_closed_form_and_beats_reactive():
    L, k = 5, 3
    res = [run(HypothesisVault(i), L, k, 1000 + i) for i in range(150)]
    assert all(r[0] for r in res)
    pred = sum((L - p - 1) / 2 for p in range(k))
    assert abs(np.mean([r[1] for r in res]) - pred) < 0.6
    assert np.mean([run(ReactiveVault(i), L, k, 1000 + i)[1] for i in range(150)]) > 2 * pred
