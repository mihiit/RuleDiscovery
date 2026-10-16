"""Evaluation wrapper for trained PPO policies (feedforward or GRU)."""
import pickle
import numpy as np
from .base_agent import BaseAgent


_JIT = {}


class PPOAgent(BaseAgent):
    def __init__(self, seed=0, model_path=None, label="ppo", params=None, recurrent=None):
        super().__init__(seed)
        import jax
        from rl import net, encode
        self.net, self.enc, self.jax = net, encode, jax
        if params is None:
            with open(model_path, "rb") as f:
                d = pickle.load(f)
            params, recurrent = d["params"], d["recurrent"]
        self.params = jax.tree_util.tree_map(lambda a: jax.numpy.asarray(a), params)
        self.recurrent, self.name = recurrent, label
        self.np_rng = np.random.default_rng(seed)
        if recurrent not in _JIT:
            _JIT[recurrent] = jax.jit(lambda p, c, a, v, h, r=recurrent: net.step(p, c, a, v, h, r))
        self._f = _JIT[recurrent]
        self.reset()

    def reset(self):
        self.h = np.zeros((1, self.net.H), np.float32)
        self.last = None

    def act(self, obs):
        c, a, v = self.enc.encode(obs, self.last)
        lg, _, h = self._f(self.params, c[None], a[None], v[None], self.h)
        self.h = np.asarray(h)
        lg = np.asarray(lg[0], np.float64); p = np.exp(lg - lg.max()); p /= p.sum()
        self.last = self.enc.ACT[int(self.np_rng.choice(7, p=p))]
        return self.last
