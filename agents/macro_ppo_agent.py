"""Evaluation wrapper for the trained macro-level PPO policies (feedforward or GRU)."""
import pickle
import numpy as np
from .base_agent import BaseAgent
from .macro_engine import MacroEngine
from .hypothesis_agent import HypothesisAgent

_JIT = {}


class MacroPPOAgent(BaseAgent):
    def __init__(self, seed=0, model_path=None, label="macro_ppo"):
        super().__init__(seed)
        import jax
        from rl import macro_net, macro_feat
        self.net, self.feat = macro_net, macro_feat
        d = pickle.load(open(model_path, "rb"))
        self.params = jax.tree_util.tree_map(lambda a: jax.numpy.asarray(a), d["params"])
        self.rec, self.name, self.belief = d["recurrent"], label, bool(d.get("belief", False))
        if self.rec not in _JIT:
            _JIT[self.rec] = jax.jit(lambda p, c, s, m, h, r=self.rec: macro_net.step(p, c, s, m, h, r))
        self._f = _JIT[self.rec]
        self.np_rng = np.random.default_rng(seed)
        self.engine = HypothesisAgent(seed) if self.belief else MacroEngine(seed)
        self.reset()

    def reset(self):
        self.engine.reset()
        self.h = np.zeros((1, self.net.H), np.float32)
        self.last, self.cur, self.before = (None, 0), None, None

    def act(self, obs):
        e = self.engine
        if not e.plan:
            cands = e.candidates(obs)
            if not cands:
                return self.np_rng.choice(["UP", "DOWN", "LEFT", "RIGHT"])
            cf, st, mask = self.feat.featurize(obs, cands, self.last, self.belief)
            lg, _, h = self._f(self.params, cf[None], st[None], mask[None], self.h)
            self.h = np.asarray(h)
            p = np.exp(np.asarray(lg[0], np.float64) - np.max(lg[0])); p *= mask; p /= p.sum()
            i = int(self.np_rng.choice(len(p), p=p))
            self.cur, self.before = e.commit(obs, i), obs
        a, kind, meta = e.plan.pop(0)
        e._cur = (a, kind, meta, obs)
        return a

    def observe(self, obs, action, nxt):
        e = self.engine
        e.observe(obs, action, nxt)
        if self.cur is not None and not e.plan:       # macro finished (or aborted)
            kind = e._cur[1] if e._cur else None
            self.last = (self.cur, self.feat.outcome(self.cur, self.before, nxt) if kind == "final" else 0)
            self.cur = None
