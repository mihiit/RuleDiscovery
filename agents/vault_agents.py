import itertools, pickle
import numpy as np

LMAX, KMAX = 8, 6
VEC = LMAX + 4 + (KMAX + 1) + 2 + 1


def vec(o):
    v = np.zeros(VEC, np.float32)
    if o["last"] >= 0: v[o["last"]] = 1
    v[LMAX + o["fb"]] = 1
    v[LMAX + 4 + o["p"]] = 1
    v[LMAX + 4 + KMAX + 1] = o["L"] / LMAX; v[LMAX + 4 + KMAX + 2] = o["k"] / KMAX; v[-1] = o["t"] / 100
    return v


class RandomVault:
    name = "random"
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def reset(self, o): pass
    def act(self, o): return int(self.rng.integers(o["L"]))


class ReactiveVault:
    """Stateless apart from the last press: never presses the same lever twice in a row."""
    name = "reactive"
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def reset(self, o): pass
    def act(self, o):
        c = [i for i in range(o["L"]) if i != o["last"]]
        return int(self.rng.choice(c))


class HypothesisVault:
    """Explicit hypothesis set: all injective codes consistent with every outcome so far; presses the lever with the highest
    posterior probability at the current position (ties broken at random). Equivalent to posterior sampling here (uniform prior)."""
    name = "hypothesis"
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)

    def reset(self, o):
        L, k = o["L"], o["k"]
        self.H = np.array(list(itertools.permutations(range(L), k)), dtype=np.int8)
        self.prev_p = 0

    def act(self, o):
        if o["fb"] in (1, 2, 3) and o["last"] >= 0:      # outcome of the previous press (made at position prev_p)
            c = self.H[:, self.prev_p]
            self.H = self.H[c == o["last"]] if o["fb"] in (1, 3) else self.H[c != o["last"]]
        self.prev_p = o["p"]
        counts = np.bincount(self.H[:, o["p"]], minlength=o["L"])
        best = np.flatnonzero(counts == counts.max())
        self.pending = int(self.rng.choice(best))
        return self.pending


class VaultPPOAgent:
    def __init__(self, seed=0, model_path=None, label="vault_ppo"):
        import jax
        from rl import vault_net as vn
        self.vn, self.jax = vn, jax
        d = pickle.load(open(model_path, "rb"))
        self.params = jax.tree_util.tree_map(jax.numpy.asarray, d["params"]); self.rec = d["recurrent"]; self.name = label
        if self.rec not in vn._JIT:
            vn._JIT[self.rec] = jax.jit(lambda p, x, h, r=self.rec: vn.step(p, x, h, r))
        self.f, self.rng = vn._JIT[self.rec], np.random.default_rng(seed)

    def reset(self, o): self.h = np.zeros((1, self.vn.H), np.float32)

    def act(self, o):
        lg, _, h = self.f(self.params, vec(o)[None], self.h); self.h = np.asarray(h)
        lg = np.asarray(lg[0], np.float64)[: o["L"]]
        p = np.exp(lg - lg.max()); p /= p.sum()
        return int(self.rng.choice(o["L"], p=p))
