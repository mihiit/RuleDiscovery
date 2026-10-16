"""Second environment family: a combination vault. A hidden injective sequence of k levers (out of L) must be pressed in order.
Pressing the next correct lever advances progress; any wrong press resets progress to 0. No navigation, different hypothesis space
(sequences, not pairings) and different dynamics from the grid world. The code is never exposed to agents."""
import numpy as np

NONE, ADVANCE, RESET, OPEN = 0, 1, 2, 3


class Vault:
    def __init__(self, L, k, max_steps, seed):
        rng = np.random.default_rng(seed)
        self.L, self.k, self.max_steps = L, k, max_steps
        self._code = tuple(int(x) for x in rng.permutation(L)[:k])   # hidden
        self.reset()

    def reset(self):
        self.p, self.t, self.last, self.fb, self.done, self.success, self.resets = 0, 0, -1, NONE, False, False, 0
        return self.obs()

    def obs(self):
        return dict(L=self.L, k=self.k, p=self.p, last=self.last, fb=self.fb, t=self.t, done=self.done, success=self.success)

    def step(self, a):
        self.t += 1; self.last = a
        if a == self._code[self.p]:
            self.p += 1
            if self.p == self.k:
                self.fb, self.done, self.success = OPEN, True, True
            else:
                self.fb = ADVANCE
        else:
            self.p, self.fb = 0, RESET
            self.resets += 1
        if self.t >= self.max_steps:
            self.done = True
        return self.obs()


def sample_sizes(rng, Ls, ks):
    return int(rng.choice(Ls)), int(rng.choice(ks))
