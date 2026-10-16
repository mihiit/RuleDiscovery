import random


class ActionSelector:
    """Utility(a) = task_reward + lam * information_gain - mu * action_cost."""

    def __init__(self, lam=0.5, mu=0.05, mode="utility"):
        self.lam, self.mu, self.mode = lam, mu, mode

    def utility(self, m):
        return m["reward"] + self.lam * m["ig"] - self.mu * m["cost"]

    def choose(self, macros, rng: random.Random):
        exploit = [m for m in macros if m["type"] == "TARGET"]
        if exploit:
            return min(exploit, key=lambda m: m["cost"])
        if self.mode == "random":
            return rng.choice(macros)
        best = max(self.utility(m) + 1e-6 * rng.random() for m in macros)
        for m in macros:
            pass
        scored = [(self.utility(m) + 1e-6 * rng.random(), i) for i, m in enumerate(macros)]
        return macros[max(scored)[1]]
