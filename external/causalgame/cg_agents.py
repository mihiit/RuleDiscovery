"""Non-LLM agents for CausalGame. All see only (survived, deployed) per batch. Budget: 10 deployments x 20 drones."""
import numpy as np

BATCH, N_DEPLOY = 20, 10


class Posterior:
    """Bayesian linear regression of batch survival rate on design features (Gaussian posterior, heteroscedastic binomial noise)."""

    def __init__(self, spec):
        self.spec, self.X, self.y, self.v = spec, [], [], []
        self.L = np.diag(spec.prior_precision())

    def add(self, d, surv, n):
        p = min(max(surv / n, 0.05), 0.95)
        self.X.append(self.spec.features(d)); self.y.append(surv / n); self.v.append(p * (1 - p) / n + 0.003)

    def fit(self):
        X, y, W = np.array(self.X), np.array(self.y), np.diag(1 / np.array(self.v))
        A = X.T @ W @ X + self.L
        self.S = np.linalg.inv(A)
        self.mu = self.S @ X.T @ W @ y

    def mean(self, d):
        return float(self.spec.features(d) @ self.mu)

    def info_gain(self, d, noise=0.012):
        f = self.spec.features(d)
        return 0.5 * np.log(1 + f @ self.S @ f / noise)


def _pool(spec, rng, anchors, n_rand=1200, n_local=300):
    pool = [spec.random(rng) for _ in range(n_rand)]
    for a in anchors:
        pool += [spec.perturb(a, rng, k_changes=int(rng.integers(1, 3))) for _ in range(n_local // max(len(anchors), 1))]
    return pool + list(anchors)


class Default:
    name = "default"

    def run(self, sess, rng):
        return sess.spec.default()


class RandomSearch:
    name = "random"

    def run(self, sess, rng):
        best, best_p = None, -1
        for i in range(N_DEPLOY):
            d = sess.spec.default() if i == 0 else sess.spec.random(rng)
            s, n = sess.deploy(d, BATCH)
            if s / n > best_p:
                best, best_p = d, s / n
        return best


class HillClimb:
    """Reactive: keep the best design seen, try a small random change, keep it if the batch survived at least as often (no model, no memory beyond the incumbent)."""
    name = "hillclimb"

    def run(self, sess, rng):
        best = sess.spec.default()
        s, n = sess.deploy(best, BATCH); best_p = s / n
        for _ in range(N_DEPLOY - 1):
            cand = sess.spec.perturb(best, rng)
            s, n = sess.deploy(cand, BATCH)
            if s / n >= best_p:
                best, best_p = cand, s / n
        return best


class ModelBased:
    """Same evidence for both: a Bayesian linear model of survival. They differ only in how the next experiment is chosen:
    posterior sampling acts on a sampled model; the hypothesis-style agent maximises expected value + lam * information gain."""

    def __init__(self, mode, lam=0.15):
        self.mode, self.lam = mode, lam
        self.name = {"psrl": "psrl", "ev": "hyp_ev"}[mode] if lam > 0 or mode == "psrl" else "hyp_ev_noig"

    def run(self, sess, rng):
        spec, post = sess.spec, Posterior(sess.spec)
        d0 = spec.default(); s, n = sess.deploy(d0, BATCH); post.add(d0, s, n); post.fit()
        for _ in range(N_DEPLOY - 1):
            pool = _pool(spec, rng, [self._best(spec, post, rng)])
            F = np.array([spec.features(d) for d in pool])
            if self.mode == "psrl":
                w = rng.multivariate_normal(post.mu, post.S)
                score = F @ w
            else:
                score = F @ post.mu + self.lam * 0.5 * np.log(1 + np.einsum("ij,jk,ik->i", F, post.S, F) / 0.012)
            d = pool[int(np.argmax(score))]
            s, n = sess.deploy(d, BATCH); post.add(d, s, n); post.fit()
        return self._best(spec, post, rng, big=True)

    @staticmethod
    def _best(spec, post, rng, big=False):
        pool = _pool(spec, rng, [spec.default()], n_rand=4000 if big else 600, n_local=600 if big else 100)
        F = np.array([spec.features(d) for d in pool])
        best = pool[int(np.argmax(F @ post.mu))]
        for _ in range(3 if big else 0):                         # local refinement around the incumbent
            loc = [spec.perturb(best, rng, 1) for _ in range(400)] + [best]
            best = loc[int(np.argmax([post.mean(d) for d in loc]))]
        return best


AGENTS = {"default": lambda: Default(), "random": lambda: RandomSearch(), "hillclimb": lambda: HillClimb(), "psrl": lambda: ModelBased("psrl"),
          "hyp_ev": lambda: ModelBased("ev", 0.15), "hyp_ev_noig": lambda: ModelBased("ev", 0.0)}
