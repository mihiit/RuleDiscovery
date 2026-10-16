import math
from dataclasses import dataclass


@dataclass
class Hypothesis:
    rule_type: str
    subject: tuple
    condition: str
    action: str
    predicted_effect: bool
    confidence: float
    evidence_count: int = 0
    contradiction_count: int = 0
    rejected: bool = False
    confirmed: bool = False

    @property
    def hid(self):
        return (self.rule_type,) + tuple(self.subject)


def entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


class RelationPrior:
    """Cross-episode beliefs (only used when transfer is enabled): does a rule bind
    same-colored objects, and do push actions work."""

    def __init__(self):
        self.same = {"KD": 0, "SB": 0}
        self.diff = {"KD": 0, "SB": 0}
        self.push_ok = 0
        self.push_n = 0

    def p_same(self, rt):
        return (self.same[rt] + 1) / (self.same[rt] + self.diff[rt] + 2)

    def push_prior(self):
        return (self.push_ok + 1) / (self.push_n + 2)


class HypothesisManager:
    def __init__(self, prior, use_prior=False, alpha=0.6, beta=0.6, reject_after=2, revision=True, exclusive=True, rho=0.05):
        # exclusive: True (hard: a key already confirmed for one door is excluded elsewhere), False (no assumption),
        # or "adaptive" (soft prior: such a key keeps weight rho; the assumption is dropped once a violation is observed).
        self.rho = rho
        self.prior, self.use_prior = prior, use_prior
        self.alpha, self.beta, self.reject_after, self.revision, self.exclusive = alpha, beta, reject_after, revision, exclusive
        self.h = {}
        self.generated = set()

    def initial_conf(self, rt, subject):
        if not self.use_prior:
            return 0.5
        if rt == "PUSH":
            return self.prior.push_prior()
        p = self.prior.p_same("SB" if rt == "SD" else rt)
        return p if subject[0] == subject[1] else 1 - p

    def get(self, hid):
        if hid not in self.h:
            rt, subj = hid[0], hid[1:]
            self.h[hid] = Hypothesis(rt, subj, "interact", "effect", True, self.initial_conf(rt, subj))
        self.generated.add(hid)
        return self.h[hid]

    def update_hypothesis(self, h, prediction_correct):
        h.evidence_count += 1
        if prediction_correct:
            h.confidence = min(1.0, h.confidence + self.alpha)
            h.confirmed = True
        else:
            h.contradiction_count += 1
            if self.revision:
                h.confidence = max(0.0, h.confidence - self.beta)
                if h.contradiction_count >= self.reject_after:
                    h.rejected = True

    def rebuild(self, memory):
        """Recompute hypothesis state from priors + whatever the memory still holds."""
        self.h = {}
        for e in memory.entries:
            self.update_hypothesis(self.get(e.hypothesis), not e.error)

    def _confirmed(self, rt):
        return [h for h in self.h.values() if h.rule_type == rt and h.confirmed]

    def bind_prob(self, rt, a, b, candidates_a):
        """P(a binds b) normalised over candidate a's for fixed b (a: key/switch color, b: door/chasm color)."""
        conf = self._confirmed(rt)
        for h in conf:
            if h.subject[1] == b:
                return 1.0 if h.subject[0] == a else 0.0
        used, soft = {h.subject[0] for h in conf} if self.exclusive else set(), False
        if self.exclusive == "adaptive":
            if self.violated(rt):           # evidence that one key/switch binds several targets: drop the assumption
                used = set()
            else:
                soft = True
        if a in used and not soft:
            return 0.0
        w = {}
        for k in candidates_a:
            if k in used and not soft:
                continue
            h = self.get((rt, k, b))
            w[k] = 0.0 if h.rejected else max(h.confidence, 1e-3)
            if soft and k in used:
                w[k] *= self.rho
        tot = sum(w.values())
        if a not in w:
            return 0.0
        return w[a] / tot if tot > 0 else 1.0 / max(1, len(w))

    def violated(self, rt):
        """True once some key (switch) has been confirmed to bind two different targets."""
        seen = {}
        for h in self._confirmed(rt):
            seen.setdefault(h.subject[0], set()).add(h.subject[1])
        return any(len(v) > 1 for v in seen.values())

    def open_prob(self, rt, a, d, cand_keys, cand_sw):
        """Open-vocabulary version: a door may be opened by a KEY (KD) or by a SWITCH (SD); normalise jointly."""
        conf = [h for h in self.h.values() if h.rule_type in ("KD", "SD") and h.confirmed]
        for h in conf:
            if h.subject[1] == d:
                return 1.0 if (h.rule_type, h.subject[0]) == (rt, a) else 0.0
        used_k = {h.subject[0] for h in conf if h.rule_type == "KD"} if self.exclusive else set()
        used_s = {h.subject[0] for h in self.h.values() if h.rule_type in ("SD", "SB") and h.confirmed} if self.exclusive else set()
        w = {}
        for k in cand_keys:
            if k not in used_k:
                h = self.get(("KD", k, d)); w[("KD", k)] = 0.0 if h.rejected else max(h.confidence, 1e-3)
        for sw in cand_sw:
            if sw not in used_s:
                h = self.get(("SD", sw, d)); w[("SD", sw)] = 0.0 if h.rejected else max(h.confidence, 1e-3)
        tot = sum(w.values())
        return w.get((rt, a), 0.0) / tot if tot > 0 else 0.0

    def push_prob(self):
        h = self.get(("PUSH", "block"))
        return 0.0 if h.rejected else max(h.confidence, 1e-3)

    def stats(self):
        hs = list(self.h.values())
        return {"generated": len(self.generated), "rejected": sum(h.rejected for h in hs),
                "retained": sum(not h.rejected for h in hs),
                "contradictions": sum(h.contradiction_count for h in hs)}
