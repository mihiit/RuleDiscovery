"""Posterior-sampling (Thompson-style) macro agent: a Bayesian model-based baseline.
Evidence is the same as the hypothesis agent's, but there are no confidences, no alpha/beta updates and no information-gain
term: at each evidence change it samples ONE complete binding consistent with all outcomes so far and acts as if it were true
(PSRL, Strens 2000; Osband et al. 2013), with nearest-first macro planning (lambda = 0)."""
from .hypothesis_agent import HypothesisAgent


class PSRLAgent(HypothesisAgent):
    def __init__(self, seed=0, ig_from_posterior=False, **kw):
        self.ig_post = ig_from_posterior
        kw.setdefault("lam", 0.0)
        kw.setdefault("label", "psrl")
        super().__init__(seed, **kw)

    def reset(self):
        super().reset()
        self._sample = {}

    def _decide(self, obs):
        """Resample when the sampled model offers no executable progress (e.g. it assigned an unreachable key)."""
        for _ in range(8):
            plan = super()._decide(obs)
            m = getattr(self, "_last_macro", None)
            if not plan or m is None or m["type"] == "TARGET" or m["reward"] > 0:
                return plan
            self._sample = {}
            if self.macro_log:
                self.macro_log.pop()
        return plan

    def _begin_decision(self, cand_keys, cand_sw, doors, chs):
        """Sticky posterior sample: keep earlier assignments while consistent with the evidence; resample only the rest."""
        cand_keys, cand_sw = sorted(cand_keys), sorted(cand_sw)
        bad = {e.hypothesis for e in self.log if e.error}
        good = {e.hypothesis for e in self.log if not e.error}
        targets = [("D", c) for _, c in doors] + [("C", c) for c in chs]
        smp, used_k, used_s = {}, set(), set()

        def hid(kind, t, o):
            return (("KD", o[1], t) if o[0] == "KD" else (("SD" if kind == "D" else "SB"), o[1], t))

        def put(kind, t, o):
            smp[(kind, t)] = o
            if self.exclusive:
                (used_k if o[0] == "KD" else used_s).add(o[1])

        for kind, t in targets:                      # confirmed bindings
            for k in cand_keys:
                if kind == "D" and ("KD", k, t) in good:
                    put(kind, t, ("KD", k))
            for s_ in cand_sw:
                if (kind == "C" and ("SB", s_, t) in good) or (kind == "D" and ("SD", s_, t) in good):
                    put(kind, t, ("S", s_))
        for kind, t in targets:                      # keep still-consistent earlier choices
            o = self._sample.get((kind, t))
            if (kind, t) in smp or o is None:
                continue
            pool = cand_keys if o[0] == "KD" else cand_sw
            if o[1] in pool and hid(kind, t, o) not in bad and o[1] not in (used_k if o[0] == "KD" else used_s):
                put(kind, t, o)
        rest = [x for x in targets if x not in smp]
        self.rng.shuffle(rest)
        for kind, t in rest:
            opts = []
            if kind == "D":
                opts += [("KD", k) for k in cand_keys if k not in used_k and ("KD", k, t) not in bad]
                if self.use_sd:
                    opts += [("S", s_) for s_ in cand_sw if s_ not in used_s and ("SD", s_, t) not in bad]
            else:
                opts += [("S", s_) for s_ in cand_sw if s_ not in used_s and ("SB", s_, t) not in bad]
            if opts:
                put(kind, t, self.rng.choice(opts))
        self._sample = smp

    def _kd_ig(self, cand_keys, cand_sw, doors, pk):
        # hybrid used in the factorial ablation: sampled-binding reward + information gain from the posterior marginals
        return HypothesisAgent._kd_probs(self, cand_keys, cand_sw, doors) if self.ig_post else pk

    def _sw_ig(self, ps, sc, chs, all_doors, cand_keys, cand_sw):
        if not self.ig_post:
            return ps
        return HypothesisAgent._sb_probs(self, sc, chs, cand_sw) + HypothesisAgent._sd_probs(self, sc, all_doors, cand_keys, cand_sw)

    def _kd_probs(self, cand_keys, cand_sw, doors):
        return {kc: {dc: float(self._sample.get(("D", dc)) == ("KD", kc)) for _, dc in doors} for kc in cand_keys}

    def _sb_probs(self, sc, chs, cand_sw):
        return [float(self._sample.get(("C", cc)) == ("S", sc)) for cc in chs]

    def _sd_probs(self, sc, doors, cand_keys, cand_sw):
        if not self.use_sd:
            return []
        return [float(self._sample.get(("D", dc)) == ("S", sc)) for _, dc in doors]
