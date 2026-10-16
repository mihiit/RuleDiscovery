"""Macro enumeration shared with the hypothesis agent, with all belief information removed (constant probabilities)."""
from .hypothesis_agent import HypothesisAgent


class MacroEngine(HypothesisAgent):
    def __init__(self, seed=0, **kw):
        kw.setdefault("label", "macro_engine")
        super().__init__(seed, **kw)

    def reset(self):
        super().reset()
        self.mgr.push_prob = lambda: 0.5

    def _kd_probs(self, cand_keys, cand_sw, doors):
        return {kc: {dc: 0.5 for _, dc in doors} for kc in cand_keys}

    def _sb_probs(self, sc, chs, cand_sw):
        return [0.5 for _ in chs]

    def _sd_probs(self, sc, doors, cand_keys, cand_sw):
        return [0.5 for _ in doors] if self.use_sd else []
