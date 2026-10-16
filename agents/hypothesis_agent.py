"""Hypothesis-driven agent: explicit hypotheses about hidden bindings, tested by chosen experiments."""
from collections import defaultdict
from environment.world import DIRS
from .base_agent import BaseAgent, MOVES, passable_from_obs, bfs_tree, path_to
from .memory import EpisodicMemory, MemoryEntry
from .hypothesis_manager import HypothesisManager, RelationPrior, entropy
from .action_selector import ActionSelector

V_GATE, GAMMA, V_TARGET = 3.0, 0.7, 10.0


class HypothesisAgent(BaseAgent):
    name = "hypothesis"

    def __init__(self, seed=0, lam=0.5, mu=0.05, alpha=0.6, beta=0.6, reject_after=2,
                 revision=True, memory_capacity=None, transfer=False, selector="utility",
                 learn=True, label=None, use_sd=False, exclusive=True):
        super().__init__(seed)
        self.sel = ActionSelector(lam, mu, selector)
        self.cfg = dict(alpha=alpha, beta=beta, reject_after=reject_after, revision=revision, exclusive=exclusive)
        self.memory_capacity, self.transfer, self.learn, self.use_sd = memory_capacity, transfer, learn, use_sd
        self.exclusive, self._enum_only, self._cands = exclusive, False, []
        self.prior = RelationPrior()
        self.name = label or "hypothesis"
        self.reset()

    # ---------------------------------------------------------------- lifecycle
    def reset(self):
        self.memory = EpisodicMemory(self.memory_capacity)
        self.mgr = HypothesisManager(self.prior, self.transfer, **self.cfg)
        self.plan, self._cur = [], None
        self.log = []                 # full entry log (independent of memory capacity)
        self.counts = defaultdict(int)
        self.trace = []
        self.macro_log = []           # (start_step, macro_type, tag)

    def end_episode(self, obs):
        if self.transfer and self.learn:
            for e in self.log:
                if e.error:
                    continue
                rt = e.hypothesis[0]
                if rt == "PUSH":
                    self.prior.push_ok += 1
                else:
                    (self.prior.same if e.hypothesis[1] == e.hypothesis[2] else self.prior.diff)[rt] += 1
            self.prior.push_n += sum(1 for e in self.log if e.hypothesis[0] == "PUSH")

    def episode_stats(self):
        s = self.mgr.stats()
        seen, red = set(), 0
        for e in self.log:
            if e.hypothesis in seen and (e.hypothesis[0] != "PUSH" or e.error):
                red += 1
            seen.add(e.hypothesis)
        s.update(tests=len(self.log), redundant=red, explore_macros=self.counts["explore"],
                 exploit_macros=self.counts["exploit"],
                 tested=[(e.hypothesis, not e.error) for e in self.log])
        return s

    # ---------------------------------------------------------------- acting
    def act(self, obs):
        if not self.plan:
            self.plan = self._decide(obs)
        if not self.plan:
            self._cur = None
            return self.rng.choice(MOVES)
        a, kind, meta = self.plan.pop(0)
        self._cur = (a, kind, meta, obs)
        return a

    def observe(self, obs, action, nxt):
        if self._cur is None:
            return
        a, kind, meta, before = self._cur
        if (kind == "move" and nxt.feedback != "moved") or (kind == "bump" and nxt.feedback == "moved"):
            self.plan = []
            return
        if kind == "final":
            self._record(meta, before, nxt)

    def _record(self, meta, before, nxt):
        obs_sum = (before.agent_position, before.facing, before.inventory)
        t = meta["type"]
        if t == "TEST":
            ok = nxt.feedback == "door_opened"
            self._add(nxt, obs_sum, t, ("KD", meta["key"], meta["door"]), not ok)
        elif t == "SWITCH":
            before_inactive = {c for k, c, p, s in before.visible_objects if k == "chasm" and s == "inactive"}
            after_active = {c for k, c, p, s in nxt.visible_objects if k == "chasm" and s == "active"}
            for c in before_inactive:
                self._add(nxt, obs_sum, t, ("SB", meta["switch"], c), c not in after_active)
            if self.use_sd:
                closed = {c for k, c, p, s in before.visible_objects if k == "door" and s == "closed"}
                opened = {c for k, c, p, s in nxt.visible_objects if k == "door" and s == "open"}
                for c in closed:
                    self._add(nxt, obs_sum, t, ("SD", meta["switch"], c), c not in opened)
        elif t == "PUSH":
            self._add(nxt, obs_sum, t, ("PUSH", "block"), nxt.feedback != "pushed")

    def _add(self, nxt, obs_sum, t, hid, error):
        e = MemoryEntry(nxt.step, obs_sum, t, nxt.feedback, hid, True, error)
        self.memory.add(e)
        self.log.append(e)

    def _pick(self, macros):
        return self.sel.choose(macros, self.rng)

    def candidates(self, obs):
        """Enumerate the macro-actions available in `obs` (used by the learned macro-level agents)."""
        self._enum_only = True
        try:
            self._decide(obs)
        finally:
            self._enum_only = False
        return self._cands

    def commit(self, obs, idx):
        m = self._cands[idx]
        self._last_macro = m
        self.macro_log.append((obs.step, m["type"], m.get("key") or m.get("switch")))
        self.plan = list(m["plan"])
        return m

    # ---------------------------------------------------------------- probability hooks (overridden by PSRL)
    def _kd_ig(self, cand_keys, cand_sw, doors, pk):
        return pk

    def _sw_ig(self, ps, sc, chs, all_doors, cand_keys, cand_sw):
        return ps

    def _begin_decision(self, cand_keys, cand_sw, doors, chs):
        pass

    def _kd_probs(self, cand_keys, cand_sw, doors):
        if self.use_sd:
            return {kc: {dc: self.mgr.open_prob("KD", kc, dc, cand_keys, cand_sw) for _, dc in doors} for kc in cand_keys}
        return {kc: {dc: self.mgr.bind_prob("KD", kc, dc, cand_keys) for _, dc in doors} for kc in cand_keys}

    def _sb_probs(self, sc, chs, cand_sw):
        return [self.mgr.bind_prob("SB", sc, cc, cand_sw) for cc in chs]

    def _sd_probs(self, sc, doors, cand_keys, cand_sw):
        if not self.use_sd:
            return []
        return [self.mgr.open_prob("SD", sc, dc, cand_keys, cand_sw) for _, dc in doors]

    # ---------------------------------------------------------------- planning
    def _decide(self, obs):
        pos, ok = obs.agent_position, passable_from_obs(obs)
        dist, prev = bfs_tree(pos, ok)
        objs = {o[2]: o for o in obs.visible_objects}
        held = obs.inventory[0] if obs.inventory else None
        keys = [(p, o[1]) for p, o in objs.items() if o[0] == "key"]
        doors = [(p, o[1]) for p, o in objs.items() if o[0] == "door" and o[3] == "closed"]
        sws = [(p, o[1]) for p, o in objs.items() if o[0] == "switch" and o[3] == "off"]
        chs = [o[1] for o in objs.values() if o[0] == "chasm" and o[3] == "inactive"]
        blocks = [p for p, o in objs.items() if o[0] == "block"]
        targets = [p for p, o in objs.items() if o[0] == "target"]
        self.mgr.rebuild(self.memory)
        cand_keys = {c for _, c in keys} | ({held} if held else set())
        cand_sw = {c for _, c in sws}
        macros = []

        def walk(goal):
            return [(a, "move", None) for a in path_to(prev, goal)]

        def face_plan(stand, tgt, final):
            """Walk to `stand`, make sure we face `tgt`, then perform final action."""
            steps = walk(stand)
            last = steps[-1][0] if steps else obs.facing
            d = next(m for m, (dr, dc) in DIRS.items() if (stand[0] + dr, stand[1] + dc) == tgt)
            if last != d:
                steps.append((d, "bump", None))
            return steps, d

        for t in targets:
            if t in dist:
                macros.append(dict(type="TARGET", reward=V_TARGET, ig=0, cost=dist[t],
                                   plan=walk(t)))
        # --- key tests / swaps
        all_doors = list(doors)
        doors = [(dp, dc) for dp, dc in doors
                 if any((dp[0] + dr, dp[1] + dc2) in dist for dr, dc2 in DIRS.values())]  # only reachable doors can be tested
        self._begin_decision(cand_keys, cand_sw, all_doors, chs)
        pk = self._kd_probs(cand_keys, cand_sw, doors)
        pig = self._kd_ig(cand_keys, cand_sw, doors, pk)
        if held:
            for dp, dc in doors:
                stands = [(dist[(dp[0] + dr, dp[1] + dc2)], (dp[0] + dr, dp[1] + dc2))
                          for dr, dc2 in DIRS.values() if (dp[0] + dr, dp[1] + dc2) in dist]
                if not stands:
                    continue
                dd, s = min(stands)
                p = pk[held][dc]
                steps, _ = face_plan(s, dp, None)
                steps.append(("INTERACT", "final", dict(type="TEST", key=held, door=dc)))
                macros.append(dict(type="TEST", reward=V_GATE * p, ig=entropy(pig[held][dc]), cost=dd + 2, plan=steps, key=held, door=dc))
        if doors:
            for kp, kc in keys:
                if kp not in dist or kc == held:
                    continue
                base = {dc: (pk[held][dc] if held else 0.0) for _, dc in doors}
                gain = max(max(0.0, pk[kc][dc] - base[dc]) for _, dc in doors)
                igv = 0.5 * max(entropy(pig[kc][dc]) for _, dc in doors)
                steps = walk(kp) + [("INTERACT", "final", dict(type="PICK", key=kc))]
                macros.append(dict(type="PICK", reward=GAMMA * V_GATE * gain, ig=igv, cost=dist[kp] + 1, plan=steps, key=kc))
        # --- switches
        for sp, sc in sws:
            if sp not in dist or not (chs or (self.use_sd and all_doors)):
                continue
            ps = self._sb_probs(sc, chs, cand_sw) + self._sd_probs(sc, all_doors, cand_keys, cand_sw)
            steps = walk(sp) + [("INTERACT", "final", dict(type="SWITCH", switch=sc))]
            macros.append(dict(type="SWITCH", reward=V_GATE * max(ps or [0]), ig=sum(entropy(p) for p in self._sw_ig(ps, sc, chs, all_doors, cand_keys, cand_sw)),
                               cost=dist[sp] + 1, plan=steps, switch=sc))
        # --- blocks
        for bp in blocks:
            best = None
            for m, (dr, dc2) in DIRS.items():
                s, beyond = (bp[0] - dr, bp[1] - dc2), (bp[0] + dr, bp[1] + dc2)
                n = obs.grid.shape[0]
                if s in dist and 0 <= beyond[0] < n and 0 <= beyond[1] < n and obs.grid[beyond] == 0:
                    if best is None or dist[s] < best[0]:
                        best = (dist[s], s)
            if best:
                steps, _ = face_plan(best[1], bp, None)
                steps.append(("PUSH", "final", dict(type="PUSH")))
                p = self.mgr.push_prob()
                macros.append(dict(type="PUSH", reward=V_GATE * p, ig=entropy(p), cost=best[0] + 2, plan=steps))
        # trace for figures
        self.trace.append((obs.step, {"KD:%s>%s" % (k, d): pk[k][d] for k in pk for d in pk[k]}))
        if not macros:
            return []
        # drop experiments with negligible expected value (e.g. already-refuted hypotheses)
        useful = [x for x in macros if x["type"] == "TARGET" or x["reward"] >= 0.1 or x["ig"] >= 0.2]
        if useful:
            macros = useful
        if self._enum_only:
            self._cands = macros
            return []
        m = self._pick(macros)
        self._last_macro = m
        last = m["plan"][-1][2] if m["plan"] and m["plan"][-1][2] else {}
        self.macro_log.append((obs.step, m["type"], last.get("key") or last.get("switch")))
        self.counts["explore" if self.sel.lam * m["ig"] > m["reward"] else "exploit"] += 1
        return list(m["plan"])
