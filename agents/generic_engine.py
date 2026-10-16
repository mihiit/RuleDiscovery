"""Generic skill library (go-to-object, INTERACT, PUSH): no key/door/switch testing semantics are given to the agent."""
from environment.world import DIRS
from .base_agent import passable_from_obs, bfs_tree, path_to
from .hypothesis_agent import HypothesisAgent


class GenericEngine(HypothesisAgent):
    def __init__(self, seed=0, **kw):
        kw.setdefault("label", "generic_engine")
        super().__init__(seed, **kw)

    def candidates(self, obs):
        pos, ok = obs.agent_position, passable_from_obs(obs)
        dist, prev = bfs_tree(pos, ok)
        n = obs.grid.shape[0]
        walk = lambda g: [(a, "move", None) for a in path_to(prev, g)]
        out = []
        for p, o in {o[2]: o for o in obs.visible_objects}.items():
            kind, color, state = o[0], o[1], o[3]
            if kind in ("key", "switch", "target") and p in dist and dist[p] > 0 and not (kind == "switch" and state == "on"):
                out.append(dict(type={"key": "GOTO_KEY", "switch": "GOTO_SWITCH", "target": "GOTO_TARGET"}[kind], color=color, cost=dist[p], plan=walk(p)))
            elif (kind == "door" and state == "closed") or kind == "block":
                best = None
                for m, (dr, dc) in DIRS.items():
                    s = (p[0] - dr, p[1] - dc)
                    if s not in dist:
                        continue
                    if kind == "block":
                        b = (p[0] + dr, p[1] + dc)
                        if not (0 <= b[0] < n and 0 <= b[1] < n and obs.grid[b] == 0):
                            continue
                    if best is None or dist[s] < best[0]:
                        best = (dist[s], s, m)
                if best:
                    d0, s, m = best
                    steps = walk(s)
                    last = steps[-1][0] if steps else obs.facing
                    if last != m:
                        steps.append((m, "bump", None))
                    if steps:
                        out.append(dict(type="GOTO_DOOR" if kind == "door" else "GOTO_BLOCK", color=color, cost=d0 + 1, plan=steps))
        out.append(dict(type="INTERACT", color="", cost=1, plan=[("INTERACT", "final", {"type": "PRIM"})]))
        out.append(dict(type="PUSH", color="", cost=1, plan=[("PUSH", "final", {"type": "PRIM"})]))
        self._cands = out
        return out
