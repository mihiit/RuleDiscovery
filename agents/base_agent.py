import random
from collections import deque
from environment.world import DIRS

MOVES = tuple(DIRS)


class BaseAgent:
    name = "base"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def reset(self):
        pass

    def act(self, obs):
        raise NotImplementedError

    def observe(self, obs, action, next_obs):
        pass

    def end_episode(self, obs):
        pass


def passable_from_obs(obs):
    objs = {o[2]: o for o in obs.visible_objects}
    n = obs.grid.shape[0]

    def ok(p):
        if not (0 <= p[0] < n and 0 <= p[1] < n) or obs.grid[p] == 1:
            return False
        o = objs.get(p)
        if o is None or o[0] in ("key", "switch", "target"):
            return True
        return (o[0] == "door" and o[3] == "open") or (o[0] == "chasm" and o[3] == "active")
    return ok


def first_step(start, goals, ok):
    """BFS; return first move toward the nearest goal (None if unreachable / already there)."""
    if start in goals:
        return None
    prev, q = {start: None}, deque([start])
    while q:
        cur = q.popleft()
        for a, (dr, dc) in DIRS.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in prev or not ok(nxt):
                continue
            prev[nxt] = (cur, a)
            if nxt in goals:
                while prev[nxt][0] != start:
                    nxt = prev[nxt][0]
                return prev[nxt][1]
            q.append(nxt)
    return None


def bfs_tree(start, ok):
    """Return (dist, prev) over passable cells from start."""
    dist, prev, q = {start: 0}, {start: None}, deque([start])
    while q:
        cur = q.popleft()
        for a, (dr, dc) in DIRS.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in dist or not ok(nxt):
                continue
            dist[nxt] = dist[cur] + 1
            prev[nxt] = (cur, a)
            q.append(nxt)
    return dist, prev


def path_to(prev, goal):
    acts = []
    while prev[goal] is not None:
        goal, a = prev[goal]
        acts.append(a)
    return acts[::-1]
