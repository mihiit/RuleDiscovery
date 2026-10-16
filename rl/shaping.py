"""Potential-based reward shaping (Ng, Harada & Russell 1999) used for RL baselines' TRAINING only.
Phi(s) = -0.1 * distance to the target when it is reachable, otherwise -1 - 0.1 * distance to the nearest useful sub-goal
(unused switch, a key when empty-handed, or a cell next to a closed door/block). Because it is a function of state,
r' = r + gamma*Phi(s') - Phi(s) leaves the optimal policy unchanged."""
from environment.world import DIRS
from agents.base_agent import passable_from_obs, bfs_tree


def potential(obs):
    ok = passable_from_obs(obs)
    dist, _ = bfs_tree(obs.agent_position, ok)
    objs = {o[2]: o for o in obs.visible_objects}
    t = [dist[p] for p, o in objs.items() if o[0] == "target" and p in dist]
    if t:
        return -0.1 * min(t)
    held = bool(obs.inventory)
    g = []
    for p, o in objs.items():
        if p in dist and ((o[0] == "switch" and o[3] == "off") or (o[0] == "key" and not held)):
            g.append(dist[p])
        if (o[0] == "block" or (o[0] == "door" and o[3] == "closed" and held)):
            g += [dist[(p[0] + dr, p[1] + dc)] for dr, dc in DIRS.values() if (p[0] + dr, p[1] + dc) in dist]
    return -1.0 - 0.1 * (min(g) if g else 20)
