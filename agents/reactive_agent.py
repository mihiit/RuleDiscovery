"""Reactive heuristic agent. Uses only the current observation plus one lose-shift bit;
it keeps NO hypothesis database and no record of which key/door pairs were tried."""
from environment.world import DIRS
from .base_agent import BaseAgent, MOVES, passable_from_obs, first_step


class ReactiveAgent(BaseAgent):
    name = "reactive"

    def __init__(self, seed=0, epsilon=0.1):
        super().__init__(seed)
        self.epsilon = epsilon
        self.shift = False

    def reset(self):
        self.shift = False

    def act(self, obs):
        if obs.feedback == "door_locked":
            self.shift = True
        elif obs.feedback in ("picked_key", "swapped_key", "door_opened"):
            self.shift = False
        if self.rng.random() < self.epsilon:
            return self.rng.choice(obs.available_actions)
        objs = {o[2]: o for o in obs.visible_objects}
        pos, ok = obs.agent_position, passable_from_obs(obs)
        held = obs.inventory[0] if obs.inventory else None
        here = objs.get(pos)
        if here and here[0] == "key" and (held is None or (self.shift and here[1] != held)):
            return "INTERACT"
        if here and here[0] == "switch" and here[3] == "off":
            return "INTERACT"
        targets = {p for p, o in objs.items() if o[0] == "target"}
        a = first_step(pos, targets, ok)
        if a:
            return a
        dr, dc = DIRS[obs.facing]
        front = objs.get((pos[0] + dr, pos[1] + dc))
        if front and front[0] == "door" and front[3] == "closed" and held and not self.shift:
            return "INTERACT"
        if front and front[0] == "block":
            return "PUSH"
        keys = [p for p, o in objs.items() if o[0] == "key" and (held is None or (self.shift and o[1] != held))]
        sws = [p for p, o in objs.items() if o[0] == "switch" and o[3] == "off"]
        goals = set(sws)
        if keys:
            goals |= {self.rng.choice(keys)} if self.shift else set(keys)
        a = first_step(pos, goals, ok)
        if a:
            return a
        for m, (dr, dc) in DIRS.items():
            o = objs.get((pos[0] + dr, pos[1] + dc))
            if o and ((o[0] == "door" and o[3] == "closed" and held) or o[0] == "block"):
                return m
        blockers = {p for p, o in objs.items() if (o[0] == "door" and o[3] == "closed" and held) or o[0] == "block"}
        adj = {(p[0] + dr, p[1] + dc) for p in blockers for dr, dc in DIRS.values()}
        a = first_step(pos, {x for x in adj if ok(x)}, ok)
        return a or self.rng.choice(MOVES)
