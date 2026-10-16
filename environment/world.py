"""Grid world. Positions are (row, col). Deterministic by default; spec.slip > 0 makes a correct key fail to open its door with that
probability (a jammed lock: the feedback is identical to using a wrong key), drawn from an RNG seeded by the environment seed."""
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple
import numpy as np
from .rules import HiddenRules

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "PUSH", "WAIT")
DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
CODES = {"empty": 0, "wall": 1, "key": 2, "door": 3, "switch": 4,
         "chasm": 5, "block": 6, "trap": 7, "target": 8, "agent": 9}


@dataclass
class Obj:
    kind: str
    color: str = ""
    state: str = ""


@dataclass(frozen=True)
class EnvironmentSpec:
    environment_id: str
    seed: int
    difficulty: int
    grid_size: int
    walls: FrozenSet[Tuple[int, int]]
    objects: Dict[Tuple[int, int], Obj]
    agent_start: Tuple[int, int]
    hidden: HiddenRules
    max_steps: int = 200
    slip: float = 0.0


@dataclass(frozen=True)
class Observation:
    grid: np.ndarray
    agent_position: Tuple[int, int]
    facing: str
    inventory: Tuple[str, ...]
    visible_objects: Tuple[Tuple[str, str, Tuple[int, int], str], ...]
    available_actions: Tuple[str, ...]
    reward: float
    done: bool
    success: bool
    feedback: str
    step: int


class World:
    def __init__(self, spec: EnvironmentSpec):
        self.spec = spec
        self._hidden = spec.hidden
        self.reset()

    def reset(self) -> Observation:
        self.objs = {p: Obj(o.kind, o.color, o.state) for p, o in self.spec.objects.items()}
        self.pos = self.spec.agent_start
        self.facing = "RIGHT"
        self.inventory = set()
        self.t = 0
        self._rng = np.random.default_rng(self.spec.seed * 7919 + 13)
        self.done = False
        self.success = False
        self.events = []  # evaluator-only log (never exposed to agents)
        return self._obs(0.0, "reset")

    def _free(self, p):
        n = self.spec.grid_size
        return 0 <= p[0] < n and 0 <= p[1] < n and p not in self.spec.walls

    def step(self, action: str) -> Observation:
        if action not in ACTIONS:
            raise ValueError(f"invalid action {action}")
        if self.done:
            return self._obs(0.0, "episode_over")
        self.t += 1
        reward, fb = 0.0, "nothing_happens"
        if action in DIRS:
            self.facing = action
            dr, dc = DIRS[action]
            tgt = (self.pos[0] + dr, self.pos[1] + dc)
            if not self._free(tgt):
                fb = "blocked_wall"
            else:
                o = self.objs.get(tgt)
                if o is None or o.kind in ("key", "switch"):
                    self.pos, fb = tgt, "moved"
                elif o.kind == "door":
                    if o.state == "open":
                        self.pos, fb = tgt, "moved"
                    else:
                        fb = "blocked_door"
                elif o.kind == "chasm":
                    if o.state == "active":
                        self.pos, fb = tgt, "moved"
                    else:
                        fb = "blocked_chasm"
                elif o.kind == "block":
                    fb = "blocked_block"
                elif o.kind == "trap":
                    self.pos, fb, reward, self.done = tgt, "trap", -1.0, True
                elif o.kind == "target":
                    self.pos, fb, reward = tgt, "reached_target", 1.0
                    self.done = self.success = True
        elif action == "INTERACT":
            here = self.objs.get(self.pos)
            dr, dc = DIRS[self.facing]
            front = self.objs.get((self.pos[0] + dr, self.pos[1] + dc))
            fpos = (self.pos[0] + dr, self.pos[1] + dc)
            if here is not None and here.kind == "key":
                old = next(iter(self.inventory), None)  # inventory capacity 1: picking swaps
                self.inventory = {here.color}
                del self.objs[self.pos]
                fb = "picked_key"
                if old is not None:
                    self.objs[self.pos] = Obj("key", old)
                    fb = "swapped_key"
            elif here is not None and here.kind == "switch" and here.state == "off":
                here.state = "on"
                fb = "switch_toggled"
                target_color = self._hidden.switch_bridge.get(here.color)
                act = []
                for o in self.objs.values():
                    if o.kind == "chasm" and o.color == target_color and o.state != "active":
                        o.state = "active"; act.append(o.color)
                tdoor = self._hidden.switch_door.get(here.color)
                for o in self.objs.values():
                    if o.kind == "door" and o.color == tdoor and o.state == "closed":
                        o.state = "open"; act.append(o.color)
                self.events.append((self.t, "switch_toggled", self.pos, tuple(act)))
            elif front is not None and front.kind == "door" and front.state == "closed":
                matches = any(front.color == self._hidden.key_door.get(k) or front.color in self._hidden.key_door_extra.get(k, ()) for k in self.inventory)
                if matches and self.spec.slip > 0 and self._rng.random() < self.spec.slip:
                    matches = False                      # jammed lock: a correct key fails
                if matches:
                    front.state = "open"
                    fb = "door_opened"
                    self.events.append((self.t, "door_opened", fpos, ()))
                else:
                    fb = "door_locked"
                    self.events.append((self.t, "door_locked", fpos, ()))
        elif action == "PUSH":
            dr, dc = DIRS[self.facing]
            fp = (self.pos[0] + dr, self.pos[1] + dc)
            bp = (fp[0] + dr, fp[1] + dc)
            f = self.objs.get(fp)
            if f is not None and f.kind == "block" and self._free(bp) and bp not in self.objs:
                self.objs[bp] = self.objs.pop(fp)
                self.pos, fb = fp, "pushed"
                self.events.append((self.t, "pushed", fp, ()))
            elif f is not None and f.kind == "block":
                fb = "push_failed"
                self.events.append((self.t, "push_failed", fp, ()))
        if self.t >= self.spec.max_steps:
            self.done = True
        return self._obs(reward, fb)

    def _obs(self, reward, fb) -> Observation:
        n = self.spec.grid_size
        grid = np.zeros((n, n), dtype=np.int8)
        for (r, c) in self.spec.walls:
            grid[r, c] = CODES["wall"]
        vis = []
        for p, o in sorted(self.objs.items()):
            grid[p] = CODES[o.kind]
            vis.append((o.kind, o.color, p, o.state))
        grid[self.pos] = CODES["agent"]
        return Observation(grid, self.pos, self.facing, tuple(sorted(self.inventory)),
                           tuple(vis), ACTIONS, reward, self.done, self.success, fb, self.t)
