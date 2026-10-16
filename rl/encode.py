"""Observation -> network inputs (shared by training and evaluation). Max grid 14x14, 8 colours."""
import numpy as np
from environment.rules import ALL_COLORS

G = 14
CI = {c: i for i, c in enumerate(ALL_COLORS)}
FEED = ["reset", "moved", "blocked_wall", "blocked_door", "blocked_chasm", "blocked_block", "trap", "reached_target",
        "picked_key", "swapped_key", "switch_toggled", "door_opened", "door_locked", "pushed", "push_failed",
        "nothing_happens", "episode_over"]
FI = {f: i for i, f in enumerate(FEED)}
ACT = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "PUSH", "WAIT"]
N_CELL, N_VEC = 40, 9 + 4 + 8 + len(FEED) + 1
FACE = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}


def encode(obs, last_action=None, max_steps=200):
    n = obs.grid.shape[0]
    cells = np.ones((G, G), np.int8)
    cells[:n, :n] = (obs.grid[:n, :n] == 1)  # walls=1, free=0
    for kind, color, (r, c), state in obs.visible_objects:
        ci = CI[color] if color else 0
        if kind == "target": v = 2
        elif kind == "trap": v = 3
        elif kind == "block": v = 4
        elif kind == "key": v = 8 + ci
        elif kind == "door": v = 5 if state == "open" else 16 + ci
        elif kind == "switch": v = 6 if state == "on" else 24 + ci
        else: v = 7 if state == "active" else 32 + ci
        cells[r, c] = v
    agent = np.zeros((G, G), np.float32)
    agent[obs.agent_position] = 1.0
    vec = np.zeros(N_VEC, np.float32)
    vec[CI[obs.inventory[0]] + 1 if obs.inventory else 0] = 1
    vec[9 + FACE[obs.facing]] = 1
    vec[13 + (ACT.index(last_action) if last_action else 7)] = 1
    vec[21 + FI.get(obs.feedback, FI["nothing_happens"])] = 1
    vec[-1] = obs.step / max_steps
    return cells, agent, vec
