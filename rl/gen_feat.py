"""Features for the generic-skill agents: skills are GOTO(object) and the primitives INTERACT / PUSH; no TEST/PICK/SWITCH semantics."""
import numpy as np
from environment.rules import ALL_COLORS
from rl.encode import FEED, FI

CI = {c: i for i, c in enumerate(ALL_COLORS)}
TYPES = ["GOTO_KEY", "GOTO_SWITCH", "GOTO_DOOR", "GOTO_BLOCK", "GOTO_TARGET", "INTERACT", "PUSH"]
FACE = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
CAND_F, STATE_F, NMAX = 16, 82, 16


def cand_feat(m):
    v = np.zeros(CAND_F, np.float32)
    v[TYPES.index(m["type"])] = 1
    if m.get("color"): v[7 + CI[m["color"]]] = 1
    v[15] = min(m["cost"], 30) / 20
    return v


def featurize(obs, cands, last):
    cf = np.zeros((NMAX, CAND_F), np.float32); mask = np.zeros(NMAX, bool)
    for i, m in enumerate(cands[:NMAX]):
        cf[i] = cand_feat(m); mask[i] = True
    st = np.zeros(STATE_F, np.float32)
    held = obs.inventory[0] if obs.inventory else None
    st[CI[held] + 1 if held else 0] = 1
    objs = obs.visible_objects
    cnt = lambda f: sum(1 for o in objs if f(o))
    st[9] = cnt(lambda o: o[0] == "door" and o[3] == "closed") / 4; st[10] = cnt(lambda o: o[0] == "switch" and o[3] == "off") / 4
    st[11] = cnt(lambda o: o[0] == "key") / 4; st[12] = cnt(lambda o: o[0] == "block") / 2
    st[13] = obs.step / 200
    st[14 + FI.get(obs.feedback, FI["nothing_happens"])] = 1
    if last is not None:
        st[31 + TYPES.index(last["type"])] = 1
        if last.get("color"): st[38 + CI[last["color"]]] = 1
    st[46 + FACE[obs.facing]] = 1
    for o in objs:
        if o[0] == "door" and o[3] == "closed": st[50 + CI[o[1]]] = 1
        elif o[0] == "chasm" and o[3] == "inactive": st[58 + CI[o[1]]] = 1
        elif o[0] == "switch" and o[3] == "off": st[66 + CI[o[1]]] = 1
        elif o[0] == "key": st[74 + CI[o[1]]] = 1
    return cf, st, mask
