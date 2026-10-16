"""Features for the learned macro-level agents. NO hypothesis/probability information is exposed:
only what a policy could read off the observation (macro type, colors involved, path cost) plus the last macro and its outcome."""
import numpy as np
from environment.rules import ALL_COLORS

CI = {c: i for i, c in enumerate(ALL_COLORS)}
TYPES = ["TARGET", "TEST", "PICK", "SWITCH", "PUSH"]
CAND_F, STATE_F, NMAX = 22, 71, 16
CAND_FB = 24                      # matched-information variant appends (expected value, information gain) per candidate


def cand_feat(m, belief=False):
    v = np.zeros(CAND_FB if belief else CAND_F, np.float32)
    v[TYPES.index(m["type"])] = 1
    item = m.get("key") or m.get("switch")
    if item:
        v[5 + CI[item]] = 1
    if m.get("door"):
        v[13 + CI[m["door"]]] = 1
    v[21] = min(m["cost"], 30) / 20
    if belief:                      # matched-information variant: expected value and information gain from the hypothesis tracker
        v[22] = min(m["reward"], 10.0) / 10.0
        v[23] = min(m["ig"], 2.0) / 2.0
    return v


def outcome(macro, before, after):
    """0 neutral, 1 success, 2 failure -- computed from observations only (same code in training and evaluation)."""
    t = macro["type"]
    if t == "TEST":
        return 1 if after.feedback == "door_opened" else 2
    if t == "PUSH":
        return 1 if after.feedback == "pushed" else 2
    if t == "SWITCH":
        n = lambda o: sum(1 for k, c, p, s in o.visible_objects if (k == "chasm" and s == "active") or (k == "door" and s == "open"))
        return 1 if n(after) > n(before) else 2
    return 0


def featurize(obs, cands, last, belief=False):
    """last = (macro dict or None, outcome int)."""
    cf = np.zeros((NMAX, CAND_FB if belief else CAND_F), np.float32)
    mask = np.zeros(NMAX, bool)
    for i, m in enumerate(cands[:NMAX]):
        cf[i] = cand_feat(m, belief); mask[i] = True
    st = np.zeros(STATE_F, np.float32)
    held = obs.inventory[0] if obs.inventory else None
    st[CI[held] + 1 if held else 0] = 1
    objs = obs.visible_objects
    cnt = lambda f: sum(1 for o in objs if f(o))
    st[9] = cnt(lambda o: o[0] == "door" and o[3] == "closed") / 4
    st[10] = cnt(lambda o: o[0] == "switch" and o[3] == "off") / 4
    st[11] = cnt(lambda o: o[0] == "key") / 4
    st[12] = cnt(lambda o: o[0] == "block") / 2
    st[13] = float(any(m["type"] == "TARGET" for m in cands))
    st[14] = obs.step / 200
    lm, oc = last
    st[15 + oc] = 1
    if lm is not None:
        st[18 + TYPES.index(lm["type"])] = 1
        item = lm.get("key") or lm.get("switch")
        if item: st[23 + CI[item]] = 1
        if lm.get("door"): st[31 + CI[lm["door"]]] = 1
    for o in objs:
        if o[0] == "door" and o[3] == "closed": st[39 + CI[o[1]]] = 1
        elif o[0] == "chasm" and o[3] == "inactive": st[47 + CI[o[1]]] = 1
        elif o[0] == "switch" and o[3] == "off": st[55 + CI[o[1]]] = 1
        elif o[0] == "key": st[63 + CI[o[1]]] = 1
    return cf, st, mask
