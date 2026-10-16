"""Procedural, solvable-by-construction environment generator."""
import random
from dataclasses import dataclass
from typing import Optional, Tuple
from .rules import (KEY_DOOR, SWITCH_BRIDGE, PUSH_BLOCK, SWITCH_DOOR, TRAP, GATE_RULES, COLORS, HiddenRules)
from .world import Obj, EnvironmentSpec, DIRS

DEFAULT_SIZE = {1: 8, 2: 10, 3: 10, 4: 12, 5: 14}
DEFAULT_RULES = {1: (KEY_DOOR,), 2: (KEY_DOOR, SWITCH_BRIDGE),
                 3: (KEY_DOOR, SWITCH_BRIDGE, PUSH_BLOCK),
                 4: (KEY_DOOR, SWITCH_BRIDGE, PUSH_BLOCK, KEY_DOOR),
                 5: (KEY_DOOR, SWITCH_BRIDGE, PUSH_BLOCK, KEY_DOOR, SWITCH_BRIDGE)}


@dataclass(frozen=True)
class EnvironmentConfig:
    seed: int = 0
    difficulty: int = 3
    rules: Optional[Tuple[str, ...]] = None
    grid_size: Optional[int] = None
    max_steps: int = 200
    colors: Tuple[str, ...] = COLORS
    shared_keys: bool = False   # second key-door gate reuses the first gate's key (assumption-violation test)
    mapping: str = "random"   # random | identity (key/switch color == door/chasm color) | anti (never equal)


def _zone(j, cols, size):
    lo = 0 if j == 0 else cols[j - 1] + 1
    hi = cols[j] - 1 if j < len(cols) else size - 1
    return lo, hi


def is_solvable(spec: EnvironmentSpec) -> bool:
    n, walls, objs, h = spec.grid_size, spec.walls, spec.objects, spec.hidden
    opened, keys, dead = set(), set(), set()

    def ok(p):
        if not (0 <= p[0] < n and 0 <= p[1] < n) or p in walls or p in dead:
            return False
        o = objs.get(p)
        if o is None or o.kind in ("key", "switch", "target"):
            return True
        return o.kind in ("door", "chasm", "block") and p in opened

    def flood():
        seen, stack = {spec.agent_start}, [spec.agent_start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS.values():
                q = (r + dr, c + dc)
                if q not in seen and ok(q):
                    seen.add(q); stack.append(q)
        return seen

    while True:
        reach = flood()
        changed = False
        for p in reach:
            o = objs.get(p)
            if o is None:
                continue
            if o.kind == "key" and o.color not in keys:
                keys.add(o.color); changed = True
            if o.kind == "switch":
                for q, oq in objs.items():
                    if q in opened:
                        continue
                    if (oq.kind == "chasm" and oq.color == h.switch_bridge.get(o.color)) or \
                            (oq.kind == "door" and oq.color == h.switch_door.get(o.color)):
                        opened.add(q); changed = True
        for q, o in objs.items():
            if q in opened:
                continue
            nbrs = [(q[0] + dr, q[1] + dc) for dr, dc in DIRS.values()]
            if o.kind == "door" and any(x in reach for x in nbrs) and \
                    any(o.color == h.key_door.get(k) or o.color in h.key_door_extra.get(k, ()) for k in keys):
                opened.add(q); changed = True
            if o.kind == "block" and (q[0], q[1] - 1) in reach:
                b1, b2 = (q[0], q[1] + 1), (q[0], q[1] + 2)
                if all(0 <= x[1] < n and x not in walls and x not in objs for x in (b1, b2)):
                    opened.add(q); dead.add(b2); changed = True
        if not changed:
            break
    return any(o.kind == "target" and p in reach for p, o in objs.items())


def _pick(mapping, target, pool, rng):
    if mapping == "identity":
        pool.remove(target); return target
    if mapping == "anti":
        cand = [c for c in pool if c != target]
        if not cand:
            raise ValueError
        c = rng.choice(cand); pool.remove(c); return c
    return pool.pop()


def _build(cfg, rules, size, k, rng):
    d = cfg.difficulty
    gates = [r for r in rules if r in GATE_RULES]
    rng.shuffle(gates)
    cols = [round((i + 1) * size / (k + 1)) for i in range(k)]
    if k and (len(set(cols)) < k or cols[-1] >= size - 1):
        return None
    walls, objs, reserved, gpos = set(), {}, set(), []
    for c in cols:
        r = rng.randrange(1, size - 1)
        walls |= {(rr, c) for rr in range(size) if rr != r}
        gpos.append((r, c))
        reserved |= {(r, c - 1), (r, c + 1), (r, c + 2)}
    pool = list(cfg.colors)
    door_cols, ch_cols = rng.sample(pool, len(pool)), rng.sample(pool, len(pool))
    key_cols, sw_cols = rng.sample(pool, len(pool)), rng.sample(pool, len(pool))
    kd, sb, sd, kd_extra, first_kd = {}, {}, {}, {}, None
    pending = []  # (item, gate_index)
    try:
        for i, (g, p) in enumerate(zip(gates, gpos)):
            if g == KEY_DOOR and cfg.shared_keys and first_kd is not None:
                dc = door_cols.pop()
                objs[p] = Obj("door", dc, "closed"); kd_extra[first_kd] = kd_extra.get(first_kd, ()) + (dc,)
            elif g == KEY_DOOR:
                dc = door_cols.pop()
                kc = _pick(cfg.mapping, dc, key_cols, rng)
                objs[p] = Obj("door", dc, "closed"); kd[kc] = dc
                first_kd = first_kd or kc
                pending.append((Obj("key", kc), i))
            elif g == SWITCH_BRIDGE:
                cc = ch_cols.pop()
                sc = _pick(cfg.mapping, cc, sw_cols, rng)
                objs[p] = Obj("chasm", cc, "inactive"); sb[sc] = cc
                pending.append((Obj("switch", sc, "off"), i))
            elif g == SWITCH_DOOR:
                dc = door_cols.pop()
                sc = _pick(cfg.mapping, dc, sw_cols, rng)
                objs[p] = Obj("door", dc, "closed"); sd[sc] = dc
                pending.append((Obj("switch", sc, "off"), i))
            else:
                objs[p] = Obj("block")
                if not all(0 <= x[1] < size and x not in walls for x in ((p[0], p[1] + 1), (p[0], p[1] + 2))):
                    return None
    except (IndexError, ValueError):
        return None
    occupied = set(walls) | set(objs)

    def free_in(lo, hi):
        return [(r, c) for r in range(size) for c in range(lo, hi + 1)
                if (r, c) not in occupied and (r, c) not in reserved]

    def place(obj, lo, hi):
        cells = free_in(lo, hi)
        if not cells:
            raise ValueError
        p = rng.choice(cells)
        objs[p] = obj; occupied.add(p)
        return p

    try:
        for item, i in pending:
            place(item, *_zone(rng.randint(0, i), cols, size))
        start = rng.choice(free_in(*_zone(0, cols, size)))
        occupied.add(start)
        place(Obj("target"), *_zone(k, cols, size))
        for _ in range(max(0, d - 2)):  # distractors
            if key_cols:
                place(Obj("key", key_cols.pop()), 0, size - 1)
            if sw_cols:
                place(Obj("switch", sw_cols.pop(), "off"), 0, size - 1)
        for _ in range(max(0, d - 1)):
            place(Obj("trap"), 0, size - 1)
    except ValueError:
        return None
    hidden = HiddenRules(tuple(rules) + ((TRAP,) if d > 1 and TRAP not in rules else ()), kd, sb, sd, kd_extra)
    return EnvironmentSpec(f"env_d{d}_s{cfg.seed}", cfg.seed, d, size, frozenset(walls),
                           objs, start, hidden, cfg.max_steps)


def generate(cfg: EnvironmentConfig) -> EnvironmentSpec:
    rules = cfg.rules or DEFAULT_RULES[cfg.difficulty]
    size = cfg.grid_size or DEFAULT_SIZE[cfg.difficulty]
    k = len([r for r in rules if r in GATE_RULES])
    for attempt in range(1000):
        rng = random.Random(cfg.seed * 7919 + attempt)
        spec = _build(cfg, rules, size, k, rng)
        if spec is not None and is_solvable(spec):
            return spec
    raise RuntimeError(f"could not generate environment for {cfg}")
