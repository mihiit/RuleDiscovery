import time
from environment.world import World


def recovery_stats(events):
    """Failure events (locked door, dud switch, failed push) and whether the gate was later opened."""
    fails, recovered = {}, set()
    activated = []
    for t, kind, pos, extra in events:
        if kind == "door_locked":
            fails.setdefault(("door", pos), t)
        elif kind == "door_opened":
            recovered.add(("door", pos))
        elif kind == "switch_toggled":
            if extra:
                activated.append(t)
            else:
                fails.setdefault(("dud", pos), t)
        elif kind == "push_failed":
            fails.setdefault(("push", pos), t)
        elif kind == "pushed":
            recovered.add(("push", pos))
    n_rec = 0
    for key, t0 in fails.items():
        if key[0] in ("door", "push") and key in recovered:
            n_rec += 1
        elif key[0] == "dud" and any(t > t0 for t in activated):
            n_rec += 1
    return len(fails), n_rec


def run_episode(spec, agent, collect_trace=False):
    env = World(spec)
    obs = env.reset()
    agent.reset()
    t0 = time.process_time()
    while not obs.done:
        a = agent.act(obs)
        nxt = env.step(a)
        agent.observe(obs, a, nxt)
        obs = nxt
    agent.end_episode(obs)
    cpu = time.process_time() - t0
    nf, nr = recovery_stats(env.events)
    res = {"env_id": spec.environment_id, "success": bool(obs.success), "steps": obs.step,
           "cpu_time": cpu, "fail_events": nf, "recovered": nr,
           "n_locked": sum(1 for e in env.events if e[1] == "door_locked")}
    if hasattr(agent, "episode_stats"):
        st = agent.episode_stats()
        h = spec.hidden
        tested = st.pop("tested")
        truth = []
        for hid, _ in tested:
            if hid[0] == "KD":
                truth.append(h.key_door.get(hid[1]) == hid[2])
            elif hid[0] == "SB":
                truth.append(h.switch_bridge.get(hid[1]) == hid[2])
            else:
                truth.append(True)
        st["hyp_correct"], st["hyp_tested"] = sum(truth), len(truth)
        res.update(st)
        if collect_trace:
            res["trace"] = agent.trace
            res["truth_kd"] = dict(h.key_door)
    return res
