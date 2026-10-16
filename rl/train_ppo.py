"""PPO (JAX, CPU) with reward shaping for training only. --model ff|gru. GAE, clipped objective, Adam."""
import os, sys, time, json, pickle, argparse, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import numpy as np, jax, jax.numpy as jnp
from rl import net, encode
from regimes import train_config
from environment.generator import generate
from environment.world import World
from rl.shaping import potential

NE, T, EPOCHS, MB = 16, 64, 3, 2
GAMMA, LAM, CLIP, VF, ENT, LR, MAXG = 0.99, 0.95, 0.2, 0.5, 0.01, 8e-4, 0.5


class Env:
    def __init__(self, dist, rng, cap=lambda: 5):
        self.dist, self.rng, self.cap = dist, rng, cap
        self.new()

    def new(self):
        self.w = World(generate(train_config(self.dist, self.rng, self.cap())))
        self.obs = self.w.reset(); self.last = None; self.ep_ret = 0.0; self.phi = potential(self.obs)

    def step(self, a_idx):
        a = encode.ACT[a_idx]
        n0 = len(self.w.events)
        o = self.w.step(a)
        r = -0.002
        for t, kind, pos, extra in self.w.events[n0:]:
            if kind == "door_opened" or (kind == "switch_toggled" and extra): r += 0.5
            elif kind == "pushed": r += 0.1
        if o.feedback == "picked_key": r += 1.0   # first pick-up while empty-handed (swaps earn nothing)
        if o.success: r += 5.0
        elif o.feedback == "trap": r -= 1.0
        phi = potential(o)   # continuous at episode end (a 0 terminal potential would reward dying/timeouts)
        r += GAMMA * phi - self.phi
        self.phi = phi
        self.obs, self.last = o, a
        return r, o.done, o.success


def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return {"m": z, "v": z, "t": 0}


def adam_step(p, g, st):
    gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
    g = jax.tree_util.tree_map(lambda x: x * jnp.minimum(1.0, MAXG / (gn + 1e-6)), g)
    t = st["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: 0.9 * m + 0.1 * g, st["m"], g)
    v = jax.tree_util.tree_map(lambda v, g: 0.999 * v + 0.001 * g * g, st["v"], g)
    p = jax.tree_util.tree_map(lambda p, m, v: p - LR * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8), p, m, v)
    return p, {"m": m, "v": v, "t": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru"); ap.add_argument("--dist", default="main")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rec = a.model == "gru"
    out = a.out or f"data/models/{a.dist}_{a.model}_s{a.seed}.pkl"
    rng = np.random.default_rng([a.seed, zlib.crc32(a.dist.encode()) % 10_000])
    prog = {"steps": 0}
    cap = lambda: int(min(5, 1 + prog["steps"] // 250_000)) if a.dist == "main" else 5   # curriculum for the mixed distribution
    envs = [Env(a.dist, np.random.default_rng(rng.integers(1 << 30)), cap) for _ in range(NE)]
    params = net.init_params(jax.random.PRNGKey(a.seed), rec)
    opt = adam_init(params)
    key = jax.random.PRNGKey(1000 + a.seed)
    stepf = jax.jit(lambda p, c, ag, v, h: net.step(p, c, ag, v, h, rec))

    def loss_fn(p, c, ag, v, rs, h0, act, oldlp, adv, ret):
        lg, val = net.unroll(p, c, ag, v, rs, h0, rec)
        lp = jax.nn.log_softmax(lg)
        nlp = jnp.take_along_axis(lp, act[..., None], -1)[..., 0]
        ratio = jnp.exp(nlp - oldlp)
        pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv))
        ent = -jnp.mean(jnp.sum(jnp.exp(lp) * lp, -1))
        return pg + VF * jnp.mean((val - ret) ** 2) - ENT * ent
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    h = np.zeros((NE, net.H), np.float32)
    enc = [encode.encode(e.obs, None) for e in envs]
    done_prev = np.ones(NE, np.float32)
    recent, curve, steps, t0, it = [], [], 0, time.time(), 0
    while steps < a.steps:
        C, A_, V_, R_, ACT_, LP, VAL, DN, RS = ([] for _ in range(9))
        h0 = h.copy(); ent_acc = []
        for t in range(T):
            c = np.stack([x[0] for x in enc]); ag = np.stack([x[1] for x in enc]); vv = np.stack([x[2] for x in enc])
            key, k = jax.random.split(key)
            hm = h * (1 - done_prev[:, None]) if rec else h
            lg, val, hn = stepf(params, c, ag, vv, hm)
            act = np.asarray(jax.random.categorical(k, lg))
            lsm = np.asarray(jax.nn.log_softmax(lg)); lpa = lsm[np.arange(NE), act]; ent_acc.append(float(-(np.exp(lsm) * lsm).sum(-1).mean()))
            C.append(c); A_.append(ag); V_.append(vv); RS.append(done_prev.copy()); ACT_.append(act); LP.append(lpa); VAL.append(np.asarray(val))
            h = np.asarray(hn); prog["steps"] = steps
            rew = np.zeros(NE, np.float32); dn = np.zeros(NE, np.float32)
            for i, e in enumerate(envs):
                r, d, s = e.step(int(act[i]))
                rew[i], dn[i] = r, d
                if d:
                    recent.append(float(s)); e.new(); enc[i] = encode.encode(e.obs, None)
                else:
                    enc[i] = encode.encode(e.obs, e.last)
            R_.append(rew); DN.append(dn); done_prev = dn; steps += NE
        c = np.stack([x[0] for x in enc]); ag = np.stack([x[1] for x in enc]); vv = np.stack([x[2] for x in enc])
        _, lastv, _ = stepf(params, c, ag, vv, h * (1 - done_prev[:, None]) if rec else h)
        lastv = np.asarray(lastv)
        VAL = np.stack(VAL); R_ = np.stack(R_); DN = np.stack(DN)
        adv = np.zeros_like(R_); g = 0
        for t in reversed(range(T)):
            nv = lastv if t == T - 1 else VAL[t + 1]
            delta = R_[t] + GAMMA * nv * (1 - DN[t]) - VAL[t]
            g = delta + GAMMA * LAM * (1 - DN[t]) * g
            adv[t] = g
        ret = adv + VAL
        data = dict(c=np.stack(C), ag=np.stack(A_), v=np.stack(V_), rs=np.stack(RS), act=np.stack(ACT_), lp=np.stack(LP), adv=adv, ret=ret)
        for _ in range(EPOCHS):
            perm = rng.permutation(NE)
            for mb in np.array_split(perm, MB):
                ad = data["adv"][:, mb]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                _, g_ = grad_fn(params, data["c"][:, mb], data["ag"][:, mb], data["v"][:, mb], data["rs"][:, mb],
                                jnp.asarray(h0[mb]) * 0 + jnp.asarray(h0[mb]) * (1 - jnp.asarray(data["rs"][0, mb])[:, None]) if rec else jnp.zeros((len(mb), net.H)),
                                data["act"][:, mb], data["lp"][:, mb], ad, data["ret"][:, mb])
                params, opt = adam_step(params, g_, opt)
        it += 1
        if it % 20 == 0:
            sr = float(np.mean(recent[-200:])) if recent else 0.0
            curve.append(dict(steps=steps, recent_success=sr, time=time.time() - t0))
            print(f"{a.dist} {a.model} s{a.seed} steps {steps} recent_success {sr:.3f} ent {np.mean(ent_acc):.2f} mean_r {float(np.mean(R_)):.3f} sps {steps/(time.time()-t0):.0f}", flush=True)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            pickle.dump({"params": jax.tree_util.tree_map(np.asarray, params), "recurrent": rec, "steps": steps}, open(out, "wb"))
            json.dump(curve, open(out.replace(".pkl", "_curve.json"), "w"))
    pickle.dump({"params": jax.tree_util.tree_map(np.asarray, params), "recurrent": rec, "steps": steps}, open(out, "wb"))
    json.dump(curve, open(out.replace(".pkl", "_curve.json"), "w"))


if __name__ == "__main__":
    main()
