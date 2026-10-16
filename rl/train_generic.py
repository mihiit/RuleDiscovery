"""PPO over macro-actions (feedforward or GRU). Usage: train_macro.py --model ff|gru --dist main|<regime> --seed 0 --episodes N"""
import os, sys, time, json, pickle, argparse, zlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np, jax, jax.numpy as jnp
from rl import gen_net as net, gen_feat as mf
from regimes import train_config
from environment.generator import generate
from environment.world import World
from agents.generic_engine import GenericEngine as MacroEngine

NE, T, EPOCHS, MB = 16, 32, 4, 2
GAMMA, LAM, CLIP, VF, ENT, LR, MAXG = 0.99, 0.95, 0.2, 0.5, 0.01, 1e-3, 0.5


class Env:
    def __init__(self, dist, rng):
        self.dist, self.rng, self.eng = dist, rng, MacroEngine(0)
        self.new()

    def new(self):
        self.w = World(generate(train_config(self.dist, self.rng)))
        self.obs = self.w.reset(); self.eng.reset(); self.last = None
        self.cands = self.eng.candidates(self.obs)
        if not self.cands:
            self.cands = []

    def featurize(self):
        return mf.featurize(self.obs, self.cands, self.last)

    def step(self, idx):
        e, w = self.eng, self.w
        before, steps0, n0 = self.obs, self.obs.step, len(w.events)
        if not self.cands:                      # no macro available: wait one step
            self.obs = w.step("WAIT"); macro, kind = None, None
        else:
            macro = e.commit(self.obs, idx)
            kind = None
            while e.plan and not self.obs.done:
                a, kind, meta = e.plan.pop(0)
                e._cur = (a, kind, meta, self.obs)
                nxt = w.step(a); e.observe(self.obs, a, nxt); self.obs = nxt
        gate = sum(1 for t, k_, p_, x_ in w.events[n0:] if k_ in ("door_opened", "pushed") or (k_ == "switch_toggled" and x_))
        r = -0.002 * (self.obs.step - steps0) + 0.2 * gate + (1.0 if self.obs.success else 0.0)
        done, succ = self.obs.done, self.obs.success
        if done:
            self.new()
        else:
            self.last = macro
            self.cands = e.candidates(self.obs)
        return r, done, succ


def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p); return {"m": z, "v": z, "t": 0}


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
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--episodes", type=int, default=60000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rec = a.model == "gru"
    out = a.out or f"data/models/generic_{a.dist}_{a.model}_s{a.seed}.pkl"
    rng = np.random.default_rng([a.seed, zlib.crc32(a.dist.encode()) % 10_000])
    envs = [Env(a.dist, np.random.default_rng(rng.integers(1 << 30))) for _ in range(NE)]
    params = net.init(jax.random.PRNGKey(a.seed), rec); opt = adam_init(params); key = jax.random.PRNGKey(7 + a.seed)
    stepf = jax.jit(lambda p, c, s, m, h: net.step(p, c, s, m, h, rec))

    def loss_fn(p, cf, st, mask, rs, h0, act, oldlp, adv, ret):
        lg, val = net.unroll(p, cf, st, mask, rs, h0, rec)
        lp = jax.nn.log_softmax(lg)
        nlp = jnp.take_along_axis(lp, act[..., None], -1)[..., 0]
        ratio = jnp.exp(nlp - oldlp)
        pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv))
        pr = jnp.exp(lp) * mask
        ent = -jnp.mean(jnp.sum(jnp.where(mask, jnp.exp(lp) * lp, 0.0), -1))
        return pg + VF * jnp.mean((val - ret) ** 2) - ENT * ent
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    h = np.zeros((NE, net.H), np.float32); done_prev = np.ones(NE, np.float32)
    feats = [e.featurize() for e in envs]
    recent, curve, eps, t0, it = [], [], 0, time.time(), 0
    while eps < a.episodes:
        CF, ST, MK, RS, AC, LP, VL, RW, DN = ([] for _ in range(9)); h0 = h.copy()
        for t in range(T):
            cf = np.stack([f[0] for f in feats]); st = np.stack([f[1] for f in feats]); mk = np.stack([f[2] for f in feats])
            mk = np.where(mk.any(1, keepdims=True), mk, np.eye(net.NMAX, dtype=bool)[0][None])    # dummy slot if no macro
            hm = h * (1 - done_prev[:, None]) if rec else h
            lg, val, hn = stepf(params, cf, st, mk, hm)
            key, k = jax.random.split(key)
            act = np.asarray(jax.random.categorical(k, lg)); lsm = np.asarray(jax.nn.log_softmax(lg))
            CF.append(cf); ST.append(st); MK.append(mk); RS.append(done_prev.copy()); AC.append(act); LP.append(lsm[np.arange(NE), act]); VL.append(np.asarray(val))
            h = np.asarray(hn); rew = np.zeros(NE, np.float32); dn = np.zeros(NE, np.float32)
            for i, e in enumerate(envs):
                r, d, s = e.step(int(act[i])); rew[i], dn[i] = r, d
                if d: recent.append(float(s)); eps += 1
                feats[i] = e.featurize()
            RW.append(rew); DN.append(dn); done_prev = dn
        cf = np.stack([f[0] for f in feats]); st = np.stack([f[1] for f in feats]); mk = np.stack([f[2] for f in feats])
        mk = np.where(mk.any(1, keepdims=True), mk, np.eye(net.NMAX, dtype=bool)[0][None])
        _, lastv, _ = stepf(params, cf, st, mk, h * (1 - done_prev[:, None]) if rec else h); lastv = np.asarray(lastv)
        VL, RW, DN = np.stack(VL), np.stack(RW), np.stack(DN)
        adv = np.zeros_like(RW); g = 0
        for t in reversed(range(T)):
            nv = lastv if t == T - 1 else VL[t + 1]
            g = RW[t] + GAMMA * nv * (1 - DN[t]) - VL[t] + GAMMA * LAM * (1 - DN[t]) * g; adv[t] = g
        ret = adv + VL
        D = dict(cf=np.stack(CF), st=np.stack(ST), mk=np.stack(MK), rs=np.stack(RS), ac=np.stack(AC), lp=np.stack(LP), adv=adv, ret=ret)
        for _ in range(EPOCHS):
            for mb in np.array_split(rng.permutation(NE), MB):
                ad = D["adv"][:, mb]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                h0m = jnp.asarray(h0[mb]) * (1 - jnp.asarray(D["rs"][0, mb])[:, None])
                _, gr = grad_fn(params, D["cf"][:, mb], D["st"][:, mb], D["mk"][:, mb], D["rs"][:, mb], h0m, D["ac"][:, mb], D["lp"][:, mb], ad, D["ret"][:, mb])
                params, opt = adam_step(params, gr, opt)
        it += 1
        if it % 25 == 0:
            sr = float(np.mean(recent[-300:])) if recent else 0.0
            curve.append(dict(steps=eps, recent_success=sr, time=time.time() - t0))
            print(f"{a.dist} macro-{a.model} s{a.seed} episodes {eps} recent_success {sr:.3f} eps/s {eps/(time.time()-t0):.1f}", flush=True)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            pickle.dump({"params": jax.tree_util.tree_map(np.asarray, params), "recurrent": rec, "episodes": eps}, open(out, "wb"))
            json.dump(curve, open(out.replace(".pkl", "_curve.json"), "w"))
    pickle.dump({"params": jax.tree_util.tree_map(np.asarray, params), "recurrent": rec, "episodes": eps}, open(out, "wb"))
    json.dump(curve, open(out.replace(".pkl", "_curve.json"), "w"))


if __name__ == "__main__":
    main()
