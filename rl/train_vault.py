"""PPO (FF or GRU) on the combination vault. Trains on code sizes (Ls x ks); evaluation on larger codes tests extrapolation."""
import os, sys, time, json, pickle, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, jax, jax.numpy as jnp
from rl import vault_net as net
from environment.vault import Vault
from agents.vault_agents import vec, LMAX

NE, T, EPOCHS, MB = 32, 64, 4, 2
GAMMA, LAM, CLIP, VF, ENT, LR, MAXG = 0.99, 0.95, 0.2, 0.5, 0.01, 1e-3, 0.5
MAXSTEPS = 150


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


class E:
    def __init__(self, Ls, ks, rng):
        self.Ls, self.ks, self.rng = Ls, ks, rng; self.new()

    def new(self):
        L, k = int(self.rng.choice(self.Ls)), int(self.rng.choice(self.ks))
        self.env = Vault(L, min(k, L), MAXSTEPS, int(self.rng.integers(1 << 30))); self.o = self.env.reset()

    def step(self, a):
        o = self.env.step(a); self.o = o
        r = -0.003 + (1.0 if o["success"] else 0.0)
        d, s = o["done"], o["success"]
        if d: self.new()
        return r, d, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=30000); ap.add_argument("--out", required=True)
    ap.add_argument("--Ls", type=int, nargs="+", default=[4, 5]); ap.add_argument("--ks", type=int, nargs="+", default=[2, 3])
    a = ap.parse_args()
    rec = a.model == "gru"
    rng = np.random.default_rng(a.seed)
    envs = [E(a.Ls, a.ks, np.random.default_rng(rng.integers(1 << 30))) for _ in range(NE)]
    params = net.init(jax.random.PRNGKey(a.seed), rec); opt = adam_init(params); key = jax.random.PRNGKey(5 + a.seed)
    stepf = jax.jit(lambda p, x, h: net.step(p, x, h, rec))

    def loss_fn(p, x, mask, rs, h0, act, oldlp, adv, ret):
        lg, val = net.unroll(p, x, rs, h0, rec)
        lg = jnp.where(mask, lg, -1e9); lp = jax.nn.log_softmax(lg)
        nlp = jnp.take_along_axis(lp, act[..., None], -1)[..., 0]
        ratio = jnp.exp(nlp - oldlp)
        pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv))
        ent = -jnp.mean(jnp.sum(jnp.where(mask, jnp.exp(lp) * lp, 0.0), -1))
        return pg + VF * jnp.mean((val - ret) ** 2) - ENT * ent
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    mk = lambda: np.stack([np.arange(LMAX) < e.o["L"] for e in envs])
    h = np.zeros((NE, net.H), np.float32); dprev = np.ones(NE, np.float32)
    recent, curve, eps, t0, it = [], [], 0, time.time(), 0
    while eps < a.episodes:
        X, M, RS, AC, LP, VL, RW, DN = ([] for _ in range(8)); h0 = h.copy()
        for t in range(T):
            x = np.stack([vec(e.o) for e in envs]); m = mk()
            hm = h * (1 - dprev[:, None]) if rec else h
            lg, val, hn = stepf(params, x, hm); lg = np.where(m, np.asarray(lg), -1e9)
            key, k = jax.random.split(key); act = np.asarray(jax.random.categorical(k, jnp.asarray(lg)))
            lsm = np.asarray(jax.nn.log_softmax(jnp.asarray(lg)))
            X.append(x); M.append(m); RS.append(dprev.copy()); AC.append(act); LP.append(lsm[np.arange(NE), act]); VL.append(np.asarray(val))
            h = np.asarray(hn); rew = np.zeros(NE, np.float32); dn = np.zeros(NE, np.float32)
            for i, e in enumerate(envs):
                r, d, s = e.step(int(act[i])); rew[i], dn[i] = r, d
                if d: recent.append(float(s)); eps += 1
            RW.append(rew); DN.append(dn); dprev = dn
        x = np.stack([vec(e.o) for e in envs]); _, lastv, _ = stepf(params, x, h * (1 - dprev[:, None]) if rec else h); lastv = np.asarray(lastv)
        VL, RW, DN = np.stack(VL), np.stack(RW), np.stack(DN); adv = np.zeros_like(RW); g = 0
        for t in reversed(range(T)):
            nv = lastv if t == T - 1 else VL[t + 1]
            g = RW[t] + GAMMA * nv * (1 - DN[t]) - VL[t] + GAMMA * LAM * (1 - DN[t]) * g; adv[t] = g
        D = dict(x=np.stack(X), m=np.stack(M), rs=np.stack(RS), ac=np.stack(AC), lp=np.stack(LP), adv=adv, ret=adv + VL)
        for _ in range(EPOCHS):
            for mb in np.array_split(rng.permutation(NE), MB):
                ad = D["adv"][:, mb]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                h0m = jnp.asarray(h0[mb]) * (1 - jnp.asarray(D["rs"][0, mb])[:, None])
                _, gr = grad_fn(params, D["x"][:, mb], D["m"][:, mb], D["rs"][:, mb], h0m, D["ac"][:, mb], D["lp"][:, mb], ad, D["ret"][:, mb])
                params, opt = adam_step(params, gr, opt)
        it += 1
        if it % 25 == 0:
            curve.append(dict(steps=eps, recent_success=float(np.mean(recent[-500:])), time=time.time() - t0))
            print(f"vault {a.model} s{a.seed} episodes {eps} recent_success {curve[-1]['recent_success']:.3f} eps/s {eps/(time.time()-t0):.0f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pickle.dump({"params": jax.tree_util.tree_map(np.asarray, params), "recurrent": rec, "episodes": eps}, open(a.out, "wb"))
    json.dump(curve, open(a.out.replace(".pkl", "_curve.json"), "w"))


if __name__ == "__main__":
    main()
