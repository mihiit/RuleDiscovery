"""Pointer-style policy over macro candidates (+ optional GRU over decision history), pure JAX."""
import jax, jax.numpy as jnp, numpy as np
from jax import lax
from .gen_feat import CAND_F, STATE_F, NMAX

H = 64


def init(key, rec):
    k = iter(jax.random.split(key, 12))
    g = lambda shape, sc=None: jax.random.normal(next(k), shape) * (sc or np.sqrt(2.0 / shape[0]))
    p = {"W1": g((CAND_F + STATE_F + H, 96)), "b1": jnp.zeros(96), "W2": g((96, 1), 0.05), "b2": jnp.zeros(1),
         "V1": g((STATE_F + H, 64)), "vb1": jnp.zeros(64), "V2": g((64, 1), 1.0), "vb2": jnp.zeros(1)}
    if rec:
        p.update(Wz=g((STATE_F + H, H)), bz=jnp.zeros(H), Wr=g((STATE_F + H, H)), br=jnp.zeros(H), Wh=g((STATE_F + H, H)), bh=jnp.zeros(H))
    return p


def gru(p, x, h):
    xh = jnp.concatenate([x, h], -1)
    z = jax.nn.sigmoid(xh @ p["Wz"] + p["bz"]); r = jax.nn.sigmoid(xh @ p["Wr"] + p["br"])
    hh = jnp.tanh(jnp.concatenate([x, r * h], -1) @ p["Wh"] + p["bh"])
    return (1 - z) * h + z * hh


def heads(p, cf, st, mask, z):
    B = cf.shape[:-2]
    inp = jnp.concatenate([cf, jnp.broadcast_to(st[..., None, :], B + (NMAX, STATE_F)), jnp.broadcast_to(z[..., None, :], B + (NMAX, H))], -1)
    lg = (jax.nn.relu(inp @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"])[..., 0]
    lg = jnp.where(mask, lg, -1e9)
    v = (jax.nn.relu(jnp.concatenate([st, z], -1) @ p["V1"] + p["vb1"]) @ p["V2"] + p["vb2"])[..., 0]
    return lg, v


def step(p, cf, st, mask, h, rec):
    if rec:
        h = gru(p, st, h); z = h
    else:
        z = jnp.zeros_like(h)
    lg, v = heads(p, cf, st, mask, z)
    return lg, v, h


def unroll(p, cf, st, mask, resets, h0, rec):
    if rec:
        def f(h, inp):
            x, r = inp
            h = gru(p, x, h * (1 - r[:, None]))
            return h, h
        _, z = lax.scan(f, h0, (st, resets))
    else:
        z = jnp.zeros(st.shape[:-1] + (H,))
    return heads(p, cf, st, mask, z)
