import jax, jax.numpy as jnp, numpy as np
from jax import lax
from agents.vault_agents import VEC, LMAX
H = 64
_JIT = {}


def init(key, rec):
    k = iter(jax.random.split(key, 8))
    g = lambda shape, sc=None: jax.random.normal(next(k), shape) * (sc or np.sqrt(2.0 / shape[0]))
    p = {"W1": g((VEC, H)), "b1": jnp.zeros(H), "Wp": g((H, LMAX), 0.05), "bp": jnp.zeros(LMAX), "Wv": g((H, 1), 1.0), "bv": jnp.zeros(1)}
    if rec:
        p.update(Wz=g((H + H, H)), bz=jnp.zeros(H), Wr=g((H + H, H)), br=jnp.zeros(H), Wh=g((H + H, H)), bh=jnp.zeros(H))
    return p


def gru(p, x, h):
    xh = jnp.concatenate([x, h], -1)
    z = jax.nn.sigmoid(xh @ p["Wz"] + p["bz"]); r = jax.nn.sigmoid(xh @ p["Wr"] + p["br"])
    return (1 - z) * h + z * jnp.tanh(jnp.concatenate([x, r * h], -1) @ p["Wh"] + p["bh"])


def step(p, x, h, rec):
    e = jax.nn.relu(x @ p["W1"] + p["b1"])
    if rec:
        h = gru(p, e, h); e = h
    return e @ p["Wp"] + p["bp"], (e @ p["Wv"] + p["bv"])[..., 0], h


def unroll(p, x, resets, h0, rec):
    e = jax.nn.relu(x @ p["W1"] + p["b1"])
    if rec:
        def f(h, inp):
            xt, r = inp
            h = gru(p, xt, h * (1 - r[:, None])); return h, h
        _, e = lax.scan(f, h0, (e, resets))
    return e @ p["Wp"] + p["bp"], (e @ p["Wv"] + p["bv"])[..., 0]
