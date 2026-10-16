"""Pure-JAX policy/value networks: CNN trunk (+ optional GRU for memory, RL^2-style)."""
import jax, jax.numpy as jnp, numpy as np
from jax import lax
from .encode import G, N_CELL, N_VEC

H = 128


def _init(key, shape, scale=None):
    fan_in = int(np.prod(shape[:-1]))
    return jax.random.normal(key, shape) * (scale or np.sqrt(2.0 / fan_in))


def init_params(key, recurrent):
    ks = iter(jax.random.split(key, 16))
    p = {"emb": _init(next(ks), (N_CELL, 16), 0.3),
         "c1": _init(next(ks), (3, 3, 17, 16)), "b1": jnp.zeros(16),
         "c2": _init(next(ks), (3, 3, 16, 16)), "b2": jnp.zeros(16),
         "d1": _init(next(ks), (7 * 7 * 16, H)), "db1": jnp.zeros(H),
         "dv": _init(next(ks), (N_VEC, 64)), "dvb": jnp.zeros(64),
         "d2": _init(next(ks), (H + 64, H)), "db2": jnp.zeros(H),
         "pi": _init(next(ks), (H, 7), 0.01), "pib": jnp.zeros(7),
         "v": _init(next(ks), (H, 1), 1.0), "vb": jnp.zeros(1)}
    if recurrent:
        p.update(Wz=_init(next(ks), (2 * H, H)), bz=jnp.zeros(H), Wr=_init(next(ks), (2 * H, H)), br=jnp.zeros(H),
                 Wh=_init(next(ks), (2 * H, H)), bh=jnp.zeros(H))
    return p


def trunk(p, cells, agent, vec):
    x = jnp.concatenate([p["emb"][cells.astype(jnp.int32)], agent[..., None]], -1)       # B,G,G,17
    dn = ("NHWC", "HWIO", "NHWC")
    x = jax.nn.relu(lax.conv_general_dilated(x, p["c1"], (2, 2), "SAME", dimension_numbers=dn) + p["b1"])
    x = jax.nn.relu(lax.conv_general_dilated(x, p["c2"], (1, 1), "SAME", dimension_numbers=dn) + p["b2"])
    x = jax.nn.relu(x.reshape(x.shape[0], -1) @ p["d1"] + p["db1"])
    v = jax.nn.relu(vec @ p["dv"] + p["dvb"])
    return jax.nn.relu(jnp.concatenate([x, v], -1) @ p["d2"] + p["db2"])


def gru(p, x, h):
    xh = jnp.concatenate([x, h], -1)
    z = jax.nn.sigmoid(xh @ p["Wz"] + p["bz"])
    r = jax.nn.sigmoid(xh @ p["Wr"] + p["br"])
    hh = jnp.tanh(jnp.concatenate([x, r * h], -1) @ p["Wh"] + p["bh"])
    return (1 - z) * h + z * hh


def heads(p, x):
    return x @ p["pi"] + p["pib"], (x @ p["v"] + p["vb"])[..., 0]


def step(p, cells, agent, vec, h, recurrent):
    """One timestep for B parallel envs -> logits, value, new hidden."""
    x = trunk(p, cells, agent, vec)
    if recurrent:
        h = gru(p, x, h); x = h
    lg, v = heads(p, x)
    return lg, v, h


def unroll(p, cells, agent, vec, resets, h0, recurrent):
    """Sequences [T,B,...] with resets[t]=1 if a new episode starts at t -> logits [T,B,7], values [T,B]."""
    T, B = cells.shape[:2]
    x = trunk(p, cells.reshape((T * B,) + cells.shape[2:]), agent.reshape((T * B,) + agent.shape[2:]),
              vec.reshape(T * B, -1)).reshape(T, B, -1)
    if recurrent:
        def f(h, inp):
            xt, rt = inp
            h = gru(p, xt, h * (1 - rt[:, None]))
            return h, h
        _, x = lax.scan(f, h0, (x, resets))
    lg, v = heads(p, x)
    return lg, v
