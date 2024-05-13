import jax
import jax.lax
import jax.numpy as jnp
import jax.random

from thesis.processes import process


def get_data(
    dp: process.Diffusion,
    y0: jax.Array,
    key,
    t0: float = 0,
    t1: float = 1,
    n_steps: int = 100,
    diffusion_scale: float = 1,
):
    d = y0.shape[0]
    ts = jnp.linspace(t0, t1, n_steps + 1, endpoint=True)
    wiener_increments = jax.random.normal(key, shape=(n_steps, d))

    def step(y, p):
        t = p[0]
        dt = p[1]
        w = p[2:]
        y_next = y + dp.drift(t, y) * dt + diffusion_scale * dp.diffusion(t, y) @ (jnp.sqrt(jnp.abs(dt)) * w)
        return y_next, y_next

    _, ys = jax.lax.scan(
        f=step,
        init=y0,
        xs=jnp.hstack((ts[:-1][:, None], (ts[1:] - ts[:-1])[:, None], wiener_increments))
    )
    return ts[1:], ys
