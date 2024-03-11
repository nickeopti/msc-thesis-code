from functools import partial
import math

import jax
from diffrax import (ControlTerm, Euler, MultiTerm, ODETerm, SaveAt,
                     VirtualBrownianTree, diffeqsolve)

import process


@partial(jax.jit, static_argnames=['t0', 't1', 'dt'])
def get_data(
    dp: process.Diffusion,
    y0: jax.Array,
    key,
    t0: float = 0,
    t1: float = 1,
    dt: float = 0.01,
):
    d = dp.diffusion.shape[0]
    brownian_motion = VirtualBrownianTree(min(t0, t1), max(t0, t1), tol=1e-3, shape=(d,), key=key)

    drift_term = ODETerm(lambda t, y, _: dp.drift)
    diffusion_term = ControlTerm(lambda t, y, _: dp.diffusion, brownian_motion)
    terms = MultiTerm(drift_term, diffusion_term)

    solver = Euler()
    saveat = SaveAt(steps=True)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat, max_steps=math.floor(abs((t1 - t0) / dt)) + 1)

    n = sol.stats['num_steps']

    return sol.ts, sol.ys, n


def get_paths(
    dp: process.Diffusion,
    y0: jax.Array,
    key,
    t0: float = 0,
    t1: float = 1,
    dt: float = 0.01,
):
    d = dp.diffusion.shape[0]
    brownian_motion = VirtualBrownianTree(min(t0, t1), max(t0, t1), tol=1e-3, shape=(d,), key=key)

    drift_term = ODETerm(lambda t, y, _: dp.drift(t, y))
    diffusion_term = ControlTerm(lambda t, y, _: dp.diffusion, brownian_motion)
    terms = MultiTerm(drift_term, diffusion_term)

    solver = Euler()
    saveat = SaveAt(steps=True)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat, max_steps=math.floor(abs((t1 - t0) / dt)) + 1)

    n = sol.stats['num_steps']

    return sol.ts, sol.ys, n
