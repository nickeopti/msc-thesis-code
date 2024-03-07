import jax
from diffrax import (ControlTerm, Euler, MultiTerm, ODETerm, SaveAt,
                     VirtualBrownianTree, diffeqsolve)

import process


@jax.jit
def get_data(
    dp: process.Diffusion,
    y0: jax.Array,
    key,
    t0: float = 0,
    t1: float = 1,
    dt: float = 0.01,
):
    brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(2,), key=key)

    drift_term = ODETerm(lambda t, y, _: dp.drift(t, y))
    diffusion_term = ControlTerm(lambda t, y, _: dp.diffusion(t, y), brownian_motion)
    terms = MultiTerm(drift_term, diffusion_term)

    solver = Euler()
    saveat = SaveAt(steps=True)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat, max_steps=int(1 / dt))

    return sol.ts, sol.ys
