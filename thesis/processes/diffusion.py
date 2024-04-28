import math
from typing import Type

import equinox as eqx
import jax
from diffrax import (ControlTerm, Euler, MultiTerm, ODETerm, SaveAt,
                     VirtualBrownianTree, diffeqsolve)

import thesis.processes.process as process


class ReverseVirtualBrownianTree(VirtualBrownianTree):
    @eqx.filter_jit
    def evaluate(
        self,
        t0,
        t1=None,
        left: bool = True,
        use_levy: bool = False,
    ):
        return super().evaluate(1 - t0, 1 - t1, left, use_levy)


def get_data(
    dp: process.Diffusion,
    y0: jax.Array,
    key,
    t0: float = 0,
    t1: float = 1,
    dt: float = 0.01,
):
    d = dp.d
    brownian_motion = VirtualBrownianTree(min(t0, t1), max(t0, t1), tol=1e-3, shape=(d,), key=key)

    drift_term = ODETerm(lambda t, y, _: dp.drift(t, y))
    diffusion_term = ControlTerm(lambda t, y, _: dp.diffusion(t, y), brownian_motion)
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
    max_steps: int = 2000,
    brownian_tree_class: Type[VirtualBrownianTree] = VirtualBrownianTree,
):
    d = dp.d
    brownian_motion = brownian_tree_class(min(t0, t1), max(t0, t1), tol=1e-3, shape=(d,), key=key)

    drift_term = ODETerm(lambda t, y, _: dp.drift(t, y))
    diffusion_term = ControlTerm(lambda t, y, _: dp.diffusion(t, y), brownian_motion)
    terms = MultiTerm(drift_term, diffusion_term)

    solver = Euler()
    saveat = SaveAt(steps=True)
    # sol = diffeqsolve(terms, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat, max_steps=math.floor(abs((t1 - t0) / dt)) + 1)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=dt, y0=y0.reshape(-1, order='F'), saveat=saveat, max_steps=max_steps)

    n = sol.stats['num_steps']

    return sol.ts, sol.ys, n
