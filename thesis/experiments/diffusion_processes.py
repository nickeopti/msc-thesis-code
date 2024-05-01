from functools import partial

import jax
import jax.numpy as jnp
from flax.training import train_state

import thesis.processes.process as process
from thesis.experiments.constraints import Constraints


def kernel(x, y, variance, gamma):
    return variance * jnp.exp(-jnp.linalg.norm(x - y)**2 / gamma / 2)

def pairwise(f, xs):
    return jax.vmap(lambda x: jax.vmap(lambda y: f(x, y))(xs))(xs)


class DiffusionProcess:
    def __init__(self, dp: process.Diffusion, c: float) -> None:
        self.dp = dp
        self.c = c

    @staticmethod
    def score_learned(t, y, state: train_state.TrainState, c: float):
        return state.apply_fn(state.params, t, y, c=c)


class Brownian(DiffusionProcess):
    @staticmethod
    def score_analytical(t, y, dp: process.Diffusion, constraints: Constraints):
        if hasattr(constraints, 'score_analytical'):
            return constraints.score_analytical(t, y, dp, constraints)

        def g(y, y0):
            return -dp.inverse_diffusion(t, y) @ (y - y0) / t

        if len(y.shape) == 1:
            y0 = constraints.initial.reshape(-1, order='F')
            return g(y, y0)
        else:
            return jax.vmap(g, in_axes=(1, 1), out_axes=1)(y, constraints.initial)


class Brownian1D(Brownian):
    def __init__(self, variance: float) -> None:
        super().__init__(
            process.brownian_motion(jnp.array([[variance]])),
            variance,
        )


class Brownian2D(Brownian):
    def __init__(self, variance: float, covariance: float) -> None:
        super().__init__(
            process.brownian_motion(jnp.array([[variance, covariance], [covariance, variance]])),
            covariance,
        )


class BrownianND(Brownian):
    def __init__(self, d: int, variance: float) -> None:
        super().__init__(
            process.brownian_motion(variance * jnp.identity(d)),
            variance,
        )


class BrownianNDWide(Brownian):
    def __init__(self, d: int, variance: float) -> None:
        super().__init__(
            process.wide_brownian_motion(variance * jnp.identity(d // 2), 2),
            variance,
        )


class BrownianWideKernel(Brownian):
    def __init__(self, variance: float, gamma: float, constraints: Constraints) -> None:
        f = partial(kernel, variance=variance, gamma=gamma)
        k = pairwise(f, constraints.initial)

        super().__init__(process.wide_brownian_motion(k, constraints.initial.shape[1]), variance)


class BrownianLongKernel(Brownian):
    def __init__(self, variance: float, gamma: float, constraints: Constraints) -> None:
        _, d = constraints.initial.shape

        f = partial(kernel, variance=variance, gamma=gamma)
        a = pairwise(f, constraints.initial)

        k = jnp.vstack(
            [
                jnp.hstack(
                    [
                        a if i == j else jnp.zeros_like(a)
                        for j in range(d)
                    ]
                )
                for i in range(d)
            ]
        )

        super().__init__(process.brownian_motion(k), variance)


class Kunita(DiffusionProcess):
    def __init__(self, variance: float, gamma: float, constraints: Constraints) -> None:
        k, d = constraints.initial.shape
        dp = process.kunita_flow(k, d, variance, gamma)

        super().__init__(dp, variance)
