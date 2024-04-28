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

    @staticmethod
    def f_bar_learned(t, y, dp: process.Diffusion, state: train_state.TrainState, c: float):
        s = state.apply_fn(state.params, t[None], y.reshape(-1, order='F')[None], c)[0]
        # s = state.apply_fn(state.params, t[None], y[None], c)[0, :, 0]  # For the wide version
        # s0 = state.apply_fn(state.params, t[None], jnp.ones_like(y.reshape(-1, order='F')[None]), c)[0]
        # k = s0.size // 2
        # jax.debug.print('{v}', v=jnp.max(jnp.abs(s0[:k] - s0[k:])))
        return dp.drift(t, y) - dp.diffusion(t, y) @ s - dp.diffusion_divergence(t, y)


class Brownian(DiffusionProcess):
    @staticmethod
    def score_analytical(t, y, dp: process.Diffusion, constraints: Constraints):
        if hasattr(constraints, 'score_analytical'):
            return constraints.score_analytical(t, y, dp, constraints)

        y0 = constraints.initial.reshape(-1, order='F')
        return -dp.inverse_diffusion(t, y) @ (y - y0) / t

    @staticmethod
    def f_bar_analytical(t, y, dp: process.Diffusion, constraints: Constraints):
        y0 = constraints.initial.reshape(-1, order='F')
        s = -dp.inverse_diffusion(t, y) @ (y - y0) / t
        return dp.drift(t, y) - dp.diffusion(t, y) @ s - dp.diffusion_divergence(t, y)


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


class BrownianWideKernel(Brownian):
    def __init__(self, variance: float, gamma: float, constraints: Constraints) -> None:
        f = partial(kernel, variance=variance, gamma=gamma)
        k = pairwise(f, constraints.initial)

        super().__init__(process.brownian_motion(k), variance)


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


class KunitaLongKernel(DiffusionProcess):
    def __init__(self, k: int, d: int, variance: float, gamma: float) -> None:
        dp = process.q_process(k, d, variance, gamma)
        # dp = process.kunita_flow(k, variance, gamma)

        super().__init__(dp, variance)
