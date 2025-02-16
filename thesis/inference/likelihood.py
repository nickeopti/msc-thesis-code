import os.path
from typing import Callable

import jax
import jax.dtypes
import jax.lax
import jax.numpy as jnp

import thesis.experiments
import thesis.experiments.constraints
import thesis.experiments.diffusion_processes
import thesis.experiments.experiments
import thesis.experiments.simulators
import thesis.lightning
import thesis.models.models
import thesis.models.networks
import thesis.models.objectives
import thesis.processes.process
from thesis.experiments.constraints import Constraints, LandmarksConstraints
from thesis.experiments.simulators import Simulator
from thesis.processes.process import Diffusion

jax.config.update("jax_enable_x64", True)


def scaled_dp(sigma: float, dp: Diffusion) -> Diffusion:
    return Diffusion(
        drift=dp.drift,
        diffusion=lambda t, y: dp.diffusion(t, y) * jnp.sqrt(sigma),
        inverse_diffusion=lambda t, y: jnp.linalg.inv(dp.diffusion(t, y) * jnp.sqrt(sigma)),
        diffusion_divergence=lambda t, y: dp.diffusion_divergence(t, y) * jnp.sqrt(sigma),
    )


def brownian_bridge_dp(sigma: float, dp: Diffusion, constraints: Constraints, t1: float) -> Diffusion:
    s_dp = scaled_dp(sigma, dp)

    return Diffusion(
        drift=lambda t, y: (constraints.terminal.reshape(y.shape, order='F') - y) / (t1 - t),
        diffusion=s_dp.diffusion,
        inverse_diffusion=s_dp.inverse_diffusion,
        diffusion_divergence=s_dp.diffusion_divergence,
    )


def reverse_time_conditioned_dp(sigma: float, dp: Diffusion, score: Callable[[Diffusion], Callable[[jax.Array, jax.Array], jax.Array]]) -> Diffusion:
    s_dp = scaled_dp(sigma, dp)

    return Diffusion(
        drift=thesis.experiments.experiments._f_bar(dp=dp, score=score(s_dp)),
        diffusion=s_dp.diffusion,
        inverse_diffusion=s_dp.inverse_diffusion,
        diffusion_divergence=s_dp.diffusion_divergence
    )


def analytical(dp: Diffusion, sigma: float, constraints: Constraints):
    return jnp.sum(
        jax.vmap(
            lambda terminal, initial: jax.scipy.stats.multivariate_normal.logpdf(terminal, initial, dp.diffusion(0, constraints.initial) @ dp.diffusion(0, constraints.initial).T * sigma).squeeze(),
            in_axes=(1, 1)
        )(constraints.terminal, constraints.initial)
    )


def stable_analytical(dp: Diffusion, sigma: float, constraints: Constraints):
    k, _ = constraints.shape

    def f(terminal, initial):
        logdet = jnp.log(jnp.diagonal(jax.lax.linalg.cholesky(sigma * dp.diffusion(0, constraints.initial) @ dp.diffusion(0, constraints.initial).T))).sum()
        slogdet = jnp.linalg.slogdet(sigma * dp.diffusion(0, constraints.initial) @ dp.diffusion(0, constraints.initial).T)
        logdet = slogdet.sign * slogdet.logabsdet
        z, *_ = jnp.linalg.lstsq(dp.diffusion(0, constraints.initial), terminal - initial)
        return -k / 2 * jnp.log(2 * jnp.pi) - logdet / 2 - 1 / 2 * z.T @ z / sigma

    return jnp.sum(
        jax.vmap(
            lambda terminal, initial: f(terminal, initial),
            in_axes=(1, 1)
        )(constraints.terminal, constraints.initial)
    )


def stable_analytical_offset(dp: Diffusion, sigma: float, constraints: Constraints):
    k, _ = constraints.shape

    def f(terminal, initial):
        z, *_ = jnp.linalg.lstsq(dp.diffusion(0, constraints.initial), terminal - initial)
        return -k / 2 * jnp.log(sigma) - 1 / 2 * z.T @ z / sigma

    return jnp.sum(
        jax.vmap(
            lambda terminal, initial: f(terminal, initial),
            in_axes=(1, 1)
        )(constraints.terminal, constraints.initial)
    )


def simulated(key: jax.dtypes.prng_key, t1: float, constraints: Constraints, dp: Diffusion, sigma: float, simulator: Simulator, M: int, N: int, likelihood: Callable = analytical):
    delta = t1 / N
    var = delta

    s_dp = scaled_dp(sigma, dp)

    def f(key):
        _, ys = simulator.simulate_sample_path(key, s_dp, constraints.initial, t0=0, t1=t1 - delta, n_steps=N)
        return likelihood(dp, sigma * var, LandmarksConstraints(ys[-1], constraints.terminal))

    return jax.scipy.special.logsumexp(jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)


def heng(key: jax.dtypes.prng_key, t0: float, t1: float, constraints: Constraints, dp: Diffusion, sigma: float, dp_bar: Diffusion, simulator: Simulator, M: int, N: int):
    delta = (t1 - t0) / N
    var = jnp.abs(delta)

    def f(key):
        ts, ys = simulator.simulate_sample_path(key, dp_bar, constraints.initial, t0=t0, t1=t1 - delta, n_steps=N)
        ts = jnp.hstack((t0, ts))
        # ys = ys.reshape((-1, *constraints.initial.shape), order='F')
        ys = jnp.vstack((constraints.initial[None], ys))

        def stable_prod(a, b, diffusion):
            z, *_ = jnp.linalg.lstsq(diffusion, b - a)
            return z.T @ z

        w = (
            jnp.prod(
                jax.vmap(
                    lambda t, y, y_next:
                        jnp.prod(
                            jax.vmap(
                                lambda a, b: jax.scipy.stats.multivariate_normal.pdf(b, a, sigma * dp.diffusion(t, y) @ dp.diffusion(t, y).T * var),
                                in_axes=(1, 1)
                            )(y, y_next)
                        ),
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
            /
            jnp.prod(
                jax.vmap(
                    lambda t, y, y_next:
                        jnp.prod(
                            jax.vmap(
                                lambda a, b, d: jax.scipy.stats.multivariate_normal.pdf(b, a + delta * d, dp_bar.diffusion(t, y) @ dp_bar.diffusion(t, y).T * var),
                                in_axes=(1, 1, 1)
                            )(y, y_next, dp_bar.drift(t, y))
                        ),
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
        )

        # w = (
        #     jnp.sum(
        #         jax.vmap(
        #             lambda t, y, y_next:
        #                 jnp.sum(
        #                     jax.vmap(
        #                         (lambda a, b: -(b - a).T @ jnp.linalg.inv(sigma * dp.diffusion(t, y) @ dp.diffusion(t, y).T * var) @ (b - a) / 2)
        #                         if not stable else
        #                         (lambda a, b: -stable_prod(a, b, jnp.sqrt(sigma) * dp.diffusion(t, y) * jnp.sqrt(var)) / 2),
        #                         in_axes=(1, 1)
        #                     )(y, y_next)
        #                 ),
        #             in_axes=(0, 0, 0)
        #         )(ts[:-1], ys[:-1], ys[1:])
        #     )
        #     -
        #     jnp.sum(
        #         jax.vmap(
        #             lambda t, y, y_next:
        #                 jnp.sum(
        #                     jax.vmap(
        #                         (lambda a, b, d: -(b - (a + delta * d)).T @ jnp.linalg.inv(dp_bar.diffusion(t, y) @ dp_bar.diffusion(t, y).T * var) @ (b - (a + delta * d)) / 2)
        #                         if not stable else
        #                         (lambda a, b, d: -stable_prod(a + delta * d, b, dp_bar.diffusion(t, y) * jnp.sqrt(var)) / 2),
        #                         in_axes=(1, 1, 1)
        #                     )(y, y_next, dp_bar.drift(t, y))
        #                 ),
        #             in_axes=(0, 0, 0)
        #         )(ts[:-1], ys[:-1], ys[1:])
        #     )
        # )

        # return w + likelihood(dp, sigma * var, LandmarksConstraints(ys[-1], constraints.terminal))

        # jnp.sum(
        #     jax.vmap(
        #         lambda terminal, initial: jax.scipy.stats.multivariate_normal.logpdf(terminal, initial, dp.diffusion(0, constraints.initial) @ dp.diffusion(0, constraints.initial).T * sigma).squeeze(),
        #         in_axes=(1, 1)
        #     )(constraints.terminal, constraints.initial)
        # )

        return w * jnp.prod(
            jax.vmap(
                lambda initial, terminal: jax.scipy.stats.multivariate_normal.pdf(terminal, initial, sigma * dp.diffusion(0, initial) @ dp.diffusion(0, initial).T * var),
                in_axes=(1, 1)
            )(ys[-1], constraints.terminal)
        )

    # return jax.scipy.special.logsumexp(jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)

    return jnp.log(jnp.sum(jax.vmap(f)(jax.random.split(key, M)))) - jnp.log(M)


def importance_sampled(key: jax.dtypes.prng_key, t0: float, t1: float, constraints: Constraints, dp: Diffusion, sigma: float, dp_bar: Diffusion, simulator: Simulator, M: int, N: int, stable: bool = False, likelihood: Callable = analytical):
    delta = (t1 - t0) / N
    var = jnp.abs(delta)

    def f(key):
        ts, ys = simulator.simulate_sample_path(key, dp_bar, constraints.initial, t0=t0, t1=t1 - delta, n_steps=N)
        ts = jnp.hstack((t0, ts))
        # ys = ys.reshape((-1, *constraints.initial.shape), order='F')
        ys = jnp.vstack((constraints.initial[None], ys))

        def stable_prod(a, b, diffusion):
            z, *_ = jnp.linalg.lstsq(diffusion, b - a)
            return z.T @ z

        w = (
            jnp.sum(
                jax.vmap(
                    lambda t, y, y_next:
                        jnp.sum(
                            jax.vmap(
                                (lambda a, b: -(b - a).T @ jnp.linalg.inv(sigma * dp.diffusion(t, y) @ dp.diffusion(t, y).T * var) @ (b - a) / 2)
                                if not stable else
                                (lambda a, b: -stable_prod(a, b, jnp.sqrt(sigma) * dp.diffusion(t, y) * jnp.sqrt(var)) / 2),
                                in_axes=(1, 1)
                            )(y, y_next)
                        ),
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
            -
            jnp.sum(
                jax.vmap(
                    lambda t, y, y_next:
                        jnp.sum(
                            jax.vmap(
                                (lambda a, b, d: -(b - (a + delta * d)).T @ jnp.linalg.inv(dp_bar.diffusion(t, y) @ dp_bar.diffusion(t, y).T * var) @ (b - (a + delta * d)) / 2)
                                if not stable else
                                (lambda a, b, d: -stable_prod(a + delta * d, b, dp_bar.diffusion(t, y) * jnp.sqrt(var)) / 2),
                                in_axes=(1, 1, 1)
                            )(y, y_next, dp_bar.drift(t, y))
                        ),
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
        )

        return w + likelihood(dp, sigma * var, LandmarksConstraints(ys[-1], constraints.terminal))

    return jax.scipy.special.logsumexp(jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)
