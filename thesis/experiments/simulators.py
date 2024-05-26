from functools import partial, wraps

import jax
import jax.numpy as jnp

import thesis.processes.diffusion as diffusion
import thesis.processes.process as process


class Simulator:
    @staticmethod
    def simulate_sample_path(key: jax.dtypes.prng_key, dp: process.Diffusion, initial: jax.Array, **kwargs) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError


class LongSimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()

        @partial(jax.jit, static_argnames=('dp', 'n_steps'))
        def simulate(key: jax.dtypes.prng_key, dp: process.Diffusion, initial: jax.Array, t0, t1, n_steps, diffusion_scale: float = 1):
            return diffusion.get_data(
                dp=dp,
                y0=initial.reshape(-1, order='F'),
                key=key,
                t0=t0,
                t1=t1,
                n_steps=n_steps,
                diffusion_scale=diffusion_scale,
            )

        self.simulate_sample_path = simulate


class AutoLongSimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()

        def make_wide(y: jax.Array, dim) -> jax.Array:
            return y.reshape((-1, dim), order='F')
        
        def make_long(y: jax.Array) -> jax.Array:
            return y.reshape(-1, order='F')

        def long_diffusion(a, dim):
            return jnp.vstack(
                [
                    jnp.hstack(
                        [
                            a if i == j else jnp.zeros_like(a)
                            for j in range(dim)
                        ]
                    )
                    for i in range(dim)
                ]
            )

        def long_dp(dp: process.Diffusion, dim: int):
            return process.Diffusion(
                drift=lambda t, y: make_long(dp.drift(t, make_wide(y, dim))),
                diffusion=lambda t, y: long_diffusion(dp.diffusion(t, make_wide(y, dim)), dim),
                inverse_diffusion=lambda t, y: jnp.linalg.inv(long_diffusion(dp.diffusion(t, make_wide(y)), dim)),
                diffusion_divergence=lambda t, y: make_long(dp.diffusion_divergence(t, make_wide(y, dim))),
            )

        @partial(jax.jit, static_argnames=('dp', 'n_steps'))
        def simulate(key: jax.dtypes.prng_key, dp: process.Diffusion, initial: jax.Array, t0, t1, n_steps, diffusion_scale: float = 1):
            ts, ys = diffusion.get_data(
                dp=long_dp(dp, initial.shape[1]),
                y0=make_long(initial),
                key=key,
                t0=t0,
                t1=t1,
                n_steps=n_steps,
                diffusion_scale=diffusion_scale,
            )
            return ts, jax.vmap(lambda y: make_wide(y, initial.shape[1]))(ys)

        self.simulate_sample_path = simulate
