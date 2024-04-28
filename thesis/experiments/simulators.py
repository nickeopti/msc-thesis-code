import jax
import jax.numpy as jnp

import thesis.processes.diffusion as diffusion
import thesis.processes.process as process


class Simulator:
    def __init__(self, dp: process.Diffusion, displacement: bool = False) -> None:
        self.dp = dp
        self.displacement = displacement

    @staticmethod
    def simulate_sample_path(key: jax.dtypes.prng_key, initial: jax.Array):
        raise NotImplementedError


class LongSimulator(Simulator):
    def __init__(self, dp: process.Diffusion, displacement: bool = False) -> None:
        super().__init__(dp, displacement)

        # TODO: Consider constructing a new diffusion
        # process, suitable for the long format, by
        # replicating the individual parts of the dp
        # here, instead of having all the dps doing that.
        # That could probably make the code even simpler.

        @jax.jit
        def simulate(key: jax.dtypes.prng_key, initial: jax.Array):
            if self.displacement:
                const = initial.reshape(-1, order='F')
                dp = process.Diffusion(
                    d=self.dp.d,
                    drift=lambda t, y: self.dp.drift(t, y + const),
                    diffusion=lambda t, y: self.dp.diffusion(t, y + const),
                    inverse_diffusion=lambda t, y: self.dp.inverse_diffusion(t, y + const),
                    diffusion_divergence=lambda t, y: self.dp.diffusion_divergence(t, y + const),
                )
                return diffusion.get_data(
                    dp=dp,
                    y0=jnp.zeros_like(initial, shape=(initial.size,)),
                    key=key,
                )
            else:
                return diffusion.get_data(
                    dp=self.dp,
                    y0=initial.reshape(-1, order='F'),
                    key=key,
                )

        self.simulate_sample_path = simulate


class AutoLongSimulator(Simulator):
    def __init__(self, dp: process.Diffusion, dim: int, displacement: bool = False) -> None:
        super().__init__(dp, displacement)

        def make_wide(y: jax.Array) -> jax.Array:
            return y.reshape((-1, dim), order='F')
        
        def make_long(y: jax.Array) -> jax.Array:
            return y.reshape(-1, order='F')

        def long_diffusion(a):
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

        long_dp = process.Diffusion(
            d=dp.d * dim,
            drift=lambda t, y: jnp.tile(dp.drift(t, make_wide(y)), dim),
            diffusion=lambda t, y: long_diffusion(dp.diffusion(t, make_wide(y))),
            inverse_diffusion=lambda t, y: jnp.linalg.inv(long_diffusion(dp.diffusion(t, make_wide(y)))),
            diffusion_divergence=lambda t, y: jnp.tile(dp.diffusion_divergence(t, make_long(y)), dim),
        )

        # @jax.jit
        def simulate(key: jax.dtypes.prng_key, initial: jax.Array):
            print(f'{long_dp.d=}')
            print(f'{long_dp.drift(0, jnp.zeros(long_dp.d)).shape=}')
            if self.displacement:
                return diffusion.get_data(
                    dp=long_dp,
                    y0=jnp.zeros_like(initial, shape=(initial.size,)),
                    key=key,
                )
            else:
                return diffusion.get_data(
                    dp=long_dp,
                    y0=make_long(initial),
                    key=key,
                )

        self.simulate_sample_path = simulate


class WideSimulator(Simulator):
    def __init__(self, dp: process.Diffusion, displacement: bool = False) -> None:
        super().__init__(dp, displacement)

        @jax.jit
        def simulate(key: jax.dtypes.prng_key, initial: jax.Array):
            _, d = initial.shape
            keys = jax.random.split(key, d)

            ts, ys, ns = jax.vmap(
                lambda key, y0: (
                    diffusion.get_data(
                        dp=self.dp,
                        y0=jnp.zeros_like(y0) if self.displacement else y0,
                        key=key,
                    )
                ),
                in_axes=(0, 1),
                out_axes=-1,
            )(keys, initial)

            # for na, nb in zip(ns, ns[1:]):
            #     assert na == nb
            # n = na
            n = ns[0]

            for ta, tb in zip(ts[:n].T, ts[:n].T[1:]):
                assert jnp.all(ta == tb)

            # for p in r:
            #     print(f'{p.shape=}')

            return ts[:, 0], ys, n

        self.simulate_sample_path = simulate
