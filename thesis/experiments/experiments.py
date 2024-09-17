import pathlib
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax.training import train_state

import thesis.processes.process as process
from thesis.experiments.constraints import Constraints, PointMixtureConstraints
from thesis.experiments.diffusion_processes import DiffusionProcess
from thesis.experiments.simulators import Simulator


def _f_bar(dp: process.Diffusion, score: Callable[[jax.Array, jax.Array], jax.Array]):
    def f(t, y):
        def g(d, s, div):
            return d - dp.diffusion(t, y) @ s - div

        if len(y.shape) == 1:
            return g(dp.drift(t, y), score(t, y), dp.diffusion_divergence(t, y))
        else:
            return jax.vmap(g, in_axes=(1, 1, 1), out_axes=1)(dp.drift(t, y), score(t, y), dp.diffusion_divergence(t, y))

    return f


class Experiment:
    def __init__(
        self,
        key: jax.dtypes.prng_key,
        constraints: Constraints,
        diffusion_process: DiffusionProcess,
        simulator: Simulator,
        displacement: bool,
        n: int,
        min_diffusion_scale: Optional[float] = None,
        max_diffusion_scale: Optional[float] = None,
    ) -> None:
        self.key = key

        self.constraints = constraints
        self.diffusion_process = diffusion_process
        self.simulator = simulator

        self.displacement = displacement
        self.n = n

        if min_diffusion_scale is not None and max_diffusion_scale is not None:
            self.diffusion_scale_range = (min_diffusion_scale, max_diffusion_scale)
            assert self.diffusion_process.c == 1, 'Variance shall be set to 1'
        elif min_diffusion_scale is not None or max_diffusion_scale is not None:
            raise ValueError('Either set both or none of {min|max}_diffusion_scale')
        else:
            self.diffusion_scale_range = None

    @property
    def dp(self) -> process.Diffusion:
        return self.diffusion_process.dp

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, key: jax.dtypes.prng_key) -> jax.Array:
        if isinstance(self.constraints, PointMixtureConstraints):
            initial = jnp.stack((self.constraints.initial_a, self.constraints.initial_b))[jax.random.bernoulli(key).astype(int)]
        else:
            initial = self.constraints.initial

        subkey1, subkey2 = jax.random.split(key)

        if self.diffusion_scale_range is not None:
            c = jax.random.uniform(subkey2, minval=self.diffusion_scale_range[0], maxval=self.diffusion_scale_range[1])
            diffusion_scale = 10**c
        else:
            c = self.diffusion_process.c
            diffusion_scale = 1

        ts, ys = self.simulator.simulate_sample_path(subkey1, self.dp, initial, t0=0, t1=1, n_steps=1000, diffusion_scale=diffusion_scale)
        if self.displacement:
            ys -= initial.reshape(ys[0].shape, order='F')

        return ts, ys, initial, c, (initial.reshape(ys[0].shape, order='F') if self.displacement else 0)

    def visualise(self, state: train_state.TrainState, plots_path: pathlib.Path):
        self.key, key = jax.random.split(self.key)

        cs = ([self.diffusion_process.c] if self.diffusion_scale_range is None else jnp.linspace(*self.diffusion_scale_range, 10))

        if self.constraints.visualise_paths is not None:
            fs = [
                (
                    _f_bar(
                        self.dp,
                        lambda t, y:
                            self.diffusion_process.score_learned(
                                t[None],
                                (y - (self.constraints.initial.reshape(y.shape, order='F') if self.displacement else 0))[None],
                                state=state,
                                c=jnp.array([c])
                            )[0]
                    ),
                    10**c,
                    ('learned' if self.diffusion_scale_range is None else f'learned_c_{10**c:.3}')
                )
                for c in cs
            ]
            if hasattr(self.diffusion_process, 'score_analytical'):
                fs.extend(
                    [
                        (
                            _f_bar(
                                self.dp,
                                partial(
                                    self.diffusion_process.score_analytical,
                                    dp=self.dp,
                                    constraints=self.constraints,
                                )
                            ),
                            10**c,
                            ('analytical' if self.diffusion_scale_range is None else f'analytical_c_{10**c:.3}')
                        )
                        for c in cs
                    ]
                )

            for f_bar, c, name in fs:
                dp_bar = process.Diffusion(
                    drift=f_bar,
                    diffusion=lambda *args: self.dp.diffusion(*args) * c,
                    inverse_diffusion=None,
                    diffusion_divergence=None,
                )

                self.constraints.visualise_paths(
                    key=key,
                    dp=dp_bar,
                    simulator=self.simulator,
                    constraints=self.constraints.reversed(),
                    filename=plots_path / f'{name}_bridge.png',
                    t0=1.0,
                    t1=0.001,
                    n_steps=1000,
                )

            for c in cs:
                self.constraints.visualise_paths(
                    key=key,
                    dp=process.Diffusion(
                        drift=self.dp.drift,
                        diffusion=lambda *args: self.dp.diffusion(*args) * 10**c,
                        inverse_diffusion=None,
                        diffusion_divergence=None,
                    ),
                    simulator=self.simulator,
                    constraints=self.constraints,
                    filename=plots_path / ('unconditional.png' if self.diffusion_scale_range is None else f'unconditional_c_{10**c:.3}.png'),
                    t0=0,
                    t1=1,
                    n_steps=1000,
                )

        if self.constraints.visualise_field is not None:
            fs = [
                (
                    lambda t, y:
                        self.diffusion_process.score_learned(
                            t,
                            y - (self.constraints.initial.reshape(y.shape[1:], order='F') if self.displacement else 0),
                            state=state,
                            c=jnp.ones_like(t) * self.diffusion_process.c
                        ),
                    'learned'
                )
            ]
            if hasattr(self.diffusion_process, 'score_analytical'):
                fs.append(
                    (
                        jax.vmap(
                            partial(
                                self.diffusion_process.score_analytical,
                                dp=self.dp,
                                constraints=self.constraints,
                            )
                        ),
                        'analytical'
                    )
                )

            for score, name in fs:
                self.constraints.visualise_field(
                    score=score,
                    filename=plots_path / f'{name}_score_vector_field.png',
                )

        if self.constraints.visualise_combination is not None:
            fs = [
                (
                    _f_bar(
                        self.dp,
                        lambda t, y:
                            self.diffusion_process.score_learned(
                                t[None],
                                (y - (self.constraints.initial.reshape(y.shape, order='F') if self.displacement else 0))[None],
                                state=state,
                                c=jnp.array([self.diffusion_process.c])
                            )[0]
                    ),
                    lambda t, y:
                        self.diffusion_process.score_learned(
                            t,
                            y - (self.constraints.initial.reshape(y.shape[1:], order='F') if self.displacement else 0),
                            state=state,
                            c=jnp.ones_like(t) * self.diffusion_process.c
                        ),
                    'learned'
                )
            ]
            if hasattr(self.diffusion_process, 'score_analytical'):
                fs.append(
                    (
                        _f_bar(
                            self.dp,
                            partial(
                                self.diffusion_process.score_analytical,
                                dp=self.dp,
                                constraints=self.constraints,
                            )
                        ),
                        jax.vmap(
                            partial(
                                self.diffusion_process.score_analytical,
                                dp=self.dp,
                                constraints=self.constraints,
                            )
                        ),
                        'analytical'
                    )
                )

            for f_bar, score, name in fs:
                dp_bar = process.Diffusion(
                    drift=f_bar,
                    diffusion=self.dp.diffusion,
                    inverse_diffusion=None,
                    diffusion_divergence=None,
                )

                self.constraints.visualise_combination(
                    score=score,
                    key=key,
                    dp=dp_bar,
                    simulator=self.simulator,
                    constraints=self.constraints.reversed(),
                    t0=1.0,
                    t1=0.001,
                    n_steps=1000,
                    filename=plots_path / f'{name}_score_vector_field_with_samples.png',
                )
