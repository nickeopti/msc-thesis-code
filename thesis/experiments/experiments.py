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

        if min_diffusion_scale and max_diffusion_scale:
            self.diffusion_scale_range = (min_diffusion_scale, max_diffusion_scale)
            assert self.diffusion_process.c == 1, 'Variance shall be set to 1'
        elif min_diffusion_scale or max_diffusion_scale:
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

        ts, ys = self.simulator.simulate_sample_path(subkey1, self.dp, initial, t0=0, t1=1, n_steps=1000)
        if self.displacement:
            ys -= initial.reshape(ys[0].shape, order='F')

        if self.diffusion_scale_range is not None:
            c = jax.random.uniform(subkey2, minval=self.diffusion_scale_range[0], maxval=self.diffusion_scale_range[1])
        else:
            c = self.diffusion_process.c

        return ts, ys, initial, c, (initial.reshape(ys[0].shape, order='F') if self.displacement else 0)

    def visualise(self, state: train_state.TrainState, plots_path: pathlib.Path):
        self.key, key = jax.random.split(self.key)

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
                                c=jnp.array([self.diffusion_process.c])
                            )[0]
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
                        'analytical'
                    )
                )

            for f_bar, name in fs:
                dp_bar = process.Diffusion(
                    drift=f_bar,
                    diffusion=self.dp.diffusion,
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

            self.constraints.visualise_paths(
                key=key,
                dp=self.dp,
                simulator=self.simulator,
                constraints=self.constraints,
                filename=plots_path / 'unconditional.png',
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
