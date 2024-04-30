import pathlib
from functools import partial
from typing import Callable

import jax
from flax.training import train_state

import thesis.processes.process as process
from thesis.experiments.constraints import Constraints
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
    ) -> None:
        self.key = key

        self.constraints = constraints
        self.diffusion_process = diffusion_process
        self.simulator = simulator

        self.displacement = displacement
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> jax.Array:
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey = jax.random.split(self.key)
        ts, ys = self.simulator.simulate_sample_path(subkey, self.diffusion_process.dp, self.constraints.initial, 0, 1, 0.01)
        if self.displacement:
            ys -= self.constraints.initial.reshape(ys[0].shape, order='F')

        return ts, ys, self.constraints.initial, self.diffusion_process.c
    
    def visualise(self, state: train_state.TrainState, plots_path: pathlib.Path):
        self.key, key = jax.random.split(self.key)

        if self.constraints.visualise_paths is not None:
            fs = [
                (
                    _f_bar(
                        self.diffusion_process.dp,
                        lambda t, y:
                            self.diffusion_process.score_learned(
                                t[None],
                                (y - (self.constraints.initial.reshape(y.shape, order='F') if self.displacement else 0))[None],
                                state=state,
                                c=self.diffusion_process.c
                            )[0]
                    ),
                    'learned'
                )
            ]
            if hasattr(self.diffusion_process, 'score_analytical'):
                fs.append(
                    (
                        _f_bar(
                            self.diffusion_process.dp,
                            partial(
                                self.diffusion_process.score_analytical,
                                dp=self.diffusion_process.dp,
                                constraints=self.constraints,
                            )
                        ),
                        'analytical'
                    )
                )

            for f_bar, name in fs:
                dp_bar = process.Diffusion(
                    drift=f_bar,
                    diffusion=self.diffusion_process.dp.diffusion,
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
                    dt=-0.001,
                )

            self.constraints.visualise_paths(
                key=key,
                dp=self.diffusion_process.dp,
                simulator=self.simulator,
                constraints=self.constraints,
                filename=plots_path / 'unconditional.png',
                t0=0,
                t1=1,
                dt=0.001,
            )

        if self.constraints.visualise_field is not None:
            fs = [
                (
                    lambda t, y:
                        self.diffusion_process.score_learned(
                            t,
                            y - (self.constraints.initial.reshape(y.shape, order='F') if self.displacement else 0),
                            state=state,
                            c=self.diffusion_process.c
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
                                dp=self.diffusion_process.dp,
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
