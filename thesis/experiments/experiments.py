import pathlib
from functools import partial

import jax
import jax.numpy as jnp
from flax.training import train_state

import thesis.processes.process as process
from thesis.experiments.constraints import Constraints
from thesis.experiments.diffusion_processes import DiffusionProcess
from thesis.experiments.simulators import Simulator


class Experiment:
    def __init__(
        self,
        key: jax.dtypes.prng_key,
        constraints: Constraints,
        diffusion_process: partial,
        simulator: partial,
        n: int,
    ) -> None:
        self.key = key

        self.constraints = constraints

        if 'constraints' in diffusion_process.func.__init__.__code__.co_varnames[:diffusion_process.func.__init__.__code__.co_argcount]:
            self.diffusion_process: DiffusionProcess = diffusion_process(constraints=self.constraints)
        else:
            self.diffusion_process: DiffusionProcess = diffusion_process()
        assert isinstance(self.diffusion_process, DiffusionProcess)

        self.simulator: Simulator = simulator(dp=self.diffusion_process.dp)
        assert isinstance(self.simulator, Simulator)

        self.n = n

    def __len__(self) -> int:
        return self.n
    
    def __getitem__(self, index: int) -> jax.Array:
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey = jax.random.split(self.key)
        ts, ys, n = self.simulator.simulate_sample_path(subkey, self.constraints.initial)

        return ts[:n], ys[:n], self.constraints.initial, self.diffusion_process.c
    
    def visualise(self, state: train_state.TrainState, plots_path: pathlib.Path):
        self.key, key = jax.random.split(self.key)

        if self.constraints.visualise_paths is not None:
            fs = [(partial(self.diffusion_process.f_bar_learned, dp=self.diffusion_process.dp, state=state, c=self.diffusion_process.c), 'learned')]
            if hasattr(self.diffusion_process, 'f_bar_analytical'):
                fs.append(
                    (
                        partial(
                            self.diffusion_process.f_bar_analytical,
                            dp=self.diffusion_process.dp,
                            constraints=Constraints(jnp.zeros_like(self.constraints.initial), self.constraints.initial) if self.simulator.displacement else self.constraints,
                        ),
                        'analytical'
                    )
                )

            for f_bar, name in fs:
                dp_bar = process.Diffusion(
                    d=self.diffusion_process.dp.d,
                    drift=f_bar,
                    diffusion=self.diffusion_process.dp.diffusion,
                    inverse_diffusion=None,
                    diffusion_divergence=None,
                )

                self.constraints.visualise_paths(
                    dp=dp_bar,
                    key=key,
                    filename=plots_path / f'{name}_bridge.png',
                    y0=self.constraints.terminal - (self.constraints.initial if self.simulator.displacement else jnp.zeros_like(self.constraints.initial)),
                    displacement=self.constraints.initial if self.simulator.displacement else jnp.zeros_like(self.constraints.initial),
                    t0=1.,
                    t1=0.001,
                    dt=-0.001,
                )

            self.constraints.visualise_paths(
                dp=self.diffusion_process.dp,
                key=key,
                filename=plots_path / 'unconditional.png',
                y0=jnp.zeros_like(self.constraints.initial) if self.simulator.displacement else self.constraints.initial,
                displacement=self.constraints.initial if self.simulator.displacement else jnp.zeros_like(self.constraints.initial),
                t0=0,
                t1=1,
                dt=0.001,
            )

        if self.constraints.visualise_field is not None:
            fs = [(partial(self.diffusion_process.score_learned, state=state, c=self.diffusion_process.c), 'learned')]
            if hasattr(self.diffusion_process, 'score_analytical'):
                fs.append(
                    (
                        jax.vmap(
                            partial(
                                self.diffusion_process.score_analytical,
                                dp=self.diffusion_process.dp,
                                constraints=Constraints(jnp.zeros_like(self.constraints.initial), self.constraints.initial) if self.simulator.displacement else self.constraints,
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
