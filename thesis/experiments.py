import abc
import pathlib
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax.training import train_state

import thesis.processes.diffusion as diffusion
import thesis.processes.process as process
from thesis.visualisations import illustrations


class Experiment(abc.ABC):
    dp: process.Diffusion

    @abc.abstractmethod
    def __len__(self) -> int:
        ...
    
    @abc.abstractmethod
    def __getitem__(self, index: int) -> Any:
        ...

    def visualise(self, state: train_state.TrainState, plots_path: pathlib.Path):
        ...


class Brownian(Experiment):
    visualise_paths: Optional[Callable] = None
    visualise_field: Optional[Callable] = None
    c: float

    @staticmethod
    def score_analytical(t, y, dp: process.Diffusion, y0):
        return -dp.inverse_diffusion @ (y - y0) / t
    
    @staticmethod
    def score_learned(t, y, state: train_state.TrainState, c: float):
        return state.apply_fn(state.params, t, y, c=c)

    @staticmethod
    def f_bar_analytical(t, y, dp: process.Diffusion, y0):
        s = -dp.inverse_diffusion @ (y - y0) / t
        return dp.drift - dp.diffusion @ s - dp.diffusion_divergence

    @staticmethod
    def f_bar_learned(t, y, dp: process.Diffusion, state: train_state.TrainState, c: float):
        s = state.apply_fn(state.params, t[None], y[None], c)[0]
        return dp.drift - dp.diffusion @ s - dp.diffusion_divergence

    def visualise(self, state: train_state.TrainState, plots_path: pathlib.Path):
        self.key, key = jax.random.split(self.key)

        if self.visualise_paths is not None:
            for f_bar, name in (
                (partial(self.f_bar_analytical, dp=self.dp, y0=self.y0), 'analytical'),
                (partial(self.f_bar_learned, dp=self.dp, state=state, c=self.c), 'learned'),
            ):
                dp_bar = process.Diffusion(
                    d=self.dp.d,
                    drift=f_bar,
                    diffusion=self.dp.diffusion,
                    inverse_diffusion=None,
                    diffusion_divergence=None,
                )

                self.visualise_paths(
                    dp=dp_bar,
                    key=key,
                    filename=plots_path / f'{name}_bridge.png',
                    y0=self.yT,
                    t0=1.,
                    t1=0.001,
                    dt=-0.001,
                )

            dp = process.Diffusion(
                d=self.dp.d,
                drift=lambda t, y: self.dp.drift,
                diffusion=self.dp.diffusion,
                inverse_diffusion=self.dp.inverse_diffusion,
                diffusion_divergence=self.dp.diffusion_divergence,
            )
            self.visualise_paths(
                dp=dp,
                key=key,
                filename=plots_path / 'unconditional.png',
                y0=self.y0,
                t0=0,
                t1=1,
                dt=0.001,
            )

        if self.visualise_field is not None:
            for score, name in (
                (jax.vmap(partial(self.score_analytical, dp=dp, y0=self.y0)), 'analytical'),
                (partial(self.score_learned, state=state, c=self.c), 'learned'),
            ):
                self.visualise_field(
                    score=score,
                    filename=plots_path / f'{name}_score_vector_field.png',
                )


class Brownian1D(Brownian):
    visualise_paths = staticmethod(illustrations.visualise_sample_paths_f_1d)
    visualise_field = staticmethod(partial(illustrations.visualise_vector_field_1d, t0=0.1, t1=1))

    def __init__(self, key, y0: jax.Array, yT: jax.Array, variance: float, n: int) -> None:
        self.key = key
        self.y0 = y0
        self.yT = yT
        self.c = variance
        self.dp = process.brownian_motion(jnp.array([[variance]]))
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey = jax.random.split(self.key)
        ts, ys, n = diffusion.get_data(dp=self.dp, y0=self.y0, key=subkey)

        return self.dp, ts[:n], ys[:n], self.y0, 0


class Brownian2D(Brownian):
    visualise_paths = staticmethod(illustrations.visualise_sample_paths_f)
    visualise_field = staticmethod(illustrations.visualise_vector_field)

    def __init__(self, key, y0: jax.Array, yT: jax.Array, variance: float, covariance: float, n: int) -> None:
        self.key = key
        self.y0 = y0
        self.yT = yT
        self.c = covariance
        self.dp = process.brownian_motion(
            jnp.array(
                [
                    [variance, covariance],
                    [covariance, variance]
                ]
            )
        )
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey = jax.random.split(self.key)
        ts, ys, n = diffusion.get_data(dp=self.dp, y0=self.y0, key=subkey)

        return self.dp, ts[:n], ys[:n], self.y0, self.dp.diffusion[0, 1]


class Brownian2DMixture(Brownian):
    visualise_paths = staticmethod(illustrations.visualise_sample_paths_f)
    visualise_field = staticmethod(illustrations.visualise_vector_field)

    def __init__(self, key, y0: jax.Array, yT: jax.Array, variance: float, covariance: float, n: int) -> None:
        self.key = key
        self.y0 = y0
        self.yT = yT
        self.c = covariance
        self.dp = process.brownian_motion(
            jnp.array(
                [
                    [variance, covariance],
                    [covariance, variance]
                ]
            )
        )
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        match index % 2:
            case 0:
                y0 = -jnp.ones(2)
            case 1:
                y0 = jnp.ones(2)

        self.key, subkey = jax.random.split(self.key)
        ts, ys, n = diffusion.get_data(dp=self.dp, y0=y0, key=subkey)

        return self.dp, ts[:n], ys[:n], y0, self.dp.diffusion[0, 1]
    
    @staticmethod
    def score_analytical(t, y, dp: process.Diffusion, y0):
        y0_1 = -jnp.ones(2)
        y0_2 = jnp.ones(2)

        a = lambda t, y, y0: jnp.exp(-(y - y0).T @ dp.inverse_diffusion @ (y - y0) / t)
        b = lambda t, y, y0: dp.inverse_diffusion @ (y - y0) * a(t, y, y0) / t
        return -1 / (a(t, y, y0_1) + a(t, y, y0_2)) * (b(t, y, y0_1) + b(t, y, y0_2))


class BrownianND(Brownian):
    def __init__(self, key, y0: jax.Array, n: int) -> None:
        self.key = key
        self.y0 = y0
        self.dp = process.brownian_motion(jnp.identity(y0.size))
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey = jax.random.split(self.key)
        ts, ys, n = diffusion.get_data(dp=self.dp, y0=self.y0, key=subkey)

        return self.dp, ts[:n], ys[:n], self.y0, 0
