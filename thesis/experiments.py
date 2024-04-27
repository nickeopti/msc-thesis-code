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
    visualise_combination: Optional[Callable] = None
    c: float

    @staticmethod
    def score_analytical(t, y, dp: process.Diffusion, y0):
        return -dp.inverse_diffusion(t, y) @ (y - y0) / t
    
    @staticmethod
    def score_learned(t, y, state: train_state.TrainState, c: float):
        return state.apply_fn(state.params, t, y, c=c)

    @staticmethod
    def f_bar_analytical(t, y, dp: process.Diffusion, y0):
        s = -dp.inverse_diffusion(t, y) @ (y - y0) / t
        return dp.drift(t, y) - dp.diffusion(t, y) @ s - dp.diffusion_divergence(t, y)

    @staticmethod
    def f_bar_learned(t, y, dp: process.Diffusion, state: train_state.TrainState, c: float):
        s = state.apply_fn(state.params, t[None], y[None], c)[0]
        return dp.drift(t, y) - dp.diffusion(t, y) @ s - dp.diffusion_divergence(t, y)

    def visualise(self, state: train_state.TrainState, plots_path: pathlib.Path):
        self.key, key = jax.random.split(self.key)

        if self.visualise_paths is not None:
            assert hasattr(self, 'y0'), 'y0 needed to be set'
            assert hasattr(self, 'yT'), 'yT needed to be set'

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

            self.visualise_paths(
                dp=self.dp,
                key=key,
                filename=plots_path / 'unconditional.png',
                y0=self.y0,
                t0=0,
                t1=1,
                dt=0.001,
            )

        if self.visualise_field is not None:
            assert hasattr(self, 'y0'), 'y0 needed to be set'

            for score, name in (
                (jax.vmap(partial(self.score_analytical, dp=self.dp, y0=self.y0)), 'analytical'),
                (partial(self.score_learned, state=state, c=self.c), 'learned'),
            ):
                self.visualise_field(
                    score=score,
                    filename=plots_path / f'{name}_score_vector_field.png',
                )

        if self.visualise_combination is not None:
            assert hasattr(self, 'y0'), 'y0 needed to be set'
            assert hasattr(self, 'yT'), 'yT needed to be set'

            for f_bar, score, name in (
                (
                    partial(self.f_bar_analytical, dp=self.dp, y0=self.y0),
                    jax.vmap(partial(self.score_analytical, dp=self.dp, y0=self.y0)),
                    'analytical'
                ),
                (
                    partial(self.f_bar_learned, dp=self.dp, state=state, c=self.c),
                    partial(self.score_learned, state=state, c=self.c),
                    'learned'
                ),
            ):
                dp_bar = process.Diffusion(
                    d=self.dp.d,
                    drift=f_bar,
                    diffusion=self.dp.diffusion,
                    inverse_diffusion=None,
                    diffusion_divergence=None,
                )

                self.visualise_combination(
                    dp=dp_bar,
                    score=score,
                    key=key,
                    filename=plots_path / f'{name}_bridge.png',
                    y0=self.yT,
                    t0=1.,
                    t1=0.001,
                    dt=-0.001,
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

        self.get_data = jax.jit(lambda y0, key: diffusion.get_data(dp=self.dp, y0=y0, key=key))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey = jax.random.split(self.key)
        ts, ys, n = self.get_data(self.y0, subkey)

        return ts[:n], ys[:n], self.y0, 0


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

        self.get_data = jax.jit(lambda y0, key: diffusion.get_data(dp=self.dp, y0=y0, key=key))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey = jax.random.split(self.key)
        ts, ys, n = self.get_data(self.y0, subkey)

        return ts[:n], ys[:n], self.y0, self.c


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

        self.get_data = jax.jit(lambda y0, key: diffusion.get_data(dp=self.dp, y0=y0, key=key))

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
        ts, ys, n = self.get_data(y0=y0, key=subkey)

        return ts[:n], ys[:n], y0, self.c
    
    @staticmethod
    def score_analytical(t, y, dp: process.Diffusion, y0):
        y0_1 = -jnp.ones(2)
        y0_2 = jnp.ones(2)

        a = lambda t, y, y0: jnp.exp(-(y - y0).T @ dp.inverse_diffusion(t, y) @ (y - y0) / t)
        b = lambda t, y, y0: dp.inverse_diffusion(t, y) @ (y - y0) * a(t, y, y0) / t
        return -1 / (a(t, y, y0_1) + a(t, y, y0_2)) * (b(t, y, y0_1) + b(t, y, y0_2))


class BrownianND(Brownian):
    def __init__(self, key, d: int, variance: float, n: int) -> None:
        self.key = key
        self.y0 = jnp.zeros(d)
        self.dp = process.brownian_motion(jnp.identity(self.y0.size) * variance)
        self.n = n

        self.get_data = jax.jit(lambda y0, key: diffusion.get_data(dp=self.dp, y0=y0, key=key))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey = jax.random.split(self.key)
        ts, ys, n = self.get_data(y0=self.y0, key=subkey)

        return ts[:n], ys[:n], self.y0, 0


class BrownianCircleLandmarks(Brownian):
    visualise_paths = staticmethod(partial(illustrations.visualise_circle_sample_paths_f, n=1))

    def __init__(self, key, k: int, radius: float, radius_T: float, variance: float, n: int) -> None:
        self.key = key

        angles = jnp.linspace(0, 2 * jnp.pi, k, endpoint=False)
        xs = jnp.cos(angles) * radius
        ys = jnp.sin(angles) * radius
        self.y0 = jnp.hstack((xs, ys))

        xs_T = jnp.cos(angles) * radius_T
        ys_T = jnp.sin(angles) * radius_T
        self.yT = jnp.hstack((xs_T, ys_T))

        self.dp = process.brownian_motion(jnp.identity(2 * k) * variance)

        self.c = 0
        self.n = n

        self.get_data = jax.jit(lambda y0, key: diffusion.get_data(dp=self.dp, y0=y0, key=key))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey = jax.random.split(self.key)
        ts, ys, n = self.get_data(y0=self.y0, key=subkey)

        return ts[:n], ys[:n], self.y0, 0


class BrownianStationaryKernelCircleLandmarks(BrownianCircleLandmarks):
    def __init__(self, key, k: int, radius: float, radius_T: float, variance: float, n: int) -> None:
        super().__init__(key, k, radius, radius_T, variance, n)

        def kernel(x, y):
            return variance * jnp.exp(-jnp.linalg.norm(x - y)**2 / 1 / 2)

        def pairwise(f, xs):
            return jax.vmap(lambda x: jax.vmap(f, (0, None))(xs, x))(xs)

        k = jnp.vstack(
            (
                jnp.hstack((pairwise(kernel, jnp.vstack((self.y0[:k], self.y0[k:])).T), jnp.zeros((k, k)))),
                jnp.hstack((jnp.zeros((k, k)), pairwise(kernel, jnp.vstack((self.y0[:k], self.y0[k:])).T)))
            )
        )

        self.dp = process.brownian_motion(k)

        self.get_data = jax.jit(lambda y0, key: diffusion.get_data(dp=self.dp, y0=y0, key=key))


class BrownianStationaryKernelCircleLandmarksFactorised(BrownianCircleLandmarks):
    visualise_paths = None
    visualise_combination = staticmethod(partial(illustrations.visualise_circle_sample_paths_f_factorised, n=1))

    def __init__(self, key, k: int, radius: float, radius_T: float, variance: float, n: int) -> None:
        super().__init__(key, k, radius, radius_T, variance, n)
        self.k = k

        self.y0 = jnp.hstack((self.y0[:k].reshape(-1, 1), self.y0[k:].reshape(-1, 1)))
        self.yT = jnp.hstack((self.yT[:k].reshape(-1, 1), self.yT[k:].reshape(-1, 1)))

        def kernel(x, y):
            return variance * jnp.exp(-jnp.linalg.norm(x - y)**2 / 0.1 / 2)

        def pairwise(f, xs):
            return jax.vmap(lambda x: jax.vmap(f, (0, None))(xs, x))(xs)

        k = pairwise(kernel, self.y0)

        self.dp = process.brownian_motion(k)

        self.get_data = jax.jit(lambda y0, key: diffusion.get_data(dp=self.dp, y0=y0, key=key))

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey_x, subkey_y = jax.random.split(self.key, 3)
        ts_x, ys_x, n_x = self.get_data(y0=self.y0[:, 0], key=subkey_x)
        ts_y, ys_y, n_y = self.get_data(y0=self.y0[:, 1], key=subkey_y)

        ys = jnp.dstack((ys_x[:n_x], ys_y[:n_y]))

        assert n_x == n_y
        assert jnp.all(ts_x[:n_x] == ts_y[:n_y])

        return ts_x[:n_x], ys, self.y0, 0

    @staticmethod
    def f_bar_learned(t, y, dp: process.Diffusion, state: train_state.TrainState, c: float):
        s = state.apply_fn(state.params, t[None], y[None], c)[0, :, 0]
        return dp.drift(t, y) - dp.diffusion(t, y) @ s - dp.diffusion_divergence(t, y)


class BrownianStationaryKernelCircleLandmarks3D(Brownian):
    visualise_paths = None
    visualise_combination = staticmethod(partial(illustrations.visualise_circle_sample_paths_f_3d, n=1))

    def __init__(self, key, k: int, radius: float, radius_T: float, vertical_distance: float, variance: float, n: int) -> None:
        self.key = key

        angles = jnp.linspace(0, 2 * jnp.pi, k, endpoint=False)
        xs = jnp.cos(angles) * radius
        ys = jnp.sin(angles) * radius
        zs = jnp.sin(angles * 5) / 10
        self.y0 = jnp.hstack((xs, ys, zs))

        xs_T = jnp.cos(angles) * radius_T
        ys_T = jnp.sin(angles) * radius_T
        zs_T = zs + vertical_distance + xs_T / 10 + ys_T / 10
        self.yT = jnp.hstack((xs_T, ys_T, zs_T))

        def kernel(x, y):
            return variance * jnp.exp(-jnp.linalg.norm(x - y)**2 / 0.1 / 2)

        def pairwise(f, xs):
            return jax.vmap(lambda x: jax.vmap(f, (0, None))(xs, x))(xs)

        k = jnp.vstack(
            (
                jnp.hstack((pairwise(kernel, jnp.vstack((self.y0[:k], self.y0[k:2*k], self.y0[2*k:])).T), jnp.zeros((k, k)), jnp.zeros((k, k)))),
                jnp.hstack((jnp.zeros((k, k)), pairwise(kernel, jnp.vstack((self.y0[:k], self.y0[k:2*k], self.y0[2*k:])).T), jnp.zeros((k, k)))),
                jnp.hstack((jnp.zeros((k, k)), jnp.zeros((k, k)), pairwise(kernel, jnp.vstack((self.y0[:k], self.y0[k:2*k], self.y0[2*k:])).T)))
            )
        )

        self.dp = process.brownian_motion(k)

        self.c = 0
        self.n = n

        self.get_data = jax.jit(lambda y0, key: diffusion.get_data(dp=self.dp, y0=y0, key=key))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey = jax.random.split(self.key)
        ts, ys, n = self.get_data(y0=self.y0, key=subkey)

        return ts[:n], ys[:n], self.y0, 0


class BrownianStationaryKernelCircleLandmarks3DFactorised(Brownian):
    visualise_paths = None
    visualise_combination = staticmethod(partial(illustrations.visualise_circle_sample_paths_f_factorised_3d, n=1))

    def __init__(self, key, k: int, radius: float, radius_T: float, vertical_distance: float, variance: float, n: int) -> None:
        self.key = key

        angles = jnp.linspace(0, 2 * jnp.pi, k, endpoint=False).reshape(-1, 1)
        xs = jnp.cos(angles) * radius
        ys = jnp.sin(angles) * radius
        zs = jnp.sin(angles * 5) / 10
        self.y0 = jnp.hstack((xs, ys, zs))

        xs_T = jnp.cos(angles) * radius_T
        ys_T = jnp.sin(angles) * radius_T
        zs_T = zs + vertical_distance + xs_T / 10 + ys_T / 10
        self.yT = jnp.hstack((xs_T, ys_T, zs_T))

        def kernel(x, y):
            return variance * jnp.exp(-jnp.linalg.norm(x - y)**2 / 0.1 / 2)

        def pairwise(f, xs):
            return jax.vmap(lambda x: jax.vmap(f, (0, None))(xs, x))(xs)

        k = pairwise(kernel, self.y0)

        self.dp = process.brownian_motion(k)

        self.c = 0
        self.n = n

        self.get_data = jax.jit(lambda y0, key: diffusion.get_data(dp=self.dp, y0=y0, key=key))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey_x, subkey_y, subkey_z = jax.random.split(self.key, 4)
        ts_x, ys_x, n_x = self.get_data(y0=self.y0[:, 0], key=subkey_x)
        ts_y, ys_y, n_y = self.get_data(y0=self.y0[:, 1], key=subkey_y)
        ts_z, ys_z, n_z = self.get_data(y0=self.y0[:, 2], key=subkey_z)

        assert n_x == n_y == n_z
        assert jnp.all(ts_x[:n_x] == ts_y[:n_y])
        assert jnp.all(ts_y[:n_y] == ts_z[:n_z])

        ys = jnp.dstack((ys_x[:n_x], ys_y[:n_y], ys_z[:n_z]))

        return ts_x[:n_x], ys, self.y0, 0

    @staticmethod
    def f_bar_learned(t, y, dp: process.Diffusion, state: train_state.TrainState, c: float):
        s = state.apply_fn(state.params, t[None], y[None], c)[0, :, 0]
        return dp.drift(t, y) - dp.diffusion(t, y) @ s - dp.diffusion_divergence(t, y)


class BrownianStationaryKernelBallLandmarks3DFactorised(Brownian):
    visualise_paths = None
    visualise_combination = staticmethod(partial(illustrations.visualise_circle_sample_paths_f_factorised_3d_ball, n=1))

    def __init__(self, key, k: int, radius: float, radius_T: float, variance: float, n: int, gamma: float = 0.1) -> None:
        self.key = key

        # Fibonacci lattice / sphere
        # https://observablehq.com/@meetamit/fibonacci-lattices
        gr = (1 + jnp.sqrt(5)) / 2  # golden ratio

        def x(i):
            return (i * gr) % 1
        
        def y(i, n):
            return i / (n - 1)
        
        def theta(i):
            return x(i) * 2 * jnp.pi
        
        def phi(i, n):
            return jnp.acos(1 - 2 * y(i, n))

        def f(i, n):
            return (
                jnp.cos(theta(i)) * jnp.sin(phi(i, n)),
                jnp.cos(phi(i, n)),
                jnp.sin(theta(i)) * jnp.sin(phi(i, n)),
            )
        
        points = jnp.vstack(tuple(f(i, k) for i in range(k)))
        self.y0 = points * radius
        self.yT = points * radius_T

        def kernel(x, y):
            return variance * jnp.exp(-jnp.linalg.norm(x - y)**2 / gamma / 2)

        def pairwise(f, xs):
            return jax.vmap(lambda x: jax.vmap(f, (0, None))(xs, x))(xs)

        k = pairwise(kernel, self.y0)

        self.dp = process.brownian_motion(k)

        self.c = 0
        self.n = n

        self.get_data = jax.jit(lambda y0, key: diffusion.get_data(dp=self.dp, y0=y0, key=key))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey_x, subkey_y, subkey_z = jax.random.split(self.key, 4)
        ts_x, ys_x, n_x = self.get_data(y0=self.y0[:, 0], key=subkey_x)
        ts_y, ys_y, n_y = self.get_data(y0=self.y0[:, 1], key=subkey_y)
        ts_z, ys_z, n_z = self.get_data(y0=self.y0[:, 2], key=subkey_z)

        assert n_x == n_y == n_z
        assert jnp.all(ts_x[:n_x] == ts_y[:n_y])
        assert jnp.all(ts_y[:n_y] == ts_z[:n_z])

        ys = jnp.dstack((ys_x[:n_x], ys_y[:n_y], ys_z[:n_z]))

        return ts_x[:n_x], ys, self.y0, 0

    @staticmethod
    def f_bar_learned(t, y, dp: process.Diffusion, state: train_state.TrainState, c: float):
        s = state.apply_fn(state.params, t[None], y[None], c)[0, :, 0]
        return dp.drift(t, y) - dp.diffusion(t, y) @ s - dp.diffusion_divergence(t, y)
