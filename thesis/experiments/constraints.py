from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

import thesis.processes.process as process
from thesis.visualisations import illustrations

PathsVisualiser = Callable
FieldVisualiser = Callable


class Constraints:
    initial: jax.Array
    terminal: jax.Array

    visualise_paths: Optional[PathsVisualiser] = None
    visualise_field: Optional[FieldVisualiser] = None

    def __init__(self, initial, terminal) -> None:
        self.initial = initial
        self.terminal = terminal


class PointConstraints(Constraints):
    def __init__(self, initial: jax.Array, terminal: jax.Array) -> None:
        self.initial = initial
        self.terminal = terminal

        match self.initial.shape:
            case (1,):
                self.visualise_paths = illustrations.visualise_sample_paths_f_1d
                self.visualise_field = partial(illustrations.visualise_vector_field_1d, t0=0.1, t1=1)
            case (2,):
                self.visualise_paths = illustrations.visualise_sample_paths_f
                self.visualise_field = illustrations.visualise_vector_field


class PointMixtureConstraints(Constraints):
    def __init__(self, initial_a: jax.Array, initial_b: jax.Array, terminal: jax.Array) -> None:
        self.initial_a = initial_a
        self.initial_b = initial_b
        self.terminal = terminal

        self.a = False

        match self.initial.shape:
            # TODO: Consider 1D mixture
            case (2,):
                self.visualise_paths = illustrations.visualise_sample_paths_f
                self.visualise_field = illustrations.visualise_vector_field

    @property
    def initial(self) -> jax.Array:
        self.a = not self.a

        if self.a:
            return self.initial_a
        else:
            return self.initial_b

    @staticmethod
    def score_analytical(t, y, dp: process.Diffusion, constraints: 'PointMixtureConstraints'):
        y0_1 = constraints.initial_a
        y0_2 = constraints.initial_b

        a = lambda t, y, y0: jnp.exp(-(y - y0).T @ dp.inverse_diffusion(t, y) @ (y - y0) / t)
        b = lambda t, y, y0: dp.inverse_diffusion(t, y) @ (y - y0) * a(t, y, y0) / t
        return -1 / (a(t, y, y0_1) + a(t, y, y0_2)) * (b(t, y, y0_1) + b(t, y, y0_2))


class LandmarksConstraints(PointConstraints):
    def __init__(self, initial: jax.Array, terminal: jax.Array) -> None:
        super().__init__(initial, terminal)

        match self.initial.shape:
            case (_, 2):
                self.visualise_paths = partial(illustrations.visualise_circle_sample_paths_f, n=3)
            case (_, 3):
                self.visualise_paths = partial(illustrations.visualise_circle_sample_paths_f_3d, n=1)


class CircleLandmarks(LandmarksConstraints):
    def __init__(self, k: int, initial_radius: float, terminal_radius: float, skewness: float = 1) -> None:
        angles = jnp.linspace(0, 2 * jnp.pi, k, endpoint=False)

        xs = jnp.cos(angles) * initial_radius * skewness
        ys = jnp.sin(angles) * initial_radius
        initial = jnp.vstack((xs, ys)).T

        xs_T = jnp.cos(angles) * terminal_radius * skewness
        ys_T = jnp.sin(angles) * terminal_radius
        terminal = jnp.vstack((xs_T, ys_T)).T

        super().__init__(initial, terminal)


class BallLandmarks(LandmarksConstraints):
    def __init__(self, k: int, initial_radius: float, terminal_radius: float, skewness: float = 1) -> None:
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

        initial = points * initial_radius * jnp.array((1, 1, skewness))
        terminal = points * terminal_radius * jnp.array((1, 1, skewness))

        super().__init__(initial, terminal)

        self.visualise_paths = partial(illustrations.visualise_circle_sample_paths_f_factorised_3d_ball, n=1)


class ButterflyLandmarks(LandmarksConstraints):
    def __init__(self, initial_butterfly: str, terminal_butterfly: str, every: int = 1) -> None:
        initial: np.ndarray = np.load(initial_butterfly)[::every]
        terminal: np.ndarray = np.load(terminal_butterfly)[::every]

        super().__init__(initial, terminal)