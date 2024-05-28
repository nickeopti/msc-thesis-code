import csv
import os.path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import thesis.processes.process as process
from thesis.experiments import Constraints
from thesis.visualisations import il


class PointConstraints(Constraints):
    def __init__(self, initial: jax.Array, terminal: jax.Array) -> None:
        self.initial = initial
        self.terminal = terminal

        match self.initial.shape:
            case (1,):
                self.visualise_paths = il.visualise_sample_paths_1d
                self.visualise_field = partial(il.visualise_vector_field_1d, t0=0.1, t1=1)
            case (2,):
                self.visualise_paths = il.visualise_sample_paths_2d
                self.visualise_field = il.visualise_vector_field_2d


class PointConstraints2D(Constraints):
    def __init__(self, initial: jax.Array, terminal: jax.Array) -> None:
        super().__init__(
            initial.reshape((-1, 2), order='F'),
            terminal.reshape((-1, 2), order='F')
        )

        self.visualise_paths = il.multiple(
            partial(il.visualise_sample_paths_2d_wide, n=1),
            partial(il.visualise_mean_sample_path_2d_wide, n=10000),
        )
        self.visualise_field = partial(il.visualise_vector_field_2d, a=-1, b=1)



class PointMixtureConstraints(Constraints):
    def __init__(self, initial_a: jax.Array, initial_b: jax.Array, terminal: jax.Array) -> None:
        self.initial_a = initial_a
        self.initial_b = initial_b
        self.terminal = terminal

        self.a = False

        match self.initial.shape:
            # TODO: Consider 1D mixture
            case (2,):
                self.visualise_paths = partial(il.visualise_sample_paths_2d, n=10)
                self.visualise_field = il.visualise_vector_field_2d
                self.visualise_combination = il.visualise_vector_field_2d_with_sample_paths

    @property
    def initial(self) -> jax.Array:
        self.a = not self.a

        if self.a:
            return self.initial_a
        else:
            return self.initial_b

    def reversed(self):
        return PointConstraints(initial=self.terminal, terminal=self.initial_b)

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
                self.visualise_paths = il.multiple(partial(il.visualise_shape_paths_2d, n=1), partial(il.visualise_shape_evolution, n=1))
            case (_, 3):
                self.visualise_paths = partial(il.visualise_shape_paths_3d, n=1)


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
        
        points = jnp.vstack(jnp.array([f(i, k) for i in range(k)]))

        initial = points * initial_radius * jnp.array((1, 1, skewness))
        terminal = points * terminal_radius * jnp.array((1, 1, skewness))

        super().__init__(initial, terminal)


class ButterflyLandmarks(LandmarksConstraints):
    def __init__(self, data_path: str, initial_species: str, terminal_species: str, every: int = 1) -> None:
        metadata = pd.read_csv(os.path.join(data_path, 'metadata.txt'), sep=';')
        landmarks = pd.read_csv(os.path.join(data_path, 'aligned.txt'), sep=',', header=None)

        initial = jnp.array(landmarks.loc[metadata['species'] == initial_species])[0].reshape((-1, 2))[::every] * 25
        terminal = jnp.array(landmarks.loc[metadata['species'] == terminal_species])[0].reshape((-1, 2))[::every] * 25

        super().__init__(initial, terminal)


class SkullLandmarks(LandmarksConstraints):
    def __init__(self, landmarks_info: str, initial_skull: str, terminal_skull: str, bone: str, every: int = 1) -> None:
        with open(landmarks_info) as csv_file:
            reader = csv.DictReader(csv_file)
            info = [row for row in reader]

        info = {row['id']: row for row in info}

        def extract_landmarks(path):
            with open(path) as csv_file:
                reader = csv.DictReader(csv_file)
                data = [{'id': row['id'], 'x': float(row['X']), 'y': float(row['Y']), 'z': float(row['Z'])} for row in reader]

            return jnp.array(
                [
                    (row['x'], row['y'], row['z'])
                    for row in [row for row in data if info[row['id']][bone] == 'x'][::every]
                ]
            )

        initial = extract_landmarks(initial_skull) * 100
        terminal = extract_landmarks(terminal_skull) * 100

        super().__init__(initial, terminal)
