import inspect
import os.path
from collections import OrderedDict
from functools import wraps
from typing import Callable

import cycler
import jax
import jax.dtypes
import jax.numpy as jnp
import jax.random
import matplotlib
import matplotlib.pyplot as plt

from thesis.experiments import Constraints, simulators
from thesis.processes import process

plt.rc('axes', prop_cycle=cycler.cycler(color=plt.colormaps.get_cmap('tab20').colors))


def _plot(f):
    @wraps(f)
    def inner(filename, *args, **kwargs):
        plt.figure()
        res = f(*args, **kwargs)
        plt.savefig(filename, dpi=600)
        plt.close()
        return res

    sig = inspect.signature(f)
    parameters = OrderedDict(sig.parameters)
    parameters['filename'] = inspect.Parameter(name='filename', kind=inspect.Parameter.KEYWORD_ONLY, annotation=str)
    parameters.move_to_end('filename', last=False)
    sig._parameters = parameters
    inner.__signature__ = sig
    return inner


def multiple(*fs):
    def call_all(*args, **kwargs):
        name, extension = os.path.splitext(kwargs['filename'])
        for i, f in enumerate(fs):
            kwargs['filename'] = f'{name}_{i}{extension}'
            f(*args, **kwargs)
    return call_all


def _wrap(ys: jax.Array) -> jax.Array:
    return jnp.hstack((ys, ys[0]))


@_plot
def visualise_sample_paths_1d(key: jax.dtypes.prng_key, dp: process.Diffusion, simulator: simulators.Simulator, constraints: Constraints, n: int = 5, **kwargs):
    for _ in range(n):
        key, subkey = jax.random.split(key)

        ts, ys = simulator.simulate_sample_path(subkey, dp, constraints.initial, **kwargs)
        plt.plot(ts, ys[:, 0], linewidth=1, alpha=0.6)
        plt.scatter(ts[-1], ys[-1], alpha=1)

    plt.xlabel('t')
    plt.ylabel('y')


@_plot
def visualise_sample_paths_2d(key: jax.dtypes.prng_key, dp: process.Diffusion, simulator: simulators.Simulator, constraints: Constraints, n: int = 5, **kwargs):
    for _ in range(n):
        key, subkey = jax.random.split(key)

        _, ys = simulator.simulate_sample_path(subkey, dp, constraints.initial, **kwargs)
        plt.plot(*ys.T, linewidth=1, alpha=0.6)
        plt.scatter(*ys[-1], alpha=1)


@_plot
def visualise_sample_paths_2d_wide(key: jax.dtypes.prng_key, dp: process.Diffusion, simulator: simulators.Simulator, constraints: Constraints, n: int = 5, **kwargs):
    for i in range(n):
        key, subkey = jax.random.split(key)

        _, ys = simulator.simulate_sample_path(subkey, dp, constraints.initial, **kwargs)
        ys = ys.reshape((-1, *constraints.initial.shape), order='F')

        for k in range(constraints.initial.shape[0]):
            plt.plot(*ys[:, k].T, color=f'C{i}', linewidth=0.2, alpha=1)
            plt.scatter(*ys[-1, k], color=f'C{i}')

    for k in range(constraints.initial.shape[0]):
        plt.plot(
            (constraints.initial[k, 0], constraints.terminal[k, 0]),
            (constraints.initial[k, 1], constraints.terminal[k, 1]),
            color='black',
            linestyle='--',
        )


@_plot
def visualise_mean_sample_path_2d_wide(key: jax.dtypes.prng_key, dp: process.Diffusion, simulator: simulators.Simulator, constraints: Constraints, n: int = 100, **kwargs):
    def compute_path(key):
        _, ys = simulator.simulate_sample_path(key, dp, constraints.initial, **kwargs)
        return ys

    ys = jnp.mean(
        jax.vmap(compute_path)(jax.random.split(key, n)),
        axis=0,
    ).reshape((-1, *constraints.initial.shape), order='F')

    for k in range(constraints.initial.shape[0]):
        plt.plot(*ys[:, k].T, color='black', linewidth=1, alpha=1)
        plt.scatter(*ys[-1, k], color='black')

    plt.gca().set_aspect('equal')


@_plot
def visualise_vector_field_1d(score: Callable[[jax.Array, jax.Array], jax.Array], n: int = 20, t0: float = 0.001, t1: float = 1, a: float = -3, b: float = 3, val: float = 5, nv: int = 51):
    xs = jnp.linspace(t0, t1, n)
    ys = jnp.linspace(a, b, n)
    xx, yy = jnp.meshgrid(xs, ys)

    v = score(xx.reshape(-1), yy.reshape(-1, 1)).T.reshape(n, n)

    plt.contourf(xx, yy, jnp.abs(v), levels=jnp.linspace(0, val, nv))
    plt.colorbar()
    plt.quiver(xs, ys, jnp.zeros((20, 20)), v.reshape(20, 20))

    plt.xlabel('t')


@_plot
def visualise_vector_field_2d(score: Callable[[jax.Array, jax.Array], jax.Array], n: int = 20, a: float = -3, b: float = 3, val: float = 5, nv: int = 51):
    xs = jnp.linspace(a, b, n)
    ys = jnp.linspace(a, b, n)
    xx, yy = jnp.meshgrid(xs, ys)

    s = jnp.stack((xx.flatten(), yy.flatten())).T
    u, v = score(jnp.ones(n**2) / 1., s).T.reshape(2, n, n)

    plt.contourf(xx, yy, jnp.sqrt(u**2 + v**2), levels=jnp.linspace(0, val, nv))
    plt.colorbar()
    plt.quiver(xs, ys, u, v)


@_plot
def visualise_shape_paths_2d(key: jax.dtypes.prng_key, dp: process.Diffusion, simulator: simulators.Simulator, constraints: Constraints, n: int = 1, **kwargs):
    k, d = constraints.initial.shape

    for _ in range(n):
        key, subkey = jax.random.split(key)

        _, ys = simulator.simulate_sample_path(subkey, dp, constraints.initial, **kwargs)
        ys = ys.reshape((-1, k * d), order='F')

        for i in range(k):
            plt.plot(*ys[:, [i, k + i]].T, linewidth=0.2, alpha=0.8, color=f'C{i}')
            plt.scatter(*ys[0, [i, k + i]], alpha=1, color=f'C{i}', marker='x')
            plt.scatter(*ys[-1, [i, k + i]], alpha=1, color=f'C{i}', marker='+')
        
        plt.plot(_wrap(ys[0, :k]), _wrap(ys[0, k:]), color='black')
        plt.plot(_wrap(ys[-1, :k]), _wrap(ys[-1, k:]), color='black', linestyle='--')

    # plt.plot(_wrap(constraints.terminal[:, 0]), _wrap(constraints.terminal[:, 1]), color='blue', alpha=0.5)
    plt.gca().set_aspect('equal')


@_plot
def visualise_shape_evolution(key: jax.dtypes.prng_key, dp: process.Diffusion, simulator: simulators.Simulator, constraints: Constraints, n: int = 1, **kwargs):
    ax = plt.subplot(projection='3d')
    cm = matplotlib.colormaps['plasma']

    k, d = constraints.initial.shape

    for _ in range(n):
        key, subkey = jax.random.split(key)

        _, ys = simulator.simulate_sample_path(subkey, dp, constraints.initial, **kwargs)
        ys = ys.reshape((-1, k * d), order='F')

        for i in range(0, ys.shape[0], 100):
            ax.plot(_wrap(ys[i, :k]), _wrap(ys[i, k:]), i / ys.shape[0], color=cm(i / ys.shape[0]))

    ax.set_aspect('equalxy')
    ax.set_zlabel('$t$')


@_plot
def visualise_shape_paths_3d_embedded(key: jax.dtypes.prng_key, dp: process.Diffusion, simulator: simulators.Simulator, constraints: Constraints, n: int = 1, **kwargs):
    ax = plt.subplot(projection='3d')

    k, d = constraints.initial.shape

    for _ in range(n):
        key, subkey = jax.random.split(key)

        _, ys = simulator.simulate_sample_path(subkey, dp, constraints.initial, **kwargs)
        ys = ys.reshape((-1, k * d), order='F')

        for i in range(k):
            ax.plot(*ys[:, [i, k + i]].T, linewidth=0.2, alpha=0.8, color=f'C{i}')

        ax.plot(_wrap(ys[0, :k]), _wrap(ys[0, k:2*k]), _wrap(ys[0, 2*k:]), color='black')
        ax.plot(_wrap(ys[-1, :k]), _wrap(ys[-1, k:2*k]), _wrap(ys[-1, 2*k:]), color='black', linestyle='--')
    
    ax.set_aspect('equalxy')


@_plot
def visualise_shape_paths_3d(key: jax.dtypes.prng_key, dp: process.Diffusion, simulator: simulators.Simulator, constraints: Constraints, n: int = 1, **kwargs):
    ax_initial = plt.subplot(2, 2, 1, projection='3d')
    ax_terminal = plt.subplot(2, 2, 2, projection='3d', sharex=ax_initial, sharey=ax_initial, sharez=ax_initial)
    ax_true = plt.subplot(2, 2, 4, projection='3d', sharex=ax_initial, sharey=ax_initial, sharez=ax_initial)

    k, d = constraints.initial.shape

    for _ in range(n):
        key, subkey = jax.random.split(key)

        _, ys = simulator.simulate_sample_path(subkey, dp, constraints.initial, **kwargs)
        ys = ys.reshape((-1, k * d), order='F')

        for i in range(k):
            ax_initial.scatter(*ys[0, [i, k + i, 2 * k + i]], alpha=1, color=f'C{i}', s=1)
            ax_terminal.scatter(*ys[-1, [i, k + i, 2 * k + i]], alpha=1, color=f'C{i}', s=1)
    
    ax_true.scatter(*constraints.terminal.T, alpha=1, color=f'C{i}', s=1)
    
    ax_initial.set_aspect('equal')
    ax_terminal.set_aspect('equal')
    ax_true.set_aspect('equal')
    plt.tight_layout()
