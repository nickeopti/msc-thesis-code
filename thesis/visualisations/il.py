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
def visualise_sample_paths_1d(key: jax.dtypes.prng_key, dp: process.Diffusion, simulator: simulators.Simulator, constraints: Constraints, n: int = 10, **kwargs):
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
            plt.plot(*ys[:, k].T, color=f'C{2*k}', linewidth=0.2, alpha=1)
            plt.scatter(*ys[-1, k], color=f'C{2*k}')

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
        return ys.reshape((-1, *constraints.initial.shape), order='F')

    ys = jax.vmap(compute_path)(jax.random.split(key, n))
    _, n_steps, *_ = ys.shape
    # ys = ys[jnp.logical_and(ys[:, n_steps // 2, 0, 0] > 0, ys[:, n_steps // 2, 1, 0] < 0)]
    n_ys, *_ = ys.shape
    means = jnp.mean(ys, axis=0)

    for i in range(0, n_ys, n_ys // 100):
        for k in range(constraints.initial.shape[0]):
            plt.plot(*ys[i, :, k].T, color=f'C{2*k}', linewidth=0.2, alpha=0.2)

    for k in range(constraints.initial.shape[0]):
        plt.plot(*means[:, k].T, color='black', linewidth=1, alpha=1)
        plt.scatter(*means[-1, k], color='black')

    plt.gca().set_xlim(-1.5, 3.5)
    plt.gca().set_ylim(-1.5, 3.5)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()


def animate_mean_sample_path_2d_wide(key: jax.dtypes.prng_key, dp: process.Diffusion, simulator: simulators.Simulator, constraints: Constraints, filename: str, n: int = 100, **kwargs):
    def compute_path(key):
        _, ys = simulator.simulate_sample_path(key, dp, constraints.initial, **kwargs)
        return ys.reshape((-1, *constraints.initial.shape), order='F')

    ys = jax.vmap(compute_path)(jax.random.split(key, n))
    _, n_steps, *_ = ys.shape
    # ys = ys[jnp.logical_and(ys[:, n_steps // 2, 0, 0] > 0, ys[:, n_steps // 2, 1, 0] < 0)]
    n_ys, *_ = ys.shape
    means = jnp.mean(ys, axis=0)

    for t in range(0, n_steps, n_steps // 100):
        plt.figure()
        if t > 0:
            for i in range(0, n_ys, n_ys // 100):
                for k in range(constraints.initial.shape[0]):
                    plt.plot(*ys[i, -t:, k].T, color=f'C{2*k}', linewidth=0.2, alpha=0.2)

            for k in range(constraints.initial.shape[0]):
                plt.plot(*means[-t:, k].T, color='black', linewidth=1, alpha=1)

        for k in range(constraints.initial.shape[0]):
            plt.scatter(*constraints.terminal[k], color=f'C{2*k}', marker='o')
            plt.scatter(*constraints.initial[k], color=f'C{2*k}', marker='*')

        plt.gca().set_xlim(-1.5, 3.5)
        plt.gca().set_ylim(-1.5, 3.5)
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.tight_layout()

        name, extension = os.path.splitext(filename)
        plt.savefig(f'{name}_{t:03d}{extension}', dpi=300)
        plt.close()


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
    # u, v = score(jnp.ones(n**2) / 2., jax.vmap(lambda x: jnp.array((x[0], 0.5, x[1], 0)))(s)).T.reshape(4, n, n)[:2]
    # r = score(jnp.ones(n**2) / 2., jax.vmap(lambda x: jnp.stack((x, jnp.array((0.5, 0)))))(s))
    # u, v = r[:, 0].T.reshape(2, n, n)
    # print(r.shape)
    # print(u.shape)
    # print(v.shape)

    plt.contourf(xx, yy, jnp.sqrt(u**2 + v**2), levels=jnp.linspace(0, val, nv))
    plt.colorbar()
    plt.quiver(xs, ys, u, v)


@_plot
def visualise_vector_field_2d_with_sample_paths(
    score: Callable[[jax.Array, jax.Array], jax.Array],
    key: jax.dtypes.prng_key, dp: process.Diffusion, simulator: simulators.Simulator, constraints: Constraints, n_samples: int = 10,
    n: int = 20, a: float = -3, b: float = 3, val: float = 5, nv: int = 51,
    **kwargs
):
    xs = jnp.linspace(a, b, n)
    ys = jnp.linspace(a, b, n)
    xx, yy = jnp.meshgrid(xs, ys)

    s = jnp.stack((xx.flatten(), yy.flatten())).T
    u, v = score(jnp.ones(n**2) / 1., s).T.reshape(2, n, n)

    plt.contourf(xx, yy, jnp.sqrt(u**2 + v**2), levels=jnp.linspace(0, val, nv))
    plt.colorbar()
    plt.quiver(xs, ys, u, v)

    for _ in range(n_samples):
        key, subkey = jax.random.split(key)

        _, ys = simulator.simulate_sample_path(subkey, dp, constraints.initial, **kwargs)
        plt.plot(*ys.T, linewidth=1, alpha=0.6)
        plt.scatter(*ys[-1], alpha=1)


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

    plt.plot(_wrap(constraints.terminal[:, 0]), _wrap(constraints.terminal[:, 1]), color='blue', alpha=0.5)
    plt.gca().set_aspect('equal')


@_plot
def visualise_shape_evolution(key: jax.dtypes.prng_key, dp: process.Diffusion, simulator: simulators.Simulator, constraints: Constraints, n: int = 1, **kwargs):
    ax = plt.subplot(projection='3d')
    cm = matplotlib.colormaps['plasma']

    k, d = constraints.initial.shape

    for _ in range(n):
        key, subkey = jax.random.split(key)

        ts, ys = simulator.simulate_sample_path(subkey, dp, constraints.initial, **kwargs)
        ys = ys.reshape((-1, k * d), order='F')

        for i in range(0, ys.shape[0], 100):
            ax.plot(_wrap(ys[i, :k]), _wrap(ys[i, k:]), ts[i], color=cm(ts[i]))
        ax.plot(_wrap(ys[-1, :k]), _wrap(ys[-1, k:]), ts[-1], color=cm(ts[-1]))

    ax.elev = 15  # default 30

    ax.set_aspect('equalxy')
    ax.set_zlabel('$t$')
    plt.tight_layout()


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
            ax_terminal.scatter(*ys[0, [i, k + i, 2 * k + i]], alpha=1, color=f'C{i}', s=1, marker='o')
            ax_terminal.scatter(*ys[-1, [i, k + i, 2 * k + i]], alpha=1, color=f'C{i}', s=1, marker='*')

            ax_terminal.plot(*ys[:, [i, k + i, 2 * k + i]].T, linewidth=0.1, alpha=0.5, color=f'C{i}')

    for i in range(k):
        ax_true.scatter(*constraints.terminal[i], alpha=1, color=f'C{i}', s=1)

    ax_initial.set_aspect('equal')
    ax_terminal.set_aspect('equal')
    ax_true.set_aspect('equal')
    plt.tight_layout()
