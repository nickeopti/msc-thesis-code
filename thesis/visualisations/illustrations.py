from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

import thesis.processes.diffusion as diffusion
import thesis.processes.process as process


def visualise_sample_paths(dp: process.Diffusion, key, filename, n: int = 5, **kwargs):
    plt.figure()

    for _ in range(n):
        key, subkey = jax.random.split(key)

        _, ys, n = diffusion.get_data(
            dp=dp,
            key=subkey,
            **kwargs
        )

        plt.plot(*ys[:n].T, linewidth=1, alpha=0.6)
        plt.scatter(*ys[n-1], alpha=1)
        print(ys[n-1])

    plt.savefig(filename, dpi=600)


def visualise_sample_paths_1d(dp: process.Diffusion, key, filename, n: int = 5, **kwargs):
    plt.figure()

    for _ in range(n):
        key, subkey = jax.random.split(key)

        ts, ys, n = diffusion.get_data(
            dp=dp,
            key=subkey,
            **kwargs
        )

        plt.plot(ts[:n], ys[:n, 0], linewidth=1, alpha=0.6)
        plt.scatter(ts[n-1], ys[n-1], alpha=1)
        print(ys[n-1])

    plt.xlabel('t')
    plt.ylabel('y')

    plt.savefig(filename, dpi=600)


def visualise_sample_paths_f(dp: process.Diffusion, key, filename, n: int = 5, **kwargs):
    plt.figure()

    for _ in range(n):
        key, subkey = jax.random.split(key)

        _, ys, n = diffusion.get_paths(
            dp=dp,
            key=subkey,
            **kwargs
        )

        plt.plot(*ys[:n].T, linewidth=1, alpha=0.6)
        plt.scatter(*ys[n-1], alpha=1)
        print(ys[n-1])

    plt.savefig(filename, dpi=600)


def visualise_sample_paths_f_1d(dp: process.Diffusion, key, filename, n: int = 5, **kwargs):
    plt.figure()

    for _ in range(n):
        key, subkey = jax.random.split(key)

        ts, ys, n = diffusion.get_paths(
            dp=dp,
            key=subkey,
            **kwargs
        )

        plt.plot(ts[:n], ys[:n, 0], linewidth=1, alpha=0.6)
        plt.scatter(ts[n-1], ys[n-1], alpha=1)
        print(ys[n-1])

    plt.xlabel('t')
    plt.ylabel('y')

    plt.savefig(filename, dpi=600)


def visualise_vector_field(score: Callable[[jax.Array, jax.Array], jax.Array], filename, n: int = 20, a: float = -3, b: float = 3, val: float = 5, nv: int = 51):
    plt.figure()

    xs = np.linspace(a, b, n)
    ys = np.linspace(a, b, n)
    xx, yy = np.meshgrid(xs, ys)

    s = np.stack((xx.flatten(), yy.flatten())).T
    u, v = score(jnp.ones(n**2) / 1., s).T.reshape(2, n, n)

    plt.contourf(xx, yy, np.sqrt(u**2 + v**2), levels=jnp.linspace(0, val, nv))
    plt.colorbar()
    plt.quiver(xs, ys, u, v)

    plt.savefig(filename, dpi=600)


def visualise_vector_field_1d(score: Callable[[jax.Array, jax.Array], jax.Array], filename, n: int = 20, t0: float = 0.001, t1: float = 1, a: float = -3, b: float = 3, val: float = 5, nv: int = 51):
    plt.figure()

    xs = np.linspace(t0, t1, n)
    ys = np.linspace(a, b, n)
    xx, yy = np.meshgrid(xs, ys)

    v = score(xx.reshape(-1), yy.reshape(-1, 1)).T.reshape(n, n)

    plt.contourf(xx, yy, jnp.abs(v), levels=jnp.linspace(0, val, nv))
    plt.colorbar()
    plt.quiver(xs, ys, jnp.zeros((20, 20)), v.reshape(20, 20))

    plt.xlabel('t')

    plt.savefig(filename, dpi=600)


def visualise_circle_sample_paths_f(dp: process.Diffusion, key, filename, n: int = 5, **kwargs):
    plt.figure()
    import cycler
    plt.rc('axes', prop_cycle=cycler.cycler(color=plt.colormaps.get_cmap('tab20').colors))

    k = dp.diffusion.shape[0] // 2

    for _ in range(n):
        key, subkey = jax.random.split(key)

        _, ys, n = diffusion.get_paths(
            dp=dp,
            key=subkey,
            **kwargs
        )

        for i in range(k):
            # print(i, *ys[0, [i, k + i]])
            plt.plot(*ys[:n, [i, k + i]].T, linewidth=1, alpha=0.6, color=f'C{i}')
            plt.scatter(*ys[n - 1, [i, k + i]], alpha=1, color=f'C{i}', marker='+')

        for i in range(k):
            plt.scatter(*ys[0, [i, k + i]], alpha=1, color=f'C{i}', marker='x')

        polygon_T = patches.Polygon(jnp.vstack((ys[0, :k], ys[0, k:])).T, fill=False)
        polygon_0 = patches.Polygon(jnp.vstack((ys[n - 1, :k], ys[n - 1, k:])).T, fill=False, linestyle='--')

        plt.gca().add_patch(polygon_0)
        plt.gca().add_patch(polygon_T)

    plt.gca().set_aspect('equal')
    plt.savefig(filename, dpi=600)
