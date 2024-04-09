from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
