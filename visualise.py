from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax.training import train_state

import diffusion
import process


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


def visualise_vector_field(score: Callable[[jax.Array, jax.Array], jax.Array], filename, n: int = 20, a: float = -3, b: float = 3):
    plt.figure()

    xs = np.linspace(a, b, n)
    ys = np.linspace(a, b, n)
    xx, yy = np.meshgrid(xs, ys)

    s = np.stack((xx.flatten(), yy.flatten())).T
    u, v = score(jnp.ones(n**2) / 2., s).T.reshape(2, n, n)

    plt.contourf(xx, yy, np.sqrt(u**2 + v**2), levels=jnp.linspace(0, 4, 50))
    plt.colorbar()
    plt.quiver(xs, ys, u, v)

    plt.savefig(filename, dpi=600)
