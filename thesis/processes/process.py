from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct

Function = Callable[[jax.Array, jax.Array], jax.Array]


class Diffusion(struct.PyTreeNode):
    d: int
    drift: Function
    diffusion: Function
    inverse_diffusion: Function
    diffusion_divergence: Function


def brownian_motion(covariance: jax.Array) -> Diffusion:
    assert covariance.ndim == 2
    assert covariance.shape[0] == covariance.shape[1]

    d = covariance.shape[0]
    inverse_covariance = jnp.linalg.inv(covariance)

    return Diffusion(
        d=d,
        drift=lambda t, y: jnp.zeros(d),
        diffusion=lambda t, y: covariance,
        inverse_diffusion=lambda t, y: inverse_covariance,
        diffusion_divergence=lambda t, y: jnp.zeros(d),
    )


def q_process(k: int, d: int, variance: float, gamma: float) -> Diffusion:
    def kernel(x: jax.Array, y: jax.Array):
        return variance * jnp.exp(-jnp.linalg.norm(x - y)**2 / gamma / 2)

    def pairwise(f: Callable[[jax.Array, jax.Array], jax.Array], xs):
        return jax.vmap(lambda x: jax.vmap(f, (0, None))(xs, x))(xs)

    def diffusion(y: jax.Array):
        y = y.reshape((-1, d), order='F')
        # _, d = y.shape
        a = pairwise(kernel, y)
        return jnp.vstack(
            [
                jnp.hstack(
                    [
                        a if i == j else jnp.zeros_like(a)
                        for j in range(d)
                    ]
                )
                for i in range(d)
            ]
        )

    def divergence(y: jax.Array):
        y = y.reshape((-1, d), order='F')
        # _, d = y.shape
        a = (
            variance / gamma *
            jax.vmap(
                lambda x_i: (
                    jnp.sum(
                        jax.vmap(
                            lambda x_j: (x_i - x_j) * jnp.exp(-jnp.linalg.norm(x_i - x_j)**2 / gamma / 2)
                        )(y)
                    )
                )
            )(y)
        )
        return jnp.hstack([a for _ in range(d)])

    return Diffusion(
        d=k * d,
        drift=lambda t, y: jnp.zeros_like(y, shape=(y.size,)),
        diffusion=lambda t, y: diffusion(y),
        inverse_diffusion=lambda t, y: jnp.linalg.inv(diffusion(y)),
        diffusion_divergence=lambda t, y: divergence(y),
    )


def kunita_flow_tall(d: int, variance: float, gamma: float) -> Diffusion:
    def kernel(x, y):
        return variance * jnp.exp(-jnp.linalg.norm(x - y)**2 / gamma / 2)

    def pairwise(f, xs):
        return jax.vmap(lambda x: jax.vmap(f, (0, None))(xs, x))(xs)

    def diffusion(y):
        a = pairwise(kernel, jnp.vstack((y[:d], y[d:])).T)
        return jnp.vstack(
            (
                jnp.hstack((a, jnp.zeros((d, d)))),
                jnp.hstack((jnp.zeros((d, d)), a))
            )
        )

    def divergence(y):
        a = (
            variance / gamma *
            jax.vmap(
                lambda x_i: (
                    jnp.sum(
                        jax.vmap(
                            lambda x_j: (x_i - x_j) * jnp.exp(-jnp.linalg.norm(x_i - x_j)**2 / gamma / 2)
                        )(jnp.vstack((y[:d], y[d:])).T)
                    )
                )
            )(jnp.vstack((y[:d], y[d:])).T)
        )
        return jnp.hstack((a, a))

    return Diffusion(
        d=2 * d,
        drift=lambda t, y: jnp.zeros(2 * d),
        diffusion=lambda t, y: diffusion(y),
        inverse_diffusion=lambda t, y: jnp.linalg.inv(diffusion(y)),
        diffusion_divergence=lambda t, y: divergence(y),
    )


def kunita_flow(d: int, variance: float, gamma: float) -> Diffusion:
    def kernel(x, y):
        return variance * jnp.exp(-jnp.linalg.norm(x - y)**2 / gamma / 2)

    def pairwise(f, xs):
        return jax.vmap(lambda x: jax.vmap(f, (0, None))(xs, x))(xs)

    return Diffusion(
        d=d,
        drift=lambda t, y: jnp.zeros(d),
        diffusion=lambda t, y: pairwise(kernel, y),
        inverse_diffusion=lambda t, y: jnp.linalg.inv(pairwise(kernel, y)),
        diffusion_divergence=(
            lambda t, y: (
                variance / gamma *
                jax.vmap(
                    lambda x_i: (
                        jnp.sum(
                            jax.vmap(
                                lambda x_j: (x_i - x_j) * jnp.exp(-jnp.linalg.norm(x_i - x_j)**2 / gamma / 2)
                            )(y)
                        )
                    )
                )(y)
            )
        ),
    )
