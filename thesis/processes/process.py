from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct

Function = Callable[[jax.Array, jax.Array], jax.Array]


class Diffusion(struct.PyTreeNode):
    drift: Function
    diffusion: Function
    inverse_diffusion: Function
    diffusion_divergence: Function


def brownian_motion(covariance: jax.Array) -> Diffusion:
    assert covariance.ndim == 2
    assert covariance.shape[0] == covariance.shape[1]

    d = covariance.shape[0]
    inverse_covariance = jnp.linalg.inv(covariance)

    # TODO: Consider making hashable based on covariance matrix
    return Diffusion(
        drift=lambda t, y: jnp.zeros(d),
        diffusion=lambda t, y: covariance,
        inverse_diffusion=lambda t, y: inverse_covariance,
        diffusion_divergence=lambda t, y: jnp.zeros(d),
    )


def wide_brownian_motion(covariance: jax.Array, dim: int) -> Diffusion:
    assert covariance.ndim == 2
    assert covariance.shape[0] == covariance.shape[1]

    d = covariance.shape[0]
    inverse_covariance = jnp.linalg.inv(covariance)

    # TODO: Consider making hashable based on covariance matrix
    return Diffusion(
        drift=lambda t, y: jnp.zeros((d, dim)),
        diffusion=lambda t, y: covariance,
        inverse_diffusion=lambda t, y: inverse_covariance,
        diffusion_divergence=lambda t, y: jnp.zeros((d, dim)),
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

    # TODO: Consider making hashable based on k, d, variance, and gamma
    return Diffusion(
        drift=lambda t, y: jnp.zeros_like(y, shape=(y.size,)),
        diffusion=lambda t, y: diffusion(y),
        inverse_diffusion=lambda t, y: jnp.linalg.inv(diffusion(y)),
        diffusion_divergence=lambda t, y: divergence(y),
    )


def kunita_flow(k: int, d: int, variance: float, gamma: float) -> Diffusion:
    def kernel(x: jax.Array, y: jax.Array):
        return variance * jnp.exp(-jnp.linalg.norm(x - y)**2 / gamma / 2)

    def pairwise(f: Callable[[jax.Array, jax.Array], jax.Array], xs):
        return jax.vmap(lambda x: jax.vmap(lambda y: f(x, y))(xs))(xs)

    def divergence(t, y):
        return (
            variance / gamma *
            jax.vmap(
                lambda x_i: (
                    jnp.sum(
                        jax.vmap(
                            lambda x_j: (x_i - x_j) * jnp.exp(-jnp.linalg.norm(x_i - x_j)**2 / gamma / 2)
                        )(y),
                        axis=0,
                        keepdims=False,
                    )
                )
            )(y)
        )

    return Diffusion(
        drift=lambda t, y: jnp.zeros((k, d)),
        diffusion=lambda t, y: pairwise(kernel, y),
        inverse_diffusion=lambda t, y: jnp.linalg(pairwise(kernel, y)),
        diffusion_divergence=divergence,
    )
