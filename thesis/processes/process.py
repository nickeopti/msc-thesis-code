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
        )
    )
