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
