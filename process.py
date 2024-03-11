from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct

Function = Callable[[jax.Array, jax.Array], jax.Array]


class Diffusion(struct.PyTreeNode):
    d: int
    drift: jax.Array
    diffusion: jax.Array
    inverse_diffusion: jax.Array
    diffusion_divergence: jax.Array


def brownian_motion(covariance: jax.Array) -> Diffusion:
    assert covariance.ndim == 2
    assert covariance.shape[0] == covariance.shape[1]
    d = covariance.shape[0]

    return Diffusion(
        d=d,
        drift=jnp.zeros(d),
        diffusion=covariance,
        inverse_diffusion=jnp.linalg.inv(covariance),
        diffusion_divergence=jnp.zeros(d),
    )
