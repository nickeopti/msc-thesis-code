from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct

Function = Callable[[jax.Array, jax.Array], jax.Array]


class Diffusion(struct.PyTreeNode):
    d: int
    drift: Function = struct.field(pytree_node=False)
    diffusion: Function = struct.field(pytree_node=False)
    inverse_diffusion: Function = struct.field(pytree_node=False)
    diffusion_divergence: Function = struct.field(pytree_node=False)


def brownian_motion(covariance: jax.Array) -> Diffusion:
    assert covariance.ndim == 2
    assert covariance.shape[0] == covariance.shape[1]

    return Diffusion(
        d=covariance.shape[0],
        drift=lambda t, y: jnp.zeros_like(y),
        diffusion=lambda t, y: covariance,
        inverse_diffusion=lambda t, y: jnp.linalg.inv(covariance),
        diffusion_divergence=lambda t, y: jnp.zeros_like(y),
    )
