import abc

import jax
import jax.numpy as jnp


class Objective(abc.ABC):
    @abc.abstractmethod
    def __call__(self, p, t, y, y_next, dt, drift, diffusion, inverse_diffusion) -> jax.Array:
        pass


class Exact(Objective):
    def __call__(self, p, t, y, y_next, dt, drift, diffusion, inverse_diffusion) -> jax.Array:
        raise ValueError('Use the ExactLong model to use the exact objective')


class Denoising(Objective):
    def __call__(self, p, t, y, y_next, dt, drift, diffusion, inverse_diffusion) -> jax.Array:
        return jnp.linalg.norm(p + inverse_diffusion @ (y_next - y - drift * dt) / dt)**2


class Heng(Denoising):
    def __call__(self, p, t, y, y_next, dt, drift, diffusion, inverse_diffusion) -> jax.Array:
        v = p + inverse_diffusion @ (y_next - y - drift * dt) / dt
        return v.T @ diffusion @ v


class Novel(Objective):
    def __call__(self, p, t, y, y_next, dt, drift, diffusion, inverse_diffusion) -> jax.Array:
        return p.T @ (diffusion * dt) @ p + 2 * p.T @ (y_next - y - drift * dt)
