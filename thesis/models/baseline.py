import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

import thesis.lightning
import thesis.processes.process as process

State = train_state.TrainState


class Model(thesis.lightning.Module[State]):
    dp: process.Diffusion
    dim: int
    learning_rate: float = 1e-3

    @nn.compact
    def __call__(self, t, y, c):
        cv = jnp.ones_like(t) * c
        z = nn.Sequential(
            [
                nn.Dense(64),
                nn.gelu,
                nn.Dense(128),
                nn.gelu,
                nn.Dense(256),
                nn.gelu,
                nn.Dense(128),
                nn.gelu,
                nn.Dense(64),
                nn.gelu,
                nn.Dense(self.dim)
            ]
        )(jnp.hstack((cv[:, None], y)))

        return z / t[:, None]

    def make_training_step(self):
        @jax.jit
        def training_step(state: State, ts, ys, v, c):
            ps = state.apply_fn(state.params, ts[1:], ys[1:], c)

            def loss(p, t, y, y_next, dt):
                # return jnp.linalg.norm(p + self.dp.inverse_diffusion(t, y) @ (y_next - y - self.dp.drift(t, y) * dt) / dt)**2
                return p.T @ self.dp.diffusion(t, y) * dt @ p + 2 * p.T @ (y_next - y - self.dp.drift(t, y) * dt)

            l = jax.vmap(loss)(ps, ts[:-1], ys[:-1], ys[1:], ts[1:] - ts[:-1])
            return jnp.mean(l)

        return training_step

    def make_validation_step(self):
        @jax.jit
        def validation_step(state: State, ts, ys, v, c):
            ps = state.apply_fn(state.params, ts[1:], ys[1:], c)

            def loss(p, t, y):
                psi = -self.dp.inverse_diffusion(t, y) @ (y - v.reshape(-1, order='F')) / t
                return jnp.linalg.norm(p - psi)**2

            l = jax.vmap(loss)(ps, ts[1:], ys[1:])
            return jnp.mean(l)

        return validation_step

    def initialise_params(self, rng):
        return self.init(rng, jnp.ones(100), jnp.ones((100, self.dim)), 0)

    def configure_optimizers(self):
        return optax.adam(self.learning_rate)


class Factorised(Model):
    @nn.compact
    def __call__(self, t, y, c):
        cv = jnp.ones_like(t) * c
        z = jax.vmap(
            lambda x: nn.Sequential(
                [
                    nn.Dense(64),
                    nn.gelu,
                    nn.Dense(128),
                    nn.gelu,
                    nn.Dense(256),
                    nn.gelu,
                    nn.Dense(128),
                    nn.gelu,
                    nn.Dense(64),
                    nn.gelu,
                    nn.Dense(self.dim),
                ]
            )(jnp.hstack((cv[:, None], x))),
            in_axes=2,
            out_axes=2,
        )(y)

        return z / t[:, None, None]

    def make_training_step(self):
        @jax.jit
        def training_step(state: State, ts, ys, v, c):
            ps = state.apply_fn(state.params, ts[1:], ys[1:], c)

            def loss(p, t, y, y_next, dt):
                return jnp.sum(
                    jax.vmap(
                        # lambda p, y, y_next: jnp.linalg.norm(p + self.dp.inverse_diffusion(t, y) @ (y_next - y - self.dp.drift(t, y) * dt) / dt)**2,
                        lambda p, y, y_next, d: p.T @ self.dp.diffusion(t, y) * dt @ p + 2 * p.T @ (y_next - y - d * dt),
                        in_axes=(1, 1, 1, 1),
                    )(p, y, y_next, self.dp.drift(t, y))
                )

            l = jax.vmap(loss)(ps, ts[:-1], ys[:-1], ys[1:], ts[1:] - ts[:-1])
            return jnp.mean(l)
        
        return training_step

    def initialise_params(self, rng):
        return self.init(rng, jnp.ones(100), jnp.ones((100, self.dim, 2)), 0)
