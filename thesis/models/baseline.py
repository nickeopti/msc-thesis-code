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
                nn.Dense(self.dp.d)
            ]
        )(jnp.hstack((cv[:, None], y)))

        return z / t[:, None]

    @staticmethod
    @jax.jit
    def training_step(state: State, dp: process.Diffusion, ts, ys, v, c):
        ps = state.apply_fn(state.params, ts[1:], ys[1:], c)

        def loss(p, t, y, y_next, dt):
            return jnp.linalg.norm(p + dp.inverse_diffusion @ (y_next - y - dp.drift * dt) / dt)**2

        l = jax.vmap(loss)(ps, ts[:-1], ys[:-1], ys[1:], ts[1:] - ts[:-1])
        return jnp.mean(l)

    @staticmethod
    @jax.jit
    def validation_step(state: State, dp: process.Diffusion, ts, ys, v, c):
        ps = state.apply_fn(state.params, ts[1:], ys[1:], c)

        def loss(p, t, y):
            psi = -dp.inverse_diffusion @ (y - v) / t
            return jnp.linalg.norm(p - psi)**2

        l = jax.vmap(loss)(ps, ts[1:], ys[1:])
        return jnp.mean(l)

    def initialise_params(self, rng):
        return self.init(rng, jnp.ones(100), jnp.ones((100, self.dp.d)), 0)

    def configure_optimizers(self):
        return optax.adam(self.learning_rate)
