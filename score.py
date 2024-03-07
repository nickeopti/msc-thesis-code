import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

import diffusion
import process
import trainer


class Model(trainer.Module):
    dp: process.Diffusion
    learning_rate: float = 1e-3

    @nn.compact
    def __call__(self, t, y):
        z = nn.Sequential(
            [
                nn.Dense(16),
                nn.gelu,
                nn.Dense(32),
                nn.gelu,
                nn.Dense(16),
                nn.gelu,
                nn.Dense(self.dp.d)
            ]
        )(y)

        return z / t[:, None]

    @staticmethod
    @jax.jit
    def training_step(params, state: train_state.TrainState, dp: process.Diffusion, ts, ys, v):
        ps = state.apply_fn(params, ts[1:], ys[1:])

        def loss(p, t, y, y_next, dt):
            return jnp.linalg.norm(p + dp.inverse_diffusion(t, y) @ (y_next - y - dp.drift(t, y) * dt) / dt)**2

        l = jax.vmap(loss)(ps, ts[:-1], ys[:-1], ys[1:], ts[1:] - ts[:-1])
        return jnp.mean(l)

    @staticmethod
    @jax.jit
    def validation_step(params, state: train_state.TrainState, dp: process.Diffusion, ts, ys, v):
        ps = state.apply_fn(params, ts[1:], ys[1:])

        def loss(p, t, y):
            psi = -dp.inverse_diffusion(t, y) @ (y - v) / t
            return jnp.linalg.norm(p - psi)**2

        l = jax.vmap(loss)(ps, ts[1:], ys[1:])
        return jnp.mean(l)

    def initialise_params(self, rng):
        return self.init(rng, jnp.ones(100), jnp.ones((100, self.dp.d)))

    def configure_optimizers(self):
        return optax.adam(self.learning_rate)

    def on_fit_end(self, params, state, log_path):
        ...
        # TODO: Use the code from `use.py` in here,
        # and make sure to log in the right place


class Dataset:
    def __init__(self, key, dp: process.Diffusion, n: int) -> None:
        self.key = key
        self.dp = dp
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey = jax.random.split(self.key)

        match index % 2:
            case 0:
                y0 = -jnp.ones(2)
            case 1:
                y0 = jnp.ones(2) * 2
            case 2:
                y0 = jnp.array((2, -1))

        ts, ys, n = diffusion.get_data(dp=self.dp, y0=y0, key=subkey)

        # return self.dp, ts, ys, y0
        return self.dp, ts[:n], ys[:n], y0


def main():
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

    # dp = process.brownian_motion(jnp.eye(2))
    dp = process.brownian_motion(jnp.array([[1, 0.6], [0.6, 1]]))
    model = Model(dp, learning_rate=1e-3)

    t = trainer.Trainer(100)
    t.fit(
        subkey1,
        model,
        Dataset(subkey2, dp, 16),
        Dataset(subkey3, dp, 4),
    )


if __name__ == '__main__':
    main()
