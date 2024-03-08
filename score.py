import pathlib
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

import diffusion
import process
import trainer
import visualise

State = train_state.TrainState


class Model(trainer.Module[State]):
    dp: process.Diffusion
    learning_rate: float = 1e-3

    @nn.compact
    def __call__(self, t, y):
        z = nn.Sequential(
            [
                nn.Dense(16),
                nn.tanh,
                nn.Dense(32),
                nn.tanh,
                nn.Dense(16),
                nn.tanh,
                nn.Dense(self.dp.d)
            ]
        )(y)

        return z / t[:, None]

    @staticmethod
    @jax.jit
    def training_step(state: State, dp: process.Diffusion, ts, ys, v):
        ps = state.apply_fn(state.params, ts[1:], ys[1:])

        def loss(p, t, y, y_next, dt):
            return jnp.linalg.norm(p + dp.inverse_diffusion(t, y) @ (y_next - y - dp.drift(t, y) * dt) / dt)**2

        l = jax.vmap(loss)(ps, ts[:-1], ys[:-1], ys[1:], ts[1:] - ts[:-1])
        return jnp.mean(l)

    @staticmethod
    @jax.jit
    def validation_step(state: State, dp: process.Diffusion, ts, ys, v):
        ps = state.apply_fn(state.params, ts[1:], ys[1:])

        def loss(p, t, y):
            psi = -dp.inverse_diffusion(t, y) @ (y - v) / t
            return jnp.linalg.norm(p - psi)**2

        l = jax.vmap(loss)(ps, ts[1:], ys[1:])
        return jnp.mean(l)

    def initialise_params(self, rng):
        return self.init(rng, jnp.ones(100), jnp.ones((100, self.dp.d)))

    def configure_optimizers(self):
        return optax.adam(self.learning_rate)

    def on_fit_end(self, state: State, log_path: pathlib.Path):
        plots_path = log_path / 'plots'
        plots_path.mkdir(parents=True, exist_ok=True)

        # Deliberately reuse random key
        key = jax.random.PRNGKey(1)
        y0 = jnp.ones(self.dp.d) * 2

        def f_bar_analytical(t, y):
            s = -self.dp.inverse_diffusion(t, y) @ (y - y0) / t
            return self.dp.drift(t, y) - self.dp.diffusion(t, y) @ s - self.dp.diffusion_divergence(t, y)

        def f_bar_learned(t, y):
            s = state.apply_fn(state.params, t[None], y[None])[0]
            return self.dp.drift(t, y) - self.dp.diffusion(t, y) @ s - self.dp.diffusion_divergence(t, y)

        for f_bar, name in ((f_bar_analytical, 'analytical'), (f_bar_learned, 'learned')):
            dp_bar = process.Diffusion(
                d=self.dp.d,
                drift=f_bar,
                diffusion=self.dp.diffusion,
                inverse_diffusion=None,
                diffusion_divergence=None,
            )

            visualise.visualise_sample_paths(
                dp=dp_bar,
                key=key,
                filename=plots_path / f'{name}_bridge.png',
                y0=y0,
                t0=1.,
                t1=0.001,
                dt=-0.001,
            )

        visualise.visualise_sample_paths(
            dp=self.dp,
            key=key,
            filename=plots_path / 'unconditional.png',
            y0=y0,
            t0=0,
            t1=1,
            dt=0.001,
        )

        y0_1 = -jnp.ones(2)
        y0_2 = jnp.ones(2)

        a = lambda t, y, y0: jnp.exp(-(y - y0).T @ self.dp.inverse_diffusion(t, y) @ (y - y0))
        s = lambda t, y: -1 / (a(t, y, y0_1) + a(t, y, y0_2)) * (self.dp.inverse_diffusion(t, y) @ (y - y0_1) * a(t, y, y0_1) + self.dp.inverse_diffusion(t, y) @ (y - y0_2) * a(t, y, y0_2))

        visualise.visualise_vector_field(
            score=jax.vmap(s),
            filename=plots_path / 'analytical_score_vector_field.png',
        )

        visualise.visualise_vector_field(
            score=partial(state.apply_fn, state.params),
            filename=plots_path / 'learned_score_vector_field.png',
        )


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
                y0 = jnp.ones(2)
            case 2:
                y0 = jnp.array((2, -1))

        ts, ys, n = diffusion.get_data(dp=self.dp, y0=y0, key=subkey)

        return self.dp, ts[:n], ys[:n], y0


def main():
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

    dp = process.brownian_motion(2 * jnp.eye(2))
    # dp = process.brownian_motion(jnp.array([[1, 0.6], [0.6, 1]]))
    model = Model(dp, learning_rate=1e-3)

    t = trainer.Trainer(500)
    t.fit(
        subkey1,
        model,
        Dataset(subkey2, dp, 16),
        Dataset(subkey3, dp, 4),
    )


if __name__ == '__main__':
    main()
