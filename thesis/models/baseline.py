import pathlib
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

import thesis.lightning
import thesis.processes.process as process
import thesis.visualisations.illustrations as illustrations

State = train_state.TrainState


class Model(thesis.lightning.Module[State]):
    dp: process.Diffusion
    learning_rate: float = 1e-3

    @nn.compact
    def __call__(self, t, y, c):
        cv = jnp.ones_like(t) * c
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
        )(jnp.hstack((cv[:, None], y)))

        return z / t[:, None]

    @staticmethod
    @jax.jit
    def training_step(state: State, dp: process.Diffusion, ts, ys, v, c):
        ps = state.apply_fn(state.params, ts[1:], ys[1:], c)

        # y0_1 = -jnp.ones(2)
        # y0_2 = jnp.ones(2)

        def loss(p, t, y, y_next, dt):
            return jnp.linalg.norm(p + dp.inverse_diffusion @ (y_next - y - dp.drift * dt) / dt)**2
            # a = lambda t, y, y0: jnp.exp(-(y - y0).T @ dp.inverse_diffusion @ (y - y0))
            # s = lambda t, y: -1 / (a(t, y, y0_1) + a(t, y, y0_2)) * (dp.inverse_diffusion @ (y - y0_1) * a(t, y, y0_1) + dp.inverse_diffusion @ (y - y0_2) * a(t, y, y0_2))
            # return jnp.linalg.norm(p - s(t, y))**2

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

    def on_fit_end(self, state: State, log_path: pathlib.Path, c: float = 0):
        plots_path = log_path / 'plots'
        plots_path.mkdir(parents=True, exist_ok=True)

        # Deliberately reuse random key
        key = jax.random.PRNGKey(1)
        y0 = jnp.ones(self.dp.d) * 2

        def f_bar_analytical(t, y):
            s = -self.dp.inverse_diffusion @ (y - y0) / t
            return self.dp.drift - self.dp.diffusion @ s - self.dp.diffusion_divergence

        def f_bar_learned(t, y):
            s = state.apply_fn(state.params, t[None], y[None], c)[0]
            return self.dp.drift - self.dp.diffusion @ s - self.dp.diffusion_divergence

        for f_bar, name in ((f_bar_analytical, 'analytical'), (f_bar_learned, 'learned')):
            dp_bar = process.Diffusion(
                d=self.dp.d,
                drift=f_bar,
                diffusion=self.dp.diffusion,
                inverse_diffusion=None,
                diffusion_divergence=None,
            )

            illustrations.visualise_sample_paths_f(
                dp=dp_bar,
                key=key,
                filename=plots_path / f'{name}_bridge.png',
                y0=y0,
                t0=1.,
                t1=0.001,
                dt=-0.001,
            )

        illustrations.visualise_sample_paths(
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

        a = lambda t, y, y0: jnp.exp(-(y - y0).T @ self.dp.inverse_diffusion @ (y - y0) / t)
        s = lambda t, y: -1 / (a(t, y, y0_1) + a(t, y, y0_2)) * (self.dp.inverse_diffusion @ (y - y0_1) * a(t, y, y0_1) / t + self.dp.inverse_diffusion @ (y - y0_2) * a(t, y, y0_2) / t)

        illustrations.visualise_vector_field(
            score=jax.vmap(s),
            filename=plots_path / 'analytical_score_vector_field.png',
        )

        illustrations.visualise_vector_field(
            score=partial(state.apply_fn, state.params, c=c),
            filename=plots_path / 'learned_score_vector_field.png',
        )
