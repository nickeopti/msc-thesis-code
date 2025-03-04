import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

from thesis import lightning
from thesis.models import objectives
from thesis.processes import process

State = train_state.TrainState


def add_ddid(objective: objectives.Objective, dp: process.Diffusion, offset):
    def compute(p, t, y, y_next, dt):
        drift = dp.drift(t, y + offset)
        diffusion = dp.diffusion(t, y + offset)
        inverse_diffusion = dp.inverse_diffusion(t, y + offset)

        return objective(p, t, y, y_next, dt, drift, diffusion, inverse_diffusion)

    return compute


class Long(lightning.Module[State]):
    dp: process.Diffusion
    network: nn.Module
    objective: objectives.Objective
    dim: int
    learning_rate: float = 1e-3

    @nn.compact
    def __call__(self, t, y, c):
        return self.network(dim=self.dim)(jnp.hstack((t[:, None], c[:, None], y))) / t[:, None]

    def make_training_step(self):
        def training_step(state: State, ts, ys, v, c, offset):
            ps = state.apply_fn(state.params, jnp.hstack(ts[:, 1:]), jnp.vstack(ys[:, 1:]), jnp.hstack(jnp.ones_like(ts[:, 1:]) * c[:, None]))

            l = jax.vmap(add_ddid(self.objective, self.dp, offset[0]))(ps, jnp.hstack(ts[:, :-1]), jnp.vstack(ys[:, :-1]), jnp.vstack(ys[:, 1:]), jnp.hstack(ts[:, 1:] - ts[:, :-1]))

            return jnp.mean(l)

        return training_step

    def make_validation_step(self):
        @jax.jit
        def validation_step(state: State, ts, ys, v, c, offset):
            ps = state.apply_fn(state.params, ts[1:], ys[1:], jnp.ones_like(ts[1:]) * c)

            def loss(p, t, y):
                psi = -self.dp.inverse_diffusion(t, y + offset[0]) @ (y - v.reshape(-1, order='F')) / t
                return jnp.linalg.norm(p - psi)**2

            l = jax.vmap(loss)(ps, ts[1:], ys[1:])
            return jnp.mean(l)

        return validation_step

    @property
    def init_params(self):
        return jnp.ones(100), jnp.ones((100, self.dim)), jnp.zeros(100)

    def initialise_params(self, rng):
        return self.init(rng, *self.init_params)

    def configure_optimizers(self):
        return optax.chain(optax.adaptive_grad_clip(2), optax.adam(self.learning_rate))


class LongCollection(Long):
    def make_training_step(self):
        def training_step(state: State, ts, ys, v, c, offset):
            print(ys.shape, v.shape, jnp.tile(jax.vmap(lambda u: u.reshape(ys.shape[2:], order='F'))(v)[:, None], (1, ys.shape[1] - 1, *([1] * (ys.ndim - 2)))).shape)
            ps = state.apply_fn(
                state.params,
                jnp.hstack(ts[:, 1:]),
                jnp.hstack(
                    (
                        # jnp.tile(v.reshape((ys.shape[0], *ys.shape[2:]), order='F'), (ys.shape[1] - 1, *([1] * (ys.ndim - 2)))),
                        # jnp.tile(jax.vmap(lambda u: u.reshape(ys.shape[2:], order='F'))(v), (ys.shape[1] - 1, *([1] * (ys.ndim - 2)))),
                        # jnp.tile(jax.vmap(lambda u: u.reshape(ys.shape[2:], order='F'))(v)[:, None], (1, ys.shape[1] - 1, *([1] * (ys.ndim - 2)))),
                        jnp.vstack(jnp.tile(jax.vmap(lambda u: u.reshape(ys.shape[2:], order='F'))(v)[:, None], (1, ys.shape[1], *([1] * (ys.ndim - 2))))[:, 1:]),
                        jnp.vstack(ys[:, 1:])
                    )
                ),
                jnp.hstack(jnp.ones_like(ts[:, 1:]) * c[:, None])
            )

            l = jax.vmap(add_ddid(self.objective, self.dp, offset[0]))(ps, jnp.hstack(ts[:, :-1]), jnp.vstack(ys[:, :-1]), jnp.vstack(ys[:, 1:]), jnp.hstack(ts[:, 1:] - ts[:, :-1]))

            return jnp.mean(l)

        return training_step

    @property
    def init_params(self):
        return jnp.ones(100), jnp.ones((100, 2 * self.dim)), jnp.zeros(100)


class LongVariance(Long):
    @nn.compact
    def __call__(self, t, y, c):
        return self.network(dim=self.dim)(jnp.hstack((t[:, None], y)), c) / t[:, None]


class LongCollectionVariance(LongCollection):
    @nn.compact
    def __call__(self, t, y, c):
        return self.network(dim=self.dim)(jnp.hstack((t[:, None], y)), c) / t[:, None]


class ExactLong(Long):
    def make_training_step(self):
        def training_step(state: State, ts, ys, v, c, offset):
            def predict(ts, ys, cs):
                return state.apply_fn(state.params, ts, ys, cs)
            
            ps = predict(jnp.hstack(ts[:, 1:]), jnp.vstack(ys[:, 1:]), jnp.hstack(jnp.ones_like(ts[:, 1:]) * c[:, None]))
            dps = jax.vmap(lambda t, y, c: jax.vmap(lambda i: jax.grad(lambda y: predict(t[None], y[None], c[None])[0, i])(y)[i])(jnp.arange(y.size)))(jnp.hstack(ts[:, 1:]), jnp.vstack(ys[:, 1:]), jnp.hstack(jnp.ones_like(ts[:, 1:]) * c[:, None]))

            l = jax.vmap(lambda p, dp: jnp.sum(dp + p**2 / 2))(ps, dps)

            return jnp.mean(l)

        return training_step


class Factorised(Long):
    @nn.compact
    def __call__(self, t, y, c):
        return jax.vmap(
            lambda x: self.network(dim=self.dim)(jnp.hstack((t[:, None], c[:, None], x))) / t[:, None],
            in_axes=2,
            out_axes=2,
        )(y)

    def make_training_step(self):
        def training_step(state: State, ts, ys, v, c, offset):
            ps = state.apply_fn(state.params, jnp.hstack(ts[:, 1:]), jnp.vstack(ys[:, 1:]), jnp.hstack(jnp.ones_like(ts[:, 1:]) * c[:, None]))

            def loss(p, t, y, y_next, dt):
                return jnp.sum(
                    jax.vmap(
                        lambda p_, y_, y_next_, drift_: self.objective(p_, t, y_, y_next_, dt, drift_, self.dp.diffusion(t, y + offset[0]), self.dp.inverse_diffusion(t, y + offset[0])),
                        in_axes=(1, 1, 1, 1),
                    )(p, y, y_next, self.dp.drift(t, y + offset[0]))
                )

            l = jax.vmap(loss)(ps, jnp.hstack(ts[:, :-1]), jnp.vstack(ys[:, :-1]), jnp.vstack(ys[:, 1:]), jnp.hstack(ts[:, 1:] - ts[:, :-1]))

            return jnp.mean(l)

        return training_step

    @property
    def init_params(self):
        return jnp.ones(100), jnp.ones((100, self.dim, 2)), jnp.zeros(100)
