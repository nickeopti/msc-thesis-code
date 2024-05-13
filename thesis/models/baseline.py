import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

import thesis.lightning
import thesis.processes.process as process

State = train_state.TrainState


def embed_time(t_emb_dim, t):
    scaling: float = 100.0
    max_period: float = 10000.0
    
    """ t shape: (batch,) """
    pe = jnp.empty((len(t), t_emb_dim))
    factor = scaling * jnp.einsum(
        'i,j->ij', 
        t,
        jnp.exp(jnp.arange(0, t_emb_dim, 2) * (-(jnp.log(max_period) / t_emb_dim)))
    )
    pe = pe.at[:, 0::2].set(jnp.sin(factor))
    pe = pe.at[:, 1::2].set(jnp.cos(factor))
    return pe


class Model(thesis.lightning.Module[State]):
    dp: process.Diffusion
    dim: int
    learning_rate: float = 1e-3

    @nn.compact
    def __call__(self, t, y, c):
        # z = y

        # for d in (64, 128, 256, 128, 64, self.dim):
        #     te = embed_time(d * 4, t)
        #     e = nn.Dense(2 * d)(te)

        #     z = nn.Dense(d)(z)
        #     z = z * (1 + e[:, :d]) + e[:, d:]
        #     z = nn.gelu(z)

        # return z

        return nn.Sequential(
            [
                nn.Dense(64),
                nn.gelu,
                nn.Dense(128),
                nn.gelu,
                nn.Dense(256),
                nn.gelu,
                nn.Dense(512),
                nn.gelu,
                nn.Dense(256),
                nn.gelu,
                nn.Dense(128),
                nn.gelu,
                nn.Dense(64),
                nn.gelu,
                nn.Dense(self.dim)
            ]
        )(jnp.hstack((t[:, None], c[:, None], y)))

    def make_training_step(self):
        def training_step(state: State, ts, ys, v, c, offset):
            ps = state.apply_fn(state.params, jnp.hstack(ts[:, 1:]), jnp.vstack(ys[:, 1:]), jnp.hstack(jnp.ones_like(ts[:, 1:]) * c[:, None]))

            def loss(p, t, y, y_next, dt):
                # return jnp.linalg.norm(p + self.dp.inverse_diffusion(t, y) @ (y_next - y - self.dp.drift(t, y) * dt) / dt)**2
                return p.T @ self.dp.diffusion(t, y + offset[0]) * dt @ p + 2 * p.T @ (y_next - y - self.dp.drift(t, y + offset[0]) * dt)

            l = jax.vmap(loss)(ps, jnp.hstack(ts[:, :-1]), jnp.vstack(ys[:, :-1]), jnp.vstack(ys[:, 1:]), jnp.hstack(ts[:, 1:] - ts[:, :-1]))
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
        return self.init(rng, jnp.ones(100), jnp.ones((100, self.dim)), jnp.zeros(100))

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
            )(jnp.hstack((t[:, None], cv[:, None], x))),
            in_axes=2,
            out_axes=2,
        )(y)

        return z # / t[:, None, None]

    def make_training_step(self):
        def training_step(state: State, ts, ys, v, c, offset):
            ps = state.apply_fn(state.params, jnp.hstack(ts[:, 1:]), jnp.vstack(ys[:, 1:]), jnp.hstack(jnp.ones_like(ts[:, 1:]) * c[:, None]))

            def loss(p, t, y, y_next, dt):
                return jnp.sum(
                    jax.vmap(
                        # lambda p, y, y_next: jnp.linalg.norm(p + self.dp.inverse_diffusion(t, y) @ (y_next - y - self.dp.drift(t, y) * dt) / dt)**2,
                        lambda p_, y_, y_next_, d_: p_.T @ self.dp.diffusion(t, y + offset[0]) * dt @ p_ + 2 * p_.T @ (y_next_ - y_ - d_ * dt),
                        in_axes=(1, 1, 1, 1),
                    )(p, y, y_next, self.dp.drift(t, y + offset[0]))
                )

            # l = jax.vmap(loss)(ps, jnp.hstack(ts[:, :-1]), jnp.vstack(ys[:, :-1]), jnp.vstack(ys[:, 1:]), jnp.hstack(ts[:, 1:] - ts[:, :-1]))
            l = jax.vmap(loss)(ps, jnp.hstack(ts[:, :-1]), jnp.vstack(ys[:, :-1]), jnp.vstack(ys[:, 1:]), jnp.hstack(ts[:, 1:] - ts[:, :-1]))
            return jnp.mean(l)
        
        return training_step

    def initialise_params(self, rng):
        return self.init(rng, jnp.ones(100), jnp.ones((100, self.dim, 2)), jnp.zeros(100))
