from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp

import thesis.processes.diffusion as diffusion
import thesis.processes.process as process


@partial(jax.jit, static_argnames=['n_steps'])
def likelihood(key, n_steps: int, covariance: jax.Array, true_covariance: jax.Array):
    dp = process.brownian_motion(covariance)
    y0 = jnp.zeros(dp.d)

    t0 = 1.0
    t1 = 0.001
    dt = (t1 - t0) / n_steps

    def f_bar_analytical(t, y):
        s = -dp.inverse_diffusion @ (y - y0) / t
        return dp.drift - dp.diffusion @ s - dp.diffusion_divergence
    
    dp_bar = process.Diffusion(
        d=dp.d,
        drift=f_bar_analytical,
        diffusion=dp.diffusion,
        inverse_diffusion=dp.inverse_diffusion,
        diffusion_divergence=dp.diffusion_divergence,
    )

    def logdet(v):
        r = jnp.linalg.slogdet(v)
        return r.sign * r.logabsdet

    def ll(y, y_next, dt):
        c = -dp.d / 2 * jnp.log(2 * jnp.pi) - logdet(dt * dp.diffusion) / 2
        return c - (y_next - y).T @ dp.inverse_diffusion @ (y_next - y) / dt / 2
    
    def compute(point, key):
        ts, ys, n = diffusion.get_paths(
            dp=dp_bar,
            key=key,
            y0=point,
            t0=t0,
            t1=t1,
            dt=dt,
        )

        # print(n, n_steps)
        # print(lax.dynamic_slice(jnp.vstack((point, ys)), (0, 0), (n_steps, dp.d)))
        # print(lax.dynamic_slice(jnp.vstack((point, ys)), (1, 0), (n_steps, dp.d)))
        # print(lax.dynamic_slice(jnp.hstack((t0, ts)), (1, ), (n_steps, )).shape)
        # print(lax.dynamic_slice(jnp.hstack((t0, ts)), (0, ), (n_steps, )).shape)
        # print(lax.dynamic_slice(jnp.hstack((t0, ts)), (1, ), (n_steps, )) - lax.dynamic_slice(jnp.hstack((t0, ts)), (0, ), (n_steps, )))
        # print(jnp.hstack((t0, ts)))
        # print(lax.dynamic_slice(jnp.hstack((t0, ts)), (0, ), (n_steps, )))
        # print(lax.dynamic_slice(jnp.hstack((t0, ts)), (1, ), (n_steps, )))
        # print(
        #     jnp.abs(
        #         lax.dynamic_slice(jnp.hstack((t0, ts)), (0, ), (n_steps, )) -
        #         # ts[1:n_steps] -
        #         lax.dynamic_slice(jnp.hstack((t0, ts)), (1, ), (n_steps, ))
        #         # ts[:n_steps-1]
        #     )
        # )

        return jnp.sum(
            jax.vmap(ll)(
                lax.dynamic_slice(jnp.vstack((point, ys)), (1, 0), (n_steps, dp.d)),
                # ys[:n_steps-1],
                lax.dynamic_slice(jnp.vstack((point, ys)), (0, 0), (n_steps, dp.d)),
                # ys[1:n_steps],
                jnp.abs(
                    lax.dynamic_slice(jnp.hstack((t0, ts)), (0, ), (n_steps, )) -
                    # ts[1:n_steps] -
                    lax.dynamic_slice(jnp.hstack((t0, ts)), (1, ), (n_steps, ))
                    # ts[:n_steps-1]
                )
            )
        )
    
    def f(key):
        point = jax.random.multivariate_normal(key, y0, true_covariance)

        path_subkeys = jax.random.split(key, 10)
        return jnp.sum(jax.vmap(compute, in_axes=(None, 0), out_axes=0)(point, path_subkeys))
    
    point_subkeys = jax.random.split(key, 10)
    return jnp.sum(jax.vmap(f)(point_subkeys)) / 100


def main_variance():
    key = jax.random.PRNGKey(0)

    with open('lls_step.csv', 'w') as f:
        f.write('n_steps,variance,loglikelihood\n')

    true_covariance = jnp.array([[0.1, 0], [0, 0.1]])

    for n_steps in (1, 2, 5, 10, 20, 50, 100, 500, 1000):
        for variance in 10**jnp.linspace(-3, 1, 50):
            covariance = jnp.array(
                [
                    [variance, 0],
                    [0, variance]
                ]
            )

            value = likelihood(key, n_steps, covariance, true_covariance)

            with open('lls_step.csv', 'a') as f:
                f.write(f'{n_steps},{variance},{value}\n')


def main_covariance():
    key = jax.random.PRNGKey(0)

    with open('lls_step_c.csv', 'w') as f:
        f.write('n_steps,covariance,loglikelihood\n')

    true_covariance = jnp.array([[1, 0], [0, 1]])

    for n_steps in (1, 2, 5, 10, 20, 50, 100, 500, 1000):
        for c in jnp.linspace(-0.8, 0.8, 20):
            covariance = jnp.array(
                [
                    [1, c],
                    [c, 1]
                ]
            )

            value = likelihood(key, n_steps, covariance, true_covariance)

            with open('lls_step_c.csv', 'a') as f:
                f.write(f'{n_steps},{c},{value}\n')


if __name__ == '__main__':
    main_covariance()
