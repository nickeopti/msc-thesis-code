import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from thesis.processes import process


analytical_likelihoods = []

# for covariance in jnp.linspace(0.1, 1.9, 10):
for covariance in (-0.8, -0.5, -0.3, 0, 0.3, 0.5, 0.8):
    dp = process.brownian_motion(
        jnp.array(
            [
                [1, covariance],
                [covariance, 1]
            ]
        )
    )

    def logdet(v):
        r = jnp.linalg.slogdet(v)
        return r.sign * r.logabsdet

    def lla(y, t):
        c = -dp.d / 2 * jnp.log(2 * jnp.pi) - logdet(1 * dp.diffusion) / 2
        return c - (y - 0).T @ dp.inverse_diffusion @ (y - 0) / t / 2

    key = jax.random.PRNGKey(1)

    a_likelihoods = []
    _, *point_subkeys = jax.random.split(key, 110)
    for point_key in point_subkeys:
        # point = jax.random.multivariate_normal(point_key, jnp.zeros(2), jnp.array([[0.8, 0], [0, 0.8]]))
        point = jax.random.multivariate_normal(point_key, jnp.zeros(2), jnp.array([[1, 0.5], [0.5, 1]]))
        print(point)

        llav = lla(point, 1)
        print(llav)
        a_likelihoods.append(llav)

    analytical_likelihoods.append((covariance, sum(a_likelihoods) / len(a_likelihoods)))

plt.figure()
plt.plot(*zip(*analytical_likelihoods), c='tab:orange', label='Analytical likelihood')
plt.legend()
plt.savefig('plots/likelihood_cov.png', dpi=600)
