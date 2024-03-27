import jax
import jax.numpy as jnp

import visualise


visualise.visualise_vector_field(lambda t, x: -x / t[:, None], 'plots/score_field_single_center_id.png', a=-1, b=1, val=6, nv=61)
c = jnp.array([[1, 0.3], [0.3, 1]])
visualise.visualise_vector_field(jax.vmap(lambda t, x: - jnp.linalg.inv(c) @ x / t), 'plots/score_field_single_center_cov_0_3.png', a=-1, b=1, val=6, nv=61)

y0_1 = -jnp.ones(2) * 1.5
y0_2 = jnp.ones(2) * 1.5

for c in (-0.3, 0, 0.3):
    covariance = c
    covariance_matrix = jnp.array(
        [
            [1, covariance],
            [covariance, 1]
        ]
    )

    a = lambda t, y, y0: jnp.exp(-(y - y0).T @ jnp.linalg.inv(t * covariance_matrix) @ (y - y0))
    s = lambda t, y: -1 / (a(t, y, y0_1) + a(t, y, y0_2)) * (jnp.linalg.inv(t * covariance_matrix) @ (y - y0_1) * a(t, y, y0_1) + jnp.linalg.inv(t * covariance_matrix) @ (y - y0_2) * a(t, y, y0_2))

    visualise.visualise_vector_field(
        score=jax.vmap(s),
        filename=f'plots/analytical_score_vector_field_cov_{str(c).replace('.', '_')}.png',
        val=6, nv=61
    )

