import pathlib
import sys

import jax.numpy as jnp

import thesis.models.baseline
import thesis.processes.process as process

covariance = -0.3
covariance_matrix = jnp.array(
    [
        [1, covariance],
        [covariance, 1]
    ]
)
dp = process.brownian_motion(covariance_matrix)

model, state = thesis.models.baseline.Model.load_from_checkpoint(
    sys.argv[1],
    dp=dp,
)

model.on_fit_end(state, pathlib.Path('plots'), c=covariance)
