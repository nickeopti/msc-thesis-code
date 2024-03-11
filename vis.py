import sys
import pathlib

import jax.numpy as jnp

import score
import process


covariance = -0.3
covariance_matrix = jnp.array(
    [
        [1, covariance],
        [covariance, 1]
    ]
)
dp = process.brownian_motion(covariance_matrix)

model, state = score.Model.load_from_checkpoint(
    sys.argv[1],
    dp=dp,
)

model.on_fit_end(state, pathlib.Path('plots'), c=covariance)
