import argparse

import jax
import jax.dtypes
import jax.lax
import jax.numpy as jnp
import selector
import selector.arguments

import thesis.experiments
import thesis.experiments.constraints
import thesis.experiments.diffusion_processes
import thesis.experiments.experiments
import thesis.experiments.simulators
import thesis.lightning
import thesis.models.models
import thesis.models.networks
import thesis.models.objectives
import thesis.processes.process
import thesis.scripts
import thesis.scripts.common
from thesis.experiments.constraints import LandmarksConstraints


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    key = jax.random.key(
        selector.get_argument(parser, 'rng_key', type=int, default=0)
    )

    constraints = LandmarksConstraints(jnp.array([[0, 0]], dtype=jnp.float32), jnp.array([[1, 1]], dtype=jnp.float32))

    diffusion_process = thesis.scripts.common._provide_constraints(
        selector.add_options_from_module(
            parser,
            'diffusion',
            thesis.experiments.diffusion_processes,
            thesis.experiments.diffusion_processes.DiffusionProcess,
        ),
        constraints=constraints,
    )
    simulator = selector.add_options_from_module(
        parser,
        'simulator',
        thesis.experiments.simulators,
        thesis.experiments.simulators.Simulator,
    )()

    n_points = selector.get_argument(parser, 'n_points', type=int)

    keys = jax.random.split(key, n_points)

    paths = jnp.array(
        [
            simulator.simulate_sample_path(key, diffusion_process.dp, constraints.initial, t0=0, t1=1, n_steps=1000)[1]
            for key in keys
        ]
    )
    points = paths[:, -1]

    with open('brownian_2d_points_collection.npy', 'wb') as f:
        jnp.save(f, paths)

    with open('brownian_2d_points_collection.txt', 'w') as f:
        f.write(','.join(map(str, points.reshape(-1, order='F'))))

    print(','.join(map(str, points.reshape(-1, order='F'))))


if __name__ == '__main__':
    main()
