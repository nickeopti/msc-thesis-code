import argparse
import os.path
import pathlib
from functools import partial

import cycler
import flax.linen as nn
import jax
import jax.dtypes
import jax.lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
from thesis.experiments.constraints import (ConstraintsCollection,
                                            LandmarksConstraints)
from thesis.experiments.simulators import Simulator
from thesis.inference import likelihood

plt.rc('axes', prop_cycle=cycler.cycler(color=plt.colormaps.get_cmap('tab20').colors))


def estimate_diffusion_mean(
    key,
    constraints: ConstraintsCollection,
    simulator: Simulator,
    ll_to_optimise,
    n: int = 100,
    learning_rate: float = 0.0001,
):
    @jax.jit
    @partial(jax.grad, argnums=1)
    def f(key, initial):
        v = jnp.sum(jax.vmap(lambda terminal: ll_to_optimise(key, initial, terminal))(constraints.initials))
        # jax.debug.print('{v}', v=v)
        return v

    mean_shape = jnp.mean(constraints.initials, axis=0)
    # diffusion_mean_estimate = (constraints.initial + constraints.terminal) / 2
    # diffusion_mean_estimate = constraints.terminal
    # diffusion_mean_estimate = constraints.initials[2]
    diffusion_mean_estimate = jnp.array([[-2, 1]], dtype=jnp.float64)

    subkeys = jax.random.split(key, n)

    estimates = [diffusion_mean_estimate]
    for i in range(n):
        plt.figure()
        for k in range(len(diffusion_mean_estimate)):
            plt.scatter(diffusion_mean_estimate[k, 0], diffusion_mean_estimate[k, 1], color=f'C{2*k}')
        # if hide_axes:
        plt.axis('off')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.savefig(os.path.join('mean_plots', f'estimage_{i:03}.png'), dpi=100)
        plt.close()

        lg = f(subkeys[i], diffusion_mean_estimate)
        # print(f'{lg=}')
        diffusion_mean_estimate += learning_rate * lg
        print(i, jnp.sqrt(jnp.mean((diffusion_mean_estimate - mean_shape)**2)))

        estimates.append(diffusion_mean_estimate)

    return diffusion_mean_estimate, jnp.array(estimates)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    key = jax.random.key(
        selector.get_argument(parser, 'rng_key', type=int, default=0)
    )
    # key, subkey = jax.random.split(key)

    constraints = selector.add_options_from_module(
        parser,
        'constraints',
        thesis.experiments.constraints,
        thesis.experiments.constraints.Constraints,
    )()
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

    sigma = selector.get_argument(parser, 'sigma', type=float)

    checkpoint = selector.get_argument(parser, 'checkpoint', type=str, default=None)
    if checkpoint:
        match selector.get_argument(parser, 'activation', type=str, default='gelu'):
            case 'relu':
                activation = nn.relu
            case 'leaky_relu':
                activation = nn.leaky_relu
            case 'elu':
                activation = nn.elu
            case 'tanh':
                activation = nn.tanh
            case 'gelu':
                activation = nn.gelu
            case other:
                raise ValueError(f'Unknown activation function {other!r} asked for')

        network = selector.add_options_from_module(
            parser, 'network', thesis.models.networks, thesis.models.networks.Network,
        )
        objective = selector.add_options_from_module(
            parser, 'objective', thesis.models.objectives, thesis.models.objectives.Objective,
        )
        model_initialiser = selector.add_options_from_module(
            parser, 'model', thesis.models.models, thesis.lightning.Module,
        )
        model_initialiser = partial(model_initialiser, network=partial(network, activation=activation), objective=objective())

        # checkpoint = selector.get_argument(parser, 'checkpoint', type=str)
        model, state = model_initialiser.func.load_from_checkpoint(
            checkpoint,
            dp=diffusion_process.dp,
            dim=simulator.simulate_sample_path(key, diffusion_process.dp, constraints.initial, t0=0, t1=1, n_steps=2)[1][0].size,
            # dim=experiment[jax.random.key(0)][1][0].shape[0],
            **model_initialiser.keywords,
        )
        
        displacement = selector.get_argument(parser, 'displacement', type=bool)

        dp_bar = likelihood.reverse_time_conditioned_dp(
            sigma,
            diffusion_process.dp,
            lambda _:
                lambda t, y:
                    diffusion_process.score_learned(
                        t[None],
                        (y.reshape(-1, order='F') - (constraints.initial.reshape(-1, order='F') if displacement else 0))[None],
                        state=state,
                        # c=jnp.ones_like(t[None]) * 0.5,
                        # c=diffusion_process.c,
                        c=jnp.array([7.5e-2]),
                    )[0].reshape(y.shape, order='F')
        )
    else:
        dp_bar = likelihood.reverse_time_conditioned_dp(
            sigma=sigma,
            dp=diffusion_process.dp,
            score=lambda dp: lambda t, y: diffusion_process.score_analytical(t, y, dp=dp, constraints=constraints)
        )

    # constraints = thesis.experiments.constraints.ButterflyLandmarksCollection(
    #     data_path='../data/butterflies/aligned_dataset_n43/',
    #     species='Papilio polytes, Papilio ambrax, Papilio slateri, Papilio protenor, Papilio deiphobus, Papilio polyxenes',
    #     every=6,
    # )
    # # diffusion_process = thesis.experiments.diffusion_processes.BrownianWideKernel(1, 0.1, constraints)
    # diffusion_process = thesis.experiments.diffusion_processes.KunitaWide(1, 0.1, constraints)
    # dp = diffusion_process.dp
    # simulator = thesis.experiments.simulators.AutoLongSimulator()

    # key = jax.random.key(42)

    likelihood_estimator = lambda key, initial, terminal: likelihood.importance_sampled(
            key=key,
            t0=1,
            t1=0,
            constraints=LandmarksConstraints(initial, terminal),
            dp=diffusion_process.dp,
            sigma=sigma,
            # dp_bar=likelihood.brownian_bridge_dp(sigma, diffusion_process.dp, constraints, 1),
            dp_bar=dp_bar,
            simulator=simulator,
            M=100,
            N=100,
            stable=True,
            likelihood=likelihood.stable_analytical_offset,
        )

    n_steps = selector.get_argument(parser, 'n_steps', type=int, default=100)
    update_rate = selector.get_argument(parser, 'update_rate', type=float, default=0.00001)
    
    estimate, estimates = estimate_diffusion_mean(
        key=key,
        constraints=constraints,
        simulator=simulator,
        ll_to_optimise=likelihood_estimator,
        n=n_steps,
        learning_rate=update_rate,
    )

    with open('diffusion_mean_estimates.npy', 'wb') as f:
        jnp.save(f, estimates)


if __name__ == '__main__':
    main()
