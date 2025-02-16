import argparse
import contextlib
import sys
from functools import partial

import flax.linen as nn
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
from thesis.inference import likelihood


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    key = jax.random.key(
        selector.get_argument(parser, 'rng_key', type=int, default=0)
    )

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

    from_sigma = selector.get_argument(parser, 'from_sigma', type=float)
    to_sigma = selector.get_argument(parser, 'to_sigma', type=float)
    n_values = selector.get_argument(parser, 'n_values', type=int)

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

        model, state = model_initialiser.func.load_from_checkpoint(
            checkpoint,
            dp=diffusion_process.dp,
            dim=simulator.simulate_sample_path(key, diffusion_process.dp, constraints.initial, t0=0, t1=1, n_steps=2)[1][0].size,
            # dim=experiment[jax.random.key(0)][1][0].shape[0],
            **model_initialiser.keywords,
        )
        
        displacement = selector.get_argument(parser, 'displacement', type=bool)
    
        dp_bar = lambda sigma: likelihood.reverse_time_conditioned_dp(
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
                        c=jnp.array([(jnp.log10(sigma) - from_sigma) / (to_sigma - from_sigma)]),
                    )[0].reshape(y.shape, order='F')
        )

        t0 = 1
        t1 = 0
    else:
        # dp_bar = lambda sigma: likelihood.reverse_time_conditioned_dp(
        #     sigma=sigma,
        #     dp=diffusion_process.dp,
        #     score=lambda dp: lambda t, y: diffusion_process.score_analytical(t, y, dp=dp, constraints=constraints)
        # )
        dp_bar = lambda sigma: likelihood.brownian_bridge_dp(
            sigma=sigma,
            dp=diffusion_process.dp,
            constraints=constraints,
            t1=1
        )
        t0 = 0
        t1 = 1

    M = selector.get_argument(parser, 'n_mc', type=int)
    N = selector.get_argument(parser, 'n_steps', type=int)

    dp = diffusion_process.dp

    estimators = (
        (
            lambda sigma: likelihood.analytical(dp=dp, sigma=sigma, constraints=constraints),
            'analytical'
        ),
        (
            lambda sigma: likelihood.stable_analytical_offset(dp=dp, sigma=sigma, constraints=constraints),
            'stable analytical'
        ),
        (
            lambda sigma: likelihood.simulated(key=key, t1=t1, constraints=constraints, dp=dp, sigma=sigma, simulator=simulator, M=M, N=N, likelihood=likelihood.analytical),
            'simulated'
        ),
        # (
        #     lambda sigma: likelihood.simulated(key=key, t1=t1, constraints=constraints, dp=dp, sigma=sigma, simulator=simulator, M=M, N=N, likelihood=likelihood.stable_analytical_offset),
        #     'stable simulated'
        # ),
        (
            lambda sigma: likelihood.heng(
                key=key,
                t0=t0,
                t1=t1,
                constraints=constraints if t1 > t0 else constraints.reversed(),
                dp=dp,
                sigma=sigma,
                dp_bar=dp_bar(sigma),
                simulator=simulator,
                M=M,
                N=N,
            ),
            'heng'
        ),
        # (
        #     lambda sigma: likelihood.importance_sampled(
        #         key=key,
        #         t0=t0,
        #         t1=t1,
        #         constraints=constraints,
        #         dp=dp,
        #         sigma=sigma,
        #         dp_bar=dp_bar(sigma),
        #         simulator=simulator,
        #         M=M,
        #         N=N,
        #         stable=True,
        #         likelihood=likelihood.analytical,
        #     ),
        #     'stable'
        # ),
        (
            lambda sigma: likelihood.importance_sampled(
                key=key,
                t0=t0,
                t1=t1,
                constraints=constraints if t1 > t0 else constraints.reversed(),
                dp=dp,
                sigma=sigma,
                dp_bar=dp_bar(sigma),
                simulator=simulator,
                M=M,
                N=N,
                stable=True,
                likelihood=likelihood.stable_analytical_offset,
            ),
            'proposed'
        )
    )

    sigmas = jnp.logspace(from_sigma, to_sigma, n_values)

    output_path = selector.get_argument(parser, 'output_path', type=str, default=None)
    with open(output_path, 'w') if output_path else contextlib.nullcontext(sys.stdout) as f:
        if output_path:
            f.write('method,sigma,ll\n')

        for estimator, name in estimators:
            if output_path:
                print(name)
            e = jax.jit(estimator)

            for sigma in sigmas:
                ll = e(sigma)
                if output_path:
                    print(sigma, ll)
                f.write(f'{name},{sigma},{ll}\n')


if __name__ == '__main__':
    main()
