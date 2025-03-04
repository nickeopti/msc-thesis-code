import argparse
import pathlib
from functools import partial

import jax
import jax.numpy as jnp
import selector
import selector.arguments
from flax import linen as nn

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
from thesis.lightning import loggers, trainer

selector.arguments.CONVERTER[jax.Array] = partial(jnp.fromstring, sep=',')


def _provide_constraints(diffusion_process: partial, constraints: thesis.experiments.constraints.Constraints) -> thesis.experiments.diffusion_processes.DiffusionProcess:
    if 'constraints' in diffusion_process.func.__init__.__code__.co_varnames[:diffusion_process.func.__init__.__code__.co_argcount]:
        return diffusion_process(constraints=constraints)
    else:
        return diffusion_process()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    key = jax.random.key(
        selector.get_argument(parser, 'rng_key', type=int, default=0)
    )
    key, subkey = jax.random.split(key)

    constraints = selector.add_options_from_module(
        parser,
        'constraints',
        thesis.experiments.constraints,
        thesis.experiments.constraints.Constraints,
    )()
    diffusion_process = _provide_constraints(
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
    experiment = selector.add_arguments(
        parser,
        'experiment',
        thesis.experiments.experiments.Experiment
    )(
        key=subkey,
        constraints=constraints,
        diffusion_process=diffusion_process,
        simulator=simulator,
    )

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

    checkpoint = selector.get_argument(parser, 'checkpoint', type=str, default=None)
    if checkpoint:
        model, state = model_initialiser.func.load_from_checkpoint(
            checkpoint,
            dp=experiment.dp,
            dim=experiment[jax.random.key(0)][1][0].shape[0],
            **model_initialiser.keywords,
        )

        path = pathlib.Path(checkpoint).parent
    else:
        key, subkey = jax.random.split(key)

        model = model_initialiser(
            dp=experiment.dp,
            dim=experiment[jax.random.key(0)][1][0].shape[0],
        )

        logger = loggers.CSVLogger(
            name=f'{experiment.constraints.__class__.__name__}_{experiment.diffusion_process.__class__.__name__}'
        )

        def callback_function(epoch: int, state):
            plots_path = logger.path / 'plots' / f'epoch_{epoch}'
            plots_path.mkdir(parents=True, exist_ok=True)

            experiment.visualise(state, plots_path)

        t = selector.add_arguments(parser, 'trainer', trainer.Trainer)(logger=logger)
        state = t.fit(
            subkey,
            model,
            experiment,
            # experiment,
            callback=(5000, callback_function),
        )

        path = logger.path

    plots_path = path / 'plots'
    plots_path.mkdir(parents=True, exist_ok=True)

    experiment.visualise(state, plots_path)


if __name__ == '__main__':
    main()
