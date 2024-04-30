import argparse
import pathlib
from functools import partial

import jax
import selector

import thesis.experiments
import thesis.experiments.constraints
import thesis.experiments.diffusion_processes
import thesis.experiments.experiments
import thesis.experiments.simulators
import thesis.lightning
import thesis.models.baseline
import thesis.processes.process
from thesis.lightning import loggers, trainer


def _provide_constraints(diffusion_process: partial, constraints: thesis.experiments.constraints.Constraints):
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

    checkpoint = selector.get_argument(parser, 'checkpoint', type=str, default=None)

    model_initialiser = selector.add_options_from_module(
        parser, 'model', thesis.models.baseline, thesis.lightning.Module,
    )

    if checkpoint:
        model, state = model_initialiser.func.load_from_checkpoint(
            checkpoint,
            dp=experiment.diffusion_process.dp,
            dim=experiment[0][1][0].shape[0],
            **model_initialiser.keywords,
        )

        path = pathlib.Path(checkpoint).parent
    else:
        key, subkey = jax.random.split(key)

        model = model_initialiser(
            dp=experiment.diffusion_process.dp,
            dim=experiment[0][1][0].shape[0],
        )

        logger = loggers.CSVLogger(
            name=f'{experiment.constraints.__class__.__name__}_{experiment.diffusion_process.__class__.__name__}'
        )

        t = selector.add_arguments(parser, 'trainer', trainer.Trainer)(logger=logger)
        state = t.fit(
            subkey,
            model,
            experiment,
            # experiment,
        )

        path = logger.path

    plots_path = path / 'plots'
    plots_path.mkdir(parents=True, exist_ok=True)

    experiment.visualise(state, plots_path)


if __name__ == '__main__':
    main()




# specify diffusion process

# obtain data set(s), based on the diffusion process

# get model (either train or load checkpoint)

# create visualisations

# possibly compute likelihood




# wiener process is almost surely non-differentiable, and thus taking infinitely small time steps results in something that explodes; hence problems in the disretised paths...

# correlate losses by taking ratio of loss of sequential batches, and learn by the gradient of that

# test whether the likelihood estimates truly are step-size dependent -- convince Stefan of its wrongness

# Idea: all jumps/steps are independent (Markov property); so consider the likelihood of seeing all observations jointly, assuming original covariance structure:
    # this is not thought trough... But it seems unlikely that all steps will be in the same direction, for instance, assuming a scalar * identity covariance matrix.
    # So consider the correlation between samples, in some sense.

# Note: There is a difference between taking the sum of the log-likelihoods of each step,
# and taking the log-likelihood of the sum of all the steps (consider the steps as vectors; their sum is the target).
# The former is what would be derived from classic high-frequency SDE analysis, and is also what has been done here.
# The latter _may_ be of interest; it shall certainly be tested! I'm pretty sure it will work in the simple Brownian case,
# because everything is already Gaussian; the big question is whether it works in general.
