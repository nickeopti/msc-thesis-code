import argparse
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import selector

import thesis.experiments
import thesis.experiments.constraints
import thesis.experiments.diffusion_processes
import thesis.experiments.experiments
import thesis.experiments.simulators
import thesis.lightning
import thesis.models.baseline
import thesis.processes.process


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

    constraints = selector.add_options_from_module(
        parser,
        'constraints',
        thesis.experiments.constraints,
        thesis.experiments.constraints.Constraints,
    )()
    # constraints.initial = constraints.initial.reshape((-1, 2), order='F')
    # constraints.terminal = constraints.terminal.reshape((-1, 2), order='F')
    diffusion_process: thesis.experiments.diffusion_processes.Brownian = _provide_constraints(
        selector.add_options_from_module(
            parser,
            'diffusion',
            thesis.experiments.diffusion_processes,
            thesis.experiments.diffusion_processes.DiffusionProcess,
        ),
        constraints=constraints,
    )
    assert isinstance(diffusion_process, thesis.experiments.diffusion_processes.Brownian)
    simulator = selector.add_options_from_module(
        parser,
        'simulator',
        thesis.experiments.simulators,
        thesis.experiments.simulators.Simulator,
    )()

    checkpoint = selector.get_argument(parser, 'checkpoint', type=str, default=None)
    if checkpoint is not None:
        model_initialiser = selector.add_options_from_module(
            parser, 'model', thesis.models.baseline, thesis.lightning.Module,
        )
        model, state = model_initialiser.func.load_from_checkpoint(
            checkpoint,
            dp=diffusion_process.dp,
            dim=simulator.simulate_sample_path(key, diffusion_process.dp, constraints.initial, t0=0, t1=1, dt=1)[1][0].shape[0],
            **model_initialiser.keywords,
        )
        displacement = selector.get_argument(parser, 'displacement', type=bool)

    def analytical(sigma, t1, constraints):
        return jax.scipy.stats.multivariate_normal.logpdf(constraints.terminal.reshape(-1, order='F'), constraints.initial.reshape(-1, order='F'), sigma * t1)

    def pedersen(key, sigma, t1, constraints, M, N):
        delta = t1 / N
        var = delta

        dp = thesis.processes.process.Diffusion(
            drift=diffusion_process.dp.drift,
            diffusion=lambda t, y: diffusion_process.dp.diffusion(t, y) * jnp.sqrt(sigma),
            inverse_diffusion=None,
            diffusion_divergence=None,
        )

        def f(key):
            _, ys = simulator.simulate_sample_path(key, dp, constraints.initial, t0=0, t1=t1 * (N-1) / N, dt=delta)
            x = ys[-1]
            return jax.scipy.stats.multivariate_normal.logpdf(constraints.terminal.reshape(-1, order='F'), x.reshape(-1, order='F'), sigma * var)

        return jax.scipy.special.logsumexp(jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)

    def pedersen_brownian_bridge(key, sigma, t1, constraints, M, N):
        delta = t1 / N
        var = delta

        dp_bar = thesis.processes.process.Diffusion(
            drift=lambda t, y: (constraints.terminal - y) / (t1 - t),
            diffusion=lambda t, y: diffusion_process.dp.diffusion(t, y) * jnp.sqrt(sigma),
            inverse_diffusion=None,
            diffusion_divergence=None,
        )

        def f(key):
            ts, ys = simulator.simulate_sample_path(key, dp_bar, constraints.initial, t0=0, t1=t1 * (N-1) / N, dt=delta)
            ts = jnp.hstack((0, ts))
            ys = jnp.vstack((constraints.initial[None], ys))
            
            w = jnp.exp(
                jnp.sum(
                    jax.vmap(
                        lambda y, y_next: jax.scipy.stats.multivariate_normal.logpdf(y_next.reshape(-1, order='F'), y.reshape(-1, order='F'), sigma * var),
                        in_axes=(0, 0)
                    )(ys[:-1], ys[1:])
                )
                -
                jnp.sum(
                    jax.vmap(
                        lambda t, y, y_next: jax.scipy.stats.multivariate_normal.logpdf(y_next.reshape(-1, order='F'), (y + delta * dp_bar.drift(t, y)).reshape(-1, order='F'), sigma * var),
                        in_axes=(0, 0, 0)
                    )(ts[:-1], ys[:-1], ys[1:])
                )
            )
            x = ys[-1]
            return {
                'a': jax.scipy.stats.multivariate_normal.logpdf(constraints.terminal.reshape(-1, order='F'), x.reshape(-1, order='F'), sigma * var).squeeze(),
                'b': w
            }
        
        return jax.scipy.special.logsumexp(**jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)

    def pedersen_brownian_bridge_reverse(key, sigma, t1, constraints, M, N, analytical: bool = True):
        delta = t1 / N
        var = delta

        dp = thesis.processes.process.Diffusion(
            drift=diffusion_process.dp.drift,
            diffusion=lambda t, y: diffusion_process.dp.diffusion(t, y) * jnp.sqrt(sigma),
            inverse_diffusion=lambda t, y: jnp.linalg.inv(diffusion_process.dp.diffusion(t, y) * jnp.sqrt(sigma)),
            diffusion_divergence=diffusion_process.dp.diffusion_divergence,
        )

        dp_bar = thesis.processes.process.Diffusion(
            drift=thesis.experiments.experiments._f_bar(
                dp=dp,
                score=(
                    partial(diffusion_process.score_analytical, dp=dp, constraints=constraints)
                    if analytical else
                    lambda t, y:
                        diffusion_process.score_learned(
                            t[None],
                            (y - (constraints.initial.reshape(y.shape, order='F') if displacement else 0))[None],
                            state=state,
                            c=sigma
                        )[0]
                ),
            ),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence
        )

        def f(key):
            ts, ys = simulator.simulate_sample_path(key, dp_bar, constraints.terminal, t0=t1, t1=delta, dt=-delta)
            ts = jnp.hstack((t1, ts))
            ys = jnp.vstack((constraints.terminal[None], ys))
            
            w = jnp.exp(
                jnp.sum(
                    jax.vmap(
                        lambda y, y_next: jax.scipy.stats.multivariate_normal.logpdf(y_next.reshape(-1, order='F'), y.reshape(-1, order='F'), sigma * var),
                        in_axes=(0, 0)
                    )(ys[:-1], ys[1:])
                )
                -
                jnp.sum(
                    jax.vmap(
                        lambda t, y, y_next: jax.scipy.stats.multivariate_normal.logpdf(y_next.reshape(-1, order='F'), (y - delta * dp_bar.drift(t, y)).reshape(-1, order='F'), sigma * var),
                        in_axes=(0, 0, 0)
                    )(ts[:-1], ys[:-1], ys[1:])
                )
            )
            x = ys[-1]
            return {
                'a': jax.scipy.stats.multivariate_normal.logpdf(constraints.initial.reshape(-1, order='F'), x.reshape(-1, order='F'), sigma * var).squeeze(),
                'b': w
            }

        return jax.scipy.special.logsumexp(**jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)

    n_mc = 10_000

    # sigmas = 10**jnp.linspace(-3.5, -2.0, 20)
    # sigmas = jnp.linspace(0.05, 1, 20)
    sigmas = jnp.linspace(0.5, 2, 20)

    lls = (jax.vmap(
        partial(
            analytical,
            t1=1,
            constraints=constraints,
        )
    ))(sigmas)

    plt.figure()
    plt.plot(sigmas, lls)
    plt.savefig('sigma_ll_analytical.png')

    lls = (jax.vmap(
        partial(
            pedersen,
            t1=1,
            constraints=constraints,
            M=n_mc,
            N=25,
        )
    ))(jax.random.split(key, 20), sigmas)

    plt.figure()
    plt.plot(sigmas, lls)
    plt.savefig('sigma_ll_pedersen.png')

    lls = (jax.vmap(
        partial(
            pedersen_brownian_bridge,
            t1=1,
            constraints=constraints,
            M=n_mc,
            N=25,
        )
    ))(jax.random.split(key, 20), sigmas)

    plt.figure()
    plt.plot(sigmas, lls)
    plt.savefig('sigma_ll_pedersen_bridge.png')

    lls = (jax.vmap(
        partial(
            pedersen_brownian_bridge_reverse,
            t1=1,
            constraints=constraints,
            M=n_mc,
            N=25,
            analytical=True,
        )
    ))(jax.random.split(key, 20), sigmas)

    plt.figure()
    plt.plot(sigmas, lls)
    plt.savefig('sigma_ll_pedersen_bridge_reverse.png')

    if checkpoint is not None:
        lls = (jax.vmap(
            partial(
                pedersen_brownian_bridge_reverse,
                t1=1,
                constraints=constraints,
                M=n_mc,
                N=25,
                analytical=False,
            )
        ))(jax.random.split(key, 20), sigmas)

        plt.figure()
        plt.plot(sigmas, lls)
        plt.savefig('sigma_ll_pedersen_bridge_reverse_learned.png')


if __name__ == '__main__':
    main()
