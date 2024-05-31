import argparse
import os
import os.path
from functools import partial
from typing import Callable

import jax
import jax.dtypes
import jax.numpy as jnp
import selector
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
from thesis.experiments.constraints import Constraints
from thesis.experiments.simulators import Simulator
from thesis.processes.process import Diffusion
from thesis.scripts.common import _provide_constraints

jax.config.update("jax_enable_x64", True)

print(jnp.zeros(10).dtype)


def unconditioned_dp(sigma: float, dp: Diffusion) -> Diffusion:
    return Diffusion(
        drift=dp.drift,
        diffusion=lambda t, y: dp.diffusion(t, y) * jnp.sqrt(sigma),
        inverse_diffusion=lambda t, y: jnp.linalg.inv(dp.diffusion(t, y) * jnp.sqrt(sigma)),
        diffusion_divergence=lambda t, y: dp.diffusion_divergence(t, y) * jnp.sqrt(sigma),
    )


def conditioned_dp_forwards(sigma: float, dp: Diffusion, constraints: Constraints, t1: float) -> Diffusion:
    udp = unconditioned_dp(sigma, dp)

    return Diffusion(
        drift=lambda t, y: (constraints.terminal.reshape(y.shape, order='F') - y) / (t1 - t),
        diffusion=udp.diffusion,
        inverse_diffusion=udp.inverse_diffusion,
        diffusion_divergence=udp.diffusion_divergence,
    )


def conditioned_dp_backwards(sigma: float, dp: Diffusion, score: Callable[[Diffusion], Callable[[jax.Array, jax.Array], jax.Array]]) -> Diffusion:
    udp = unconditioned_dp(sigma, dp)

    return Diffusion(
        drift=thesis.experiments.experiments._f_bar(dp=udp, score=score(udp)),
        diffusion=udp.diffusion,
        inverse_diffusion=udp.inverse_diffusion,
        diffusion_divergence=udp.diffusion_divergence
    )


def analytical(sigma: float, t1: float, constraints: Constraints):
    return jax.scipy.stats.multivariate_normal.logpdf(constraints.terminal.reshape(-1, order='F'), constraints.initial.reshape(-1, order='F'), sigma * t1 * jnp.eye(constraints.initial.size))


def pedersen(key: jax.dtypes.prng_key, t1: float, constraints: Constraints, dp: Diffusion, simulator: Simulator, M: int, N: int):
    delta = t1 / N
    var = delta

    def f(key):
        ts, ys = simulator.simulate_sample_path(key, dp, constraints.initial, t0=0, t1=t1 - delta, n_steps=N)
        d = ys[0].shape[1] if len(ys[0].shape) > 1 else 1
        return jax.scipy.stats.multivariate_normal.logpdf(constraints.terminal.reshape(-1, order='F'), ys[-1].reshape(-1, order='F'), thesis.processes.process.long_diffusion(dp.diffusion(ts[-1], ys[-1]) @ dp.diffusion(ts[-1], ys[-1]).T, d) * var)

    return jax.scipy.special.logsumexp(jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)


def pedersen_brownian_bridge(key: jax.dtypes.prng_key, t1: float, constraints: Constraints, dp: Diffusion, dp_bar: Diffusion, simulator: Simulator, M: int, N: int):
    delta = t1 / N
    var = delta

    def f(key):
        ts, ys = simulator.simulate_sample_path(key, dp_bar, constraints.initial, t0=0, t1=t1 - delta, n_steps=N)
        ts = jnp.hstack((0, ts))
        ys = jnp.vstack((constraints.initial.reshape(ys[0].shape, order='F')[None], ys))

        d = ys[0].shape[1] if len(ys[0].shape) > 1 else 1

        w = jnp.exp(
            jnp.sum(
                jax.vmap(
                    lambda t, y, y_next: -(y_next.reshape(-1, order='F') - y.reshape(-1, order='F')).T @ jnp.linalg.inv(thesis.processes.process.long_diffusion(dp.diffusion(t, y) @ dp.diffusion(t, y).T, d) * var) @ (y_next.reshape(-1, order='F') - y.reshape(-1, order='F')) / 2,
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
            -
            jnp.sum(
                jax.vmap(
                    lambda t, y, y_next: -(y_next.reshape(-1, order='F') - (y + delta * dp_bar.drift(t, y)).reshape(-1, order='F')).T @ jnp.linalg.inv(thesis.processes.process.long_diffusion(dp_bar.diffusion(t, y) @ dp_bar.diffusion(t, y).T, d) * var) @ (y_next.reshape(-1, order='F') - (y + delta * dp_bar.drift(t, y)).reshape(-1, order='F')) / 2,
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
        )
        x = ys[-1]

        return {
            'a': jax.scipy.stats.multivariate_normal.logpdf(constraints.terminal.reshape(-1, order='F'), x.reshape(-1, order='F'), thesis.processes.process.long_diffusion(dp.diffusion(ts[-1], ys[-1]) @ dp.diffusion(ts[-1], ys[-1]).T, d) * var).squeeze(),
            'b': w
        }
    
    return jax.scipy.special.logsumexp(**jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)


def pedersen_brownian_bridge_reverse(key: jax.dtypes.prng_key, t1: float, constraints: Constraints, dp: Diffusion, dp_bar: Diffusion, simulator: Simulator, M: int, N: int, use_every: int):
    delta = t1 / N
    var = delta

    def f(key):
        ts, ys = simulator.simulate_sample_path(key, dp_bar, constraints.terminal, t0=t1, t1=delta, n_steps=N)
        ts = jnp.hstack((t1, ts))
        ys = jnp.vstack((constraints.terminal.reshape(ys[0].shape, order='F')[None], ys))

        d = ys[0].shape[1] if len(ys[0].shape) > 1 else 1

        # print(dp.diffusion(ts[0], ys[0])[::use_every, ::use_every].shape)
        # print(ys[0].reshape(-1, order='F')[::use_every].shape)

        # jax.debug.print(
        #     'd: {v}',
        #     v=jnp.linalg.inv(thesis.processes.process.long_diffusion(dp.diffusion(ts[5], ys[5][::use_every]) @ dp.diffusion(ts[5], ys[5][::use_every]).T, d) * var).shape
        # )
        # jax.debug.print(
        #     'v: {v}',
        #     v=(ys[6].reshape(-1, order='F') - ys[5].reshape(-1, order='F'))[::use_every].T
        # )
        # jax.debug.print(
        #     'vTd: {v}',
        #     v=((ys[6].reshape(-1, order='F') - ys[5].reshape(-1, order='F'))[::use_every].T @ jnp.linalg.inv(thesis.processes.process.long_diffusion(dp.diffusion(ts[5], ys[5][::use_every]) @ dp.diffusion(ts[5], ys[5][::use_every]).T, d) * var))
        # )
        # jax.debug.print(
        #     'vTdv: {v}',
        #     v=((ys[6].reshape(-1, order='F') - ys[5].reshape(-1, order='F'))[::use_every].T @ jnp.linalg.inv(thesis.processes.process.long_diffusion(dp.diffusion(ts[5], ys[5][::use_every]) @ dp.diffusion(ts[5], ys[5][::use_every]).T, d) * var) @ (ys[6].reshape(-1, order='F') - ys[5].reshape(-1, order='F'))[::use_every])
        # )
        # jax.debug.print(
        #     'vTdv2: {v}',
        #     v=((ys[6].reshape(-1, order='F') - ys[5].reshape(-1, order='F'))[::use_every].reshape((1, -1)) @ jnp.linalg.inv(thesis.processes.process.long_diffusion(dp.diffusion(ts[5], ys[5][::use_every]) @ dp.diffusion(ts[5], ys[5][::use_every]).T, d) * var) @ (ys[6].reshape(-1, order='F') - ys[5].reshape(-1, order='F'))[::use_every].reshape((-1, 1)))
        # )

        # jax.debug.print('sigma: {v}', v=dp.diffusion(ts[0], ys[0])[0, 0])
        # jax.debug.print(
        #     'f: {v}',
        #     v=jax.vmap(
        #         lambda t, y, y_next: -(y_next.reshape(-1, order='F') - y.reshape(-1, order='F'))[::use_every].T @ jnp.linalg.inv(thesis.processes.process.long_diffusion(dp.diffusion(t, y[::use_every]) @ dp.diffusion(t, y[::use_every]).T, d) * var) @ (y_next.reshape(-1, order='F') - y.reshape(-1, order='F'))[::use_every] / 2,
        #         in_axes=(0, 0, 0)
        #     )(ts[:-1], ys[:-1], ys[1:])
        # )
        # jax.debug.print(
        #     's: {v}',
        #     v=jax.vmap(
        #         lambda t, y, y_next: -(y_next.reshape(-1, order='F') - (y - delta * dp_bar.drift(t, y)).reshape(-1, order='F'))[::use_every].T @ jnp.linalg.inv(thesis.processes.process.long_diffusion(dp_bar.diffusion(t, y[::use_every]) @ dp_bar.diffusion(t, y[::use_every]).T, d) * var) @ (y_next.reshape(-1, order='F') - (y - delta * dp_bar.drift(t, y)).reshape(-1, order='F'))[::use_every] / 2,
        #         in_axes=(0, 0, 0)
        #     )(ts[:-1], ys[:-1], ys[1:])
        # )
        # jax.debug.print(
        #     'diff: {v}',
        #     v=(
        #         jax.vmap(
        #             lambda t, y, y_next: -(y_next.reshape(-1, order='F') - y.reshape(-1, order='F'))[::use_every].T @ jnp.linalg.inv(thesis.processes.process.long_diffusion(dp.diffusion(t, y[::use_every]) @ dp.diffusion(t, y[::use_every]).T, d) * var) @ (y_next.reshape(-1, order='F') - y.reshape(-1, order='F'))[::use_every] / 2,
        #             in_axes=(0, 0, 0)
        #         )(ts[:-1], ys[:-1], ys[1:])
        #     )
        #     -
        #     (
        #         jax.vmap(
        #             lambda t, y, y_next: -(y_next.reshape(-1, order='F') - (y - delta * dp_bar.drift(t, y)).reshape(-1, order='F'))[::use_every].T @ jnp.linalg.inv(thesis.processes.process.long_diffusion(dp_bar.diffusion(t, y[::use_every]) @ dp_bar.diffusion(t, y[::use_every]).T, d) * var) @ (y_next.reshape(-1, order='F') - (y - delta * dp_bar.drift(t, y)).reshape(-1, order='F'))[::use_every] / 2,
        #             in_axes=(0, 0, 0)
        #         )(ts[:-1], ys[:-1], ys[1:])
        #     )
        # )
        # jax.debug.print(
        #     '{v}',
        #     v=jnp.sum(
        #         jax.vmap(
        #             lambda t, y, y_next: -(y_next.reshape(-1, order='F') - y.reshape(-1, order='F'))[::use_every].T @ jnp.linalg.inv(thesis.processes.process.long_diffusion(dp.diffusion(t, y[::use_every]) @ dp.diffusion(t, y[::use_every]).T, d) * var) @ (y_next.reshape(-1, order='F') - y.reshape(-1, order='F'))[::use_every] / 2,
        #             in_axes=(0, 0, 0)
        #         )(ts[:-1], ys[:-1], ys[1:])
        #     )
        #     -
        #     jnp.sum(
        #         jax.vmap(
        #             lambda t, y, y_next: -(y_next.reshape(-1, order='F') - (y - delta * dp_bar.drift(t, y)).reshape(-1, order='F'))[::use_every].T @ jnp.linalg.inv(thesis.processes.process.long_diffusion(dp_bar.diffusion(t, y[::use_every]) @ dp_bar.diffusion(t, y[::use_every]).T, d) * var) @ (y_next.reshape(-1, order='F') - (y - delta * dp_bar.drift(t, y)).reshape(-1, order='F'))[::use_every] / 2,
        #             in_axes=(0, 0, 0)
        #         )(ts[:-1], ys[:-1], ys[1:])
        #     )
        # )

        w = jnp.exp(
            jnp.sum(
                jax.vmap(
                    lambda t, y, y_next: -(y_next.reshape(-1, order='F') - y.reshape(-1, order='F'))[::use_every].T @ jnp.linalg.inv(thesis.processes.process.long_diffusion(dp.diffusion(t, y[::use_every]) @ dp.diffusion(t, y[::use_every]).T, d) * var) @ (y_next.reshape(-1, order='F') - y.reshape(-1, order='F'))[::use_every] / 2,
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
            -
            jnp.sum(
                jax.vmap(
                    lambda t, y, y_next: -(y_next.reshape(-1, order='F') - (y - delta * dp_bar.drift(t, y)).reshape(-1, order='F'))[::use_every].T @ jnp.linalg.inv(thesis.processes.process.long_diffusion(dp_bar.diffusion(t, y[::use_every]) @ dp_bar.diffusion(t, y[::use_every]).T, d) * var) @ (y_next.reshape(-1, order='F') - (y - delta * dp_bar.drift(t, y)).reshape(-1, order='F'))[::use_every] / 2,
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
        )
        x = ys[-1]

        # jax.debug.print(
        #     'w: {v}',
        #     v=w
        # )

        # jax.debug.print(
        #     'l: {v}',
        #     v=jax.scipy.stats.multivariate_normal.logpdf(constraints.initial.reshape(-1, order='F')[::use_every], x.reshape(-1, order='F')[::use_every], thesis.processes.process.long_diffusion(dp.diffusion(ts[-1], ys[-1][::use_every]) @ dp.diffusion(ts[-1], ys[-1][::use_every]).T, d) * var).squeeze()
        # )
        return {
            'a': jax.scipy.stats.multivariate_normal.logpdf(constraints.initial.reshape(-1, order='F')[::use_every], x.reshape(-1, order='F')[::use_every], thesis.processes.process.long_diffusion(dp.diffusion(ts[-1], ys[-1][::use_every]) @ dp.diffusion(ts[-1], ys[-1][::use_every]).T, d) * var).squeeze(),
            'b': w
        }

    return jax.scipy.special.logsumexp(**jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)


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
    diffusion_process = _provide_constraints(
        selector.add_options_from_module(
            parser,
            'diffusion',
            thesis.experiments.diffusion_processes,
            thesis.experiments.diffusion_processes.DiffusionProcess,
        ),
        constraints=constraints,
    )
    # assert isinstance(diffusion_process, thesis.experiments.diffusion_processes.Brownian)
    simulator = selector.add_options_from_module(
        parser,
        'simulator',
        thesis.experiments.simulators,
        thesis.experiments.simulators.Simulator,
    )()

    checkpoint = selector.get_argument(parser, 'checkpoint', type=str, default=None)
    if checkpoint is not None:
        network = selector.add_options_from_module(
            parser, 'network', thesis.models.networks, thesis.models.networks.Network,
        )
        objective = selector.add_options_from_module(
            parser, 'objective', thesis.models.objectives, thesis.models.objectives.Objective,
        )
        model_initialiser = selector.add_options_from_module(
            parser, 'model', thesis.models.models, thesis.lightning.Module,
        )
        model_initialiser = partial(model_initialiser, network=partial(network, activation=nn.gelu), objective=objective())

        multiple = selector.get_argument(parser, 'multiple', type=bool, default=False)
        print(multiple)
        if multiple:
            states = [
                (
                    path.name,
                    model_initialiser.func.load_from_checkpoint(
                        os.path.join(path.path, 'checkpoints'),
                        dp=diffusion_process.dp,
                        dim=simulator.simulate_sample_path(key, diffusion_process.dp, constraints.initial, t0=0, t1=1, n_steps=1)[1][0].shape[0],
                        **model_initialiser.keywords,
                    )[1]
                )
                for path in os.scandir(checkpoint) if path.is_dir()
            ]
        else:
            model, state = model_initialiser.func.load_from_checkpoint(
                checkpoint,
                dp=diffusion_process.dp,
                dim=simulator.simulate_sample_path(key, diffusion_process.dp, constraints.initial, t0=0, t1=1, n_steps=1)[1][0].shape[0],
                **model_initialiser.keywords,
            )

        displacement = selector.get_argument(parser, 'displacement', type=bool)


    filename = 'pedersen.csv'
    with open(filename, 'w') as f:
        f.write('method,parameter,ll\n')

    n_mc = selector.get_argument(parser, 'n_mc', type=int, default=10_000)
    n_s = selector.get_argument(parser, 'n_s', type=int, default=25)
    k = selector.get_argument(parser, 'n_values', type=int, default=20)

    from_sigma = selector.get_argument(parser, 'from_sigma', type=float)
    to_sigma = selector.get_argument(parser, 'to_sigma', type=float)
    log_scale = selector.get_argument(parser, 'log_scale', type=bool, default=False)

    # sigmas = 10**jnp.linspace(from_sigma, to_sigma, k) if log_scale else jnp.linspace(from_sigma, to_sigma, k)

    dp = diffusion_process.dp

    use_every = selector.get_argument(parser, 'use_every', type=int, default=1)

    # # Analytical
    # f = jax.jit(lambda sigma: analytical(sigma=sigma, t1=1, constraints=constraints))
    # lls = [
    #     f(sigma) for sigma in sigmas
    # ]
    # with open(filename, 'a') as f:
    #     f.writelines(f'analytical,{p},{ll}\n' for p, ll in zip(sigmas, lls))

    if isinstance(diffusion_process, thesis.experiments.diffusion_processes.Brownian):
        # Pedersen
        f = jax.jit(lambda key, sigma: pedersen(key=key, t1=1, constraints=constraints, dp=unconditioned_dp(sigma, dp), simulator=simulator, M=n_mc, N=n_s))
        lls = [
            f(key, sigma) for key, sigma in zip(jax.random.split(key, k), sigmas)
        ]
        with open(filename, 'a') as f:
            f.writelines(f'pedersen,{p},{ll}\n' for p, ll in zip(sigmas, lls))

        # Importance, forwards
        f = jax.jit(lambda key, sigma: pedersen_brownian_bridge(key=key, t1=1, constraints=constraints, dp=unconditioned_dp(sigma, dp), dp_bar=conditioned_dp_forwards(sigma, dp, constraints, 1), simulator=simulator, M=n_mc, N=n_s))
        lls = [
            f(key, sigma) for key, sigma in zip(jax.random.split(key, k), sigmas)
        ]
        with open(filename, 'a') as f:
            f.writelines(f'pedersen_bridge,{p},{ll}\n' for p, ll in zip(sigmas, lls))

        # Importance, backwards
        f = jax.jit(lambda key, sigma: pedersen_brownian_bridge_reverse(key=key, t1=1, constraints=constraints, dp=unconditioned_dp(sigma, dp), dp_bar=conditioned_dp_backwards(sigma, dp, lambda udp: partial(diffusion_process.score_analytical, dp=udp, constraints=constraints)), simulator=simulator, M=n_mc, N=n_s, use_every=use_every))
        lls = [
            f(key, sigma) for key, sigma in zip(jax.random.split(key, k), sigmas)
        ]
        with open(filename, 'a') as f:
            f.writelines(f'pedersen_bridge_reverse,{p},{ll}\n' for p, ll in zip(sigmas, lls))

    # Importance, backwards, learned
    if checkpoint is not None and not multiple:
        f = jax.jit(
            lambda key, sigma:
                pedersen_brownian_bridge_reverse(
                    key=key,
                    t1=1,
                    constraints=constraints,
                    dp=unconditioned_dp(sigma, dp),
                    dp_bar=conditioned_dp_backwards(
                        sigma,
                        dp,
                        lambda _:
                            lambda t, y:
                                diffusion_process.score_learned(
                                    t[None],
                                    (y - (constraints.initial.reshape(y.shape, order='F') if displacement else 0))[None],
                                    state=state,
                                    c=jnp.ones_like(t[None]) * 0.5,
                                )[0]
                    ),
                    simulator=simulator,
                    M=n_mc,
                    N=n_s,
                    use_every=use_every,
                )
        )
        lls = [
            f(key, sigma) for key, sigma in zip(jax.random.split(key, k), sigmas)
        ]
        with open(filename, 'a') as f:
            f.writelines(f'pedersen_bridge_reverse_learned,{p},{ll}\n' for p, ll in zip(sigmas, lls))
        
    if checkpoint is not None and multiple:
        for name, state in states:
            f = jax.jit(
                lambda key, sigma:
                    pedersen_brownian_bridge_reverse(
                        key=key,
                        t1=1,
                        constraints=constraints,
                        dp=unconditioned_dp(sigma**2, dp),
                        dp_bar=conditioned_dp_backwards(
                            sigma**2,
                            dp,
                            lambda _:
                                lambda t, y:
                                    diffusion_process.score_learned(
                                        t[None],
                                        (y - (constraints.initial.reshape(y.shape, order='F') if displacement else 0))[None],
                                        state=state,
                                        c=jnp.ones_like(t[None]) * sigma,
                                    )[0]
                        ),
                        simulator=simulator,
                        M=n_mc,
                        N=n_s,
                        use_every=use_every,
                    )
            )
            print(f'{name},{f(key, float(name))}')


if __name__ == '__main__':
    main()
