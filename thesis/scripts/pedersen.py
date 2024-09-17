import argparse
import os
import os.path
from functools import partial
from typing import Callable

import jax
import jax.dtypes
import jax.lax
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
from thesis.experiments.constraints import Constraints, LandmarksConstraints
from thesis.experiments.simulators import Simulator
from thesis.processes.process import Diffusion
from thesis.scripts.common import _provide_constraints

jax.config.update("jax_enable_x64", True)


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


def analytical(dp: Diffusion, sigma: float, constraints: Constraints):
    return jnp.sum(
        jax.vmap(
            lambda terminal, initial: jax.scipy.stats.multivariate_normal.logpdf(terminal, initial, dp.diffusion(0, constraints.initial) @ dp.diffusion(0, constraints.initial).T * sigma).squeeze(),
            in_axes=(1, 1)
        )(constraints.terminal, constraints.initial)
    )


def stable_analytical(dp: Diffusion, sigma: float, constraints: Constraints):
    k, _ = constraints.shape

    def f(terminal, initial):
        logdet = jnp.log(jnp.diagonal(jax.lax.linalg.cholesky(sigma * dp.diffusion(0, constraints.initial) @ dp.diffusion(0, constraints.initial).T))).sum()
        slogdet = jnp.linalg.slogdet(sigma * dp.diffusion(0, constraints.initial) @ dp.diffusion(0, constraints.initial).T)
        logdet = slogdet.sign * slogdet.logabsdet
        z, *_ = jnp.linalg.lstsq(dp.diffusion(0, constraints.initial), terminal - initial)
        return -k / 2 * jnp.log(2 * jnp.pi) - logdet / 2 - 1 / 2 * z.T @ z / sigma

    return jnp.sum(
        jax.vmap(
            lambda terminal, initial: f(terminal, initial),
            in_axes=(1, 1)
        )(constraints.terminal, constraints.initial)
    )


def stable_analytical_offset(dp: Diffusion, sigma: float, constraints: Constraints):
    k, _ = constraints.shape

    def f(terminal, initial):
        z, *_ = jnp.linalg.lstsq(dp.diffusion(0, constraints.initial), terminal - initial)
        return -k / 2 * jnp.log(sigma) - 1 / 2 * z.T @ z / sigma

    return jnp.sum(
        jax.vmap(
            lambda terminal, initial: f(terminal, initial),
            in_axes=(1, 1)
        )(constraints.terminal, constraints.initial)
    )


def pedersen(key: jax.dtypes.prng_key, t1: float, constraints: Constraints, dp: Diffusion, simulator: Simulator, M: int, N: int):
    delta = t1 / N
    var = delta

    def f(key):
        _, ys = simulator.simulate_sample_path(key, dp, constraints.initial, t0=0, t1=t1 - delta, n_steps=N)
        return analytical(dp, var, LandmarksConstraints(ys[-1], constraints.terminal))

    return jax.scipy.special.logsumexp(jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)


def stable_pedersen(key: jax.dtypes.prng_key, t1: float, constraints: Constraints, dp: Diffusion, sigma: float, simulator: Simulator, M: int, N: int):
    delta = t1 / N
    var = delta

    def f(key):
        _, ys = simulator.simulate_sample_path(key, unconditioned_dp(sigma, dp), constraints.initial, t0=0, t1=t1 - delta, n_steps=N)
        return stable_analytical(dp, sigma * var, LandmarksConstraints(ys[-1], constraints.terminal))

    return jax.scipy.special.logsumexp(jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)


def stable_pedersen_offset(key: jax.dtypes.prng_key, t1: float, constraints: Constraints, dp: Diffusion, sigma: float, simulator: Simulator, M: int, N: int):
    delta = t1 / N
    var = delta

    def f(key):
        _, ys = simulator.simulate_sample_path(key, unconditioned_dp(sigma, dp), constraints.initial, t0=0, t1=t1 - delta, n_steps=N)
        return stable_analytical_offset(dp, sigma * var, LandmarksConstraints(ys[-1], constraints.terminal))

    return jax.scipy.special.logsumexp(jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)


def pedersen_brownian_bridge(key: jax.dtypes.prng_key, t1: float, constraints: Constraints, dp: Diffusion, dp_bar: Diffusion, simulator: Simulator, M: int, N: int, sigma: float):
    delta = t1 / N
    var = delta

    def f(key):
        ts, ys = simulator.simulate_sample_path(key, dp_bar, constraints.initial, t0=0, t1=t1 - delta, n_steps=N)
        ts = jnp.hstack((0, ts))
        ys = jnp.vstack((constraints.initial[None], ys))
        
        w = (
            jnp.sum(
                jax.vmap(
                    lambda t, y, y_next:
                        jnp.sum(
                            jax.vmap(
                                lambda a, b: -(b - a).T @ jnp.linalg.inv(dp.diffusion(t, y) @ dp.diffusion(t, y).T * var) @ (b - a) / 2,
                                in_axes=(1, 1)
                            )(y, y_next)
                        ),
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
            -
            jnp.sum(
                jax.vmap(
                    lambda t, y, y_next:
                        jnp.sum(
                            jax.vmap(
                                lambda a, b, d: -(b - (a + delta * d)).T @ jnp.linalg.inv(dp_bar.diffusion(t, y) @ dp_bar.diffusion(t, y).T * var) @ (b - (a + delta * d)) / 2,
                                in_axes=(1, 1, 1)
                            )(y, y_next, dp_bar.drift(t, y))
                        ),
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
        )
        x = ys[-1]

        return w + jnp.sum(
            jax.vmap(
                lambda terminal, a: jax.scipy.stats.multivariate_normal.logpdf(terminal, a, dp.diffusion(ts[-1], ys[-1]) @ dp.diffusion(ts[-1], ys[-1]).T * var).squeeze(),
                in_axes=(1, 1)
            )(constraints.terminal, x)
        )

    return jax.scipy.special.logsumexp(jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)


def stable_pedersen_brownian_bridge_offset(key: jax.dtypes.prng_key, t1: float, constraints: Constraints, dp: Diffusion, dp_bar: Diffusion, simulator: Simulator, M: int, N: int, sigma: float):
    delta = t1 / N
    var = delta

    def f(key):
        ts, ys = simulator.simulate_sample_path(key, dp_bar, constraints.initial, t0=0, t1=t1 - delta, n_steps=N)
        ts = jnp.hstack((0, ts))
        ys = jnp.vstack((constraints.initial[None], ys))

        def stable_prod(a, b, diffusion):
            z, *_ = jnp.linalg.lstsq(diffusion, b - a)
            return z.T @ z

        w = (
            jnp.sum(
                jax.vmap(
                    lambda t, y, y_next:
                        jnp.sum(
                            jax.vmap(
                                # lambda a, b: -(b - a).T @ jnp.linalg.inv(sigma * dp.diffusion(t, y) @ dp.diffusion(t, y).T * var) @ (b - a) / 2,
                                lambda a, b: -stable_prod(a, b, jnp.sqrt(sigma) * dp.diffusion(t, y) * jnp.sqrt(var)) / 2,
                                in_axes=(1, 1)
                            )(y, y_next)
                        ),
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
            -
            jnp.sum(
                jax.vmap(
                    lambda t, y, y_next:
                        jnp.sum(
                            jax.vmap(
                                # lambda a, b, d: -(b - (a + delta * d)).T @ jnp.linalg.inv(dp_bar.diffusion(t, y) @ dp_bar.diffusion(t, y).T * var) @ (b - (a + delta * d)) / 2,
                                lambda a, b, d: -stable_prod(a + delta * d, b, dp_bar.diffusion(t, y) * jnp.sqrt(var)) / 2,
                                in_axes=(1, 1, 1)
                            )(y, y_next, dp_bar.drift(t, y))
                        ),
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
        )

        return w + stable_analytical_offset(dp, sigma * var, LandmarksConstraints(ys[-1], constraints.terminal))

    return jax.scipy.special.logsumexp(jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)


def pedersen_brownian_bridge_reverse(key: jax.dtypes.prng_key, t1: float, constraints: Constraints, dp: Diffusion, dp_bar: Diffusion, simulator: Simulator, M: int, N: int, use_every: int):
    delta = t1 / N
    var = delta

    def f(key):
        ts, ys = simulator.simulate_sample_path(key, dp_bar, constraints.terminal, t0=t1, t1=delta, n_steps=N)
        ts = jnp.hstack((t1, ts))
        ys = jnp.vstack((constraints.terminal[None], ys))
        
        w = (
            jnp.sum(
                jax.vmap(
                    lambda t, y, y_next:
                        jnp.sum(
                            jax.vmap(
                                lambda a, b: -(b - a).T @ jnp.linalg.inv(dp.diffusion(t, y) @ dp.diffusion(t, y).T * var) @ (b - a) / 2,
                                in_axes=(1, 1)
                            )(y, y_next)
                        ),
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
            -
            jnp.sum(
                jax.vmap(
                    lambda t, y, y_next:
                        jnp.sum(
                            jax.vmap(
                                lambda a, b, d: -(b - (a - delta * d)).T @ jnp.linalg.inv(dp_bar.diffusion(t, y) @ dp_bar.diffusion(t, y).T * var) @ (b - (a - delta * d)) / 2,
                                in_axes=(1, 1, 1)
                            )(y, y_next, dp_bar.drift(t, y))
                        ),
                    in_axes=(0, 0, 0)
                )(ts[:-1], ys[:-1], ys[1:])
            )
        )
        x = ys[-1]

        return w + jnp.sum(
            jax.vmap(
                lambda initial, a: jax.scipy.stats.multivariate_normal.logpdf(initial, a, dp.diffusion(ts[-1], ys[-1]) @ dp.diffusion(ts[-1], ys[-1]).T * var).squeeze(),
                in_axes=(1, 1)
            )(constraints.initial, x)
        )

    return jax.scipy.special.logsumexp(jax.vmap(f)(jax.random.split(key, M))) - jnp.log(M)


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
    print(constraints.shape)
    print(constraints.initial.min(), constraints.initial.max())
    diffusion_process: thesis.experiments.diffusion_processes.Brownian = _provide_constraints(
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

    sigmas = 10**jnp.linspace(from_sigma, to_sigma, k) if log_scale else jnp.linspace(from_sigma, to_sigma, k)

    dp = diffusion_process.dp

    # Diffusion mean estimation testing
    def shape(key):
        ts, ys = simulator.simulate_sample_path(key, dp, constraints.initial, t0=0, t1=1, n_steps=10)
        return ys[-1]
    

    def ll_to_optimise(initial, terminal):
        # sigma = dp.diffusion(jnp.zeros(1), initial)[0, 0]
        # print(sigma)
        sigma = 0.1

        return stable_pedersen_brownian_bridge_offset(
            key=key,
            t1=1,
            constraints=LandmarksConstraints(initial, terminal),
            dp=dp,
            dp_bar=conditioned_dp_forwards(sigma, dp, constraints, 1),
            simulator=simulator,
            M=n_mc,
            N=n_s,
            sigma=sigma,
        )
    
    shapes = jax.vmap(shape)(jax.random.split(key, 10))

    @jax.jit
    @jax.grad
    def f(initial):
        return jnp.sum(jax.vmap(lambda terminal: ll_to_optimise(initial, terminal))(shapes))
    
    mean_shape = jnp.mean(shapes, axis=0)
    # diffusion_mean_estimate = (constraints.initial + constraints.terminal) / 2
    diffusion_mean_estimate = constraints.terminal
    # diffusion_mean_estimate = constraints.initial
    # diffusion_mean_estimate = mean_shape

    for i in range(100):
        diffusion_mean_estimate += 0.0001 * f(diffusion_mean_estimate)
        print(i, jnp.sqrt(jnp.mean((diffusion_mean_estimate - mean_shape)**2)))

    # shape_grad = jax.grad(f)(jnp.mean(shapes, axis=0))
    # print(f'{shape_grad.shape=}')
    
    # End of diffusion mean testing

    # Analytical
    f = jax.jit(lambda sigma: analytical(dp=dp, sigma=sigma, constraints=constraints))
    lls = [
        f(sigma) for sigma in sigmas
    ]
    with open(filename, 'a') as f:
        f.writelines(f'analytical,{p},{ll}\n' for p, ll in zip(sigmas, lls))

    # Analytical, stable
    f = jax.jit(lambda sigma: stable_analytical(dp=dp, sigma=sigma, constraints=constraints))
    lls = [
        f(sigma) for sigma in sigmas
    ]
    with open(filename, 'a') as f:
        f.writelines(f'stable_analytical,{p},{ll}\n' for p, ll in zip(sigmas, lls))
    
    # Analytical, stable, constant offset
    f = jax.jit(lambda sigma: stable_analytical_offset(dp=dp, sigma=sigma, constraints=constraints))
    lls = [
        f(sigma) for sigma in sigmas
    ]
    with open(filename, 'a') as f:
        f.writelines(f'stable_analytical_offset,{p},{ll}\n' for p, ll in zip(sigmas, lls))

    # Pedersen
    f = jax.jit(lambda key, sigma: pedersen(key=key, t1=1, constraints=constraints, dp=unconditioned_dp(sigma, dp), simulator=simulator, M=n_mc, N=n_s))
    lls = [
        f(key, sigma) for key, sigma in zip(jax.random.split(key, k), sigmas)
    ]
    with open(filename, 'a') as f:
        f.writelines(f'pedersen,{p},{ll}\n' for p, ll in zip(sigmas, lls))

    # Pedersen, stable
    f = jax.jit(lambda key, sigma: stable_pedersen(key=key, t1=1, constraints=constraints, dp=dp, sigma=sigma, simulator=simulator, M=n_mc, N=n_s))
    lls = [
        f(key, sigma) for key, sigma in zip(jax.random.split(key, k), sigmas)
    ]
    with open(filename, 'a') as f:
        f.writelines(f'pedersen_stable,{p},{ll}\n' for p, ll in zip(sigmas, lls))

    # Pedersen, stable, constant offset
    f = jax.jit(lambda key, sigma: stable_pedersen_offset(key=key, t1=1, constraints=constraints, dp=dp, sigma=sigma, simulator=simulator, M=n_mc, N=n_s))
    lls = [
        f(key, sigma) for key, sigma in zip(jax.random.split(key, k), sigmas)
    ]
    with open(filename, 'a') as f:
        f.writelines(f'pedersen_stable_offset,{p},{ll}\n' for p, ll in zip(sigmas, lls))

    # Importance, forwards
    f = jax.jit(lambda key, sigma: pedersen_brownian_bridge(key=key, t1=1, constraints=constraints, dp=unconditioned_dp(sigma, dp), dp_bar=conditioned_dp_forwards(sigma, dp, constraints, 1), simulator=simulator, M=n_mc, N=n_s, sigma=sigma))
    lls = [
        f(key, sigma) for key, sigma in zip(jax.random.split(key, k), sigmas)
    ]
    with open(filename, 'a') as f:
        f.writelines(f'pedersen_bridge,{p},{ll}\n' for p, ll in zip(sigmas, lls))

    # Importance, forwards, stable, constant offset
    f = jax.jit(lambda key, sigma: stable_pedersen_brownian_bridge_offset(key=key, t1=1, constraints=constraints, dp=dp, dp_bar=conditioned_dp_forwards(sigma, dp, constraints, 1), simulator=simulator, M=n_mc, N=n_s, sigma=sigma))
    lls = [
        f(key, sigma) for key, sigma in zip(jax.random.split(key, k), sigmas)
    ]
    with open(filename, 'a') as f:
        f.writelines(f'pedersen_bridge_stable_offset,{p},{ll}\n' for p, ll in zip(sigmas, lls))

    # Importance, backwards
    f = jax.jit(lambda key, sigma: pedersen_brownian_bridge_reverse(key=key, t1=1, constraints=constraints, dp=unconditioned_dp(sigma, dp), dp_bar=conditioned_dp_backwards(sigma, dp, lambda udp: partial(diffusion_process.score_analytical, dp=udp, constraints=constraints)), simulator=simulator, M=n_mc, N=n_s))
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
