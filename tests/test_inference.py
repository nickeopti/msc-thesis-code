import jax
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt

from thesis.experiments import constraints, diffusion_processes, simulators
from thesis.inference import likelihood

constraint = constraints.CircleLandmarks(100, 2.5, 2, 1)
diffusion_process = diffusion_processes.BrownianWideKernel(1, 0.1, constraint)
dp = diffusion_process.dp
simulator = simulators.AutoLongSimulator()

sigma = 0.1
t1 = 1
M = 100
N = 1000
key = jax.random.key(42)


@jax.jit
def analytical(key, sigma):
    return likelihood.analytical(dp=dp, sigma=sigma, constraints=constraint)


@jax.jit
def stable_analytical(key, sigma):
    return likelihood.stable_analytical(dp=dp, sigma=sigma, constraints=constraint)


@jax.jit
def stable_analytical_offset(key, sigma):
    return likelihood.stable_analytical_offset(dp=dp, sigma=sigma, constraints=constraint)


@jax.jit
def simulated(key, sigma):
    return likelihood.simulated(key=key, t1=t1, constraints=constraint, dp=dp, sigma=sigma, simulator=simulator, M=M, N=N, likelihood=likelihood.analytical)


@jax.jit
def stable_simulated(key, sigma):
    return likelihood.simulated(key=key, t1=t1, constraints=constraint, dp=dp, sigma=sigma, simulator=simulator, M=M, N=N, likelihood=likelihood.stable_analytical)


@jax.jit
def stable_simulated_offset(key, sigma):
    return likelihood.simulated(key=key, t1=t1, constraints=constraint, dp=dp, sigma=sigma, simulator=simulator, M=M, N=N, likelihood=likelihood.stable_analytical_offset)


@jax.jit
def heng(key, sigma):
    dp_bar = likelihood.brownian_bridge_dp(
        sigma=sigma,
        dp=dp,
        constraints=constraint,
        t1=1
    )

    return likelihood.heng(
        key=key,
        t0=0,
        t1=1,
        constraints=constraint,
        dp=dp,
        sigma=sigma,
        dp_bar=dp_bar,
        simulator=simulator,
        M=M,
        N=N,
    )


@jax.jit
def importance_sampled(key, sigma):
    dp_bar = likelihood.brownian_bridge_dp(
        sigma=sigma,
        dp=dp,
        constraints=constraint,
        t1=1
    )

    return likelihood.importance_sampled(
        key=key,
        t0=0,
        t1=1,
        constraints=constraint,
        dp=dp,
        sigma=sigma,
        dp_bar=dp_bar,
        simulator=simulator,
        M=M,
        N=N, 
        stable=False,
        likelihood=likelihood.analytical,
    )


@jax.jit
def stable_importance_sampled_offset(key, sigma):
    dp_bar = likelihood.brownian_bridge_dp(
        sigma=sigma,
        dp=dp,
        constraints=constraint,
        t1=1
    )

    return likelihood.importance_sampled(
        key=key,
        t0=0,
        t1=1,
        constraints=constraint,
        dp=dp,
        sigma=sigma,
        dp_bar=dp_bar,
        simulator=simulator,
        M=M,
        N=N,
        stable=True,
        likelihood=likelihood.stable_analytical_offset,
    )


@jax.jit
def reverse_time_importance_sampled(key, sigma):
    dp_bar = likelihood.reverse_time_conditioned_dp(
        sigma=sigma,
        dp=dp,
        score=lambda dp: lambda t, y: diffusion_process.score_analytical(t, y, dp=dp, constraints=constraint)
    )

    return likelihood.importance_sampled(
        key=key,
        t0=1,
        t1=0,
        constraints=constraint.reversed(),
        dp=dp,
        sigma=sigma,
        dp_bar=dp_bar,
        simulator=simulator,
        M=M,
        N=N, 
        stable=False,
        likelihood=likelihood.analytical,
    )


@jax.jit
def stable_reverse_time_importance_sampled_offset(key, sigma):
    dp_bar = likelihood.reverse_time_conditioned_dp(
        sigma=sigma,
        dp=dp,
        score=lambda dp: lambda t, y: diffusion_process.score_analytical(t, y, dp=dp, constraints=constraint)
    )

    return likelihood.importance_sampled(
        key=key,
        t0=1,
        t1=0,
        constraints=constraint.reversed(),
        dp=dp,
        sigma=sigma,
        dp_bar=dp_bar,
        simulator=simulator,
        M=M,
        N=N, 
        stable=True,
        likelihood=likelihood.stable_analytical_offset,
    )


def test_analytical():
    analytical(key, sigma)


def test_stable_analytical():
    stable_analytical(key, sigma)


def test_stable_analytical_offset():
    stable_analytical_offset(key, sigma)


def test_simulated():
    simulated(key, sigma)


def test_stable_simulated():
    stable_simulated(key, sigma)


def test_stable_simulated_offset():
    stable_analytical_offset(key, sigma)


def test_heng():
    heng(key, sigma)


def test_importance_sampled():
    importance_sampled(key, sigma)


def test_stable_importance_sampled_offset():
    stable_importance_sampled_offset(key, sigma)


def test_reverse_time_importance_sampled():
    reverse_time_importance_sampled(key, sigma)

def test_stable_reverse_time_importance_sampled_offset():
    stable_reverse_time_importance_sampled_offset(key, sigma)


def test_comparison():
    sigmas = jnp.logspace(-2.5, -0.5, 50)
    # sigmas = jnp.logspace(-1.5, -0.5, 10)
    # sigmas = jnp.logspace(-5, -3, 10)

    methods = (
        analytical,
        stable_analytical,
        stable_analytical_offset,
        simulated,
        stable_simulated,
        stable_simulated_offset,
        heng,
        importance_sampled,
        stable_importance_sampled_offset,
        reverse_time_importance_sampled,
        stable_reverse_time_importance_sampled_offset,
    )

    fig, axes = plt.subplots(3, 4, figsize=(12, 10))

    for f, ax in zip(methods, axes.flat):
        lls = [f(key, sigma) for key, sigma in zip(jax.random.split(key, len(sigmas)), sigmas)]
        ax.plot(sigmas, lls)
        ax.set_xscale('log')
        ax.set_title(f.__name__)

    fig.savefig('likelihood-estimation-comparison.png', dpi=600)
