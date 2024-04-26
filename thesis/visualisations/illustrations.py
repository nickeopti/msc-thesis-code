from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

import thesis.processes.diffusion as diffusion
import thesis.processes.process as process


def visualise_sample_paths(dp: process.Diffusion, key, filename, n: int = 5, **kwargs):
    plt.figure()

    for _ in range(n):
        key, subkey = jax.random.split(key)

        _, ys, n = diffusion.get_data(
            dp=dp,
            key=subkey,
            **kwargs
        )

        plt.plot(*ys[:n].T, linewidth=1, alpha=0.6)
        plt.scatter(*ys[n-1], alpha=1)
        print(ys[n-1])

    plt.savefig(filename, dpi=600)


def visualise_sample_paths_1d(dp: process.Diffusion, key, filename, n: int = 5, **kwargs):
    plt.figure()

    for _ in range(n):
        key, subkey = jax.random.split(key)

        ts, ys, n = diffusion.get_data(
            dp=dp,
            key=subkey,
            **kwargs
        )

        plt.plot(ts[:n], ys[:n, 0], linewidth=1, alpha=0.6)
        plt.scatter(ts[n-1], ys[n-1], alpha=1)
        print(ys[n-1])

    plt.xlabel('t')
    plt.ylabel('y')

    plt.savefig(filename, dpi=600)


def visualise_sample_paths_f(dp: process.Diffusion, key, filename, n: int = 5, **kwargs):
    plt.figure()

    for _ in range(n):
        key, subkey = jax.random.split(key)

        _, ys, n = diffusion.get_paths(
            dp=dp,
            key=subkey,
            **kwargs
        )

        plt.plot(*ys[:n].T, linewidth=1, alpha=0.6)
        plt.scatter(*ys[n-1], alpha=1)
        print(ys[n-1])

    plt.savefig(filename, dpi=600)


def visualise_sample_paths_f_1d(dp: process.Diffusion, key, filename, n: int = 5, **kwargs):
    plt.figure()

    for _ in range(n):
        key, subkey = jax.random.split(key)

        ts, ys, n = diffusion.get_paths(
            dp=dp,
            key=subkey,
            **kwargs
        )

        plt.plot(ts[:n], ys[:n, 0], linewidth=1, alpha=0.6)
        plt.scatter(ts[n-1], ys[n-1], alpha=1)
        print(ys[n-1])

    plt.xlabel('t')
    plt.ylabel('y')

    plt.savefig(filename, dpi=600)


def visualise_vector_field(score: Callable[[jax.Array, jax.Array], jax.Array], filename, n: int = 20, a: float = -3, b: float = 3, val: float = 5, nv: int = 51):
    plt.figure()

    xs = np.linspace(a, b, n)
    ys = np.linspace(a, b, n)
    xx, yy = np.meshgrid(xs, ys)

    s = np.stack((xx.flatten(), yy.flatten())).T
    u, v = score(jnp.ones(n**2) / 1., s).T.reshape(2, n, n)

    plt.contourf(xx, yy, np.sqrt(u**2 + v**2), levels=jnp.linspace(0, val, nv))
    plt.colorbar()
    plt.quiver(xs, ys, u, v)

    plt.savefig(filename, dpi=600)


def visualise_vector_field_1d(score: Callable[[jax.Array, jax.Array], jax.Array], filename, n: int = 20, t0: float = 0.001, t1: float = 1, a: float = -3, b: float = 3, val: float = 5, nv: int = 51):
    plt.figure()

    xs = np.linspace(t0, t1, n)
    ys = np.linspace(a, b, n)
    xx, yy = np.meshgrid(xs, ys)

    v = score(xx.reshape(-1), yy.reshape(-1, 1)).T.reshape(n, n)

    plt.contourf(xx, yy, jnp.abs(v), levels=jnp.linspace(0, val, nv))
    plt.colorbar()
    plt.quiver(xs, ys, jnp.zeros((20, 20)), v.reshape(20, 20))

    plt.xlabel('t')

    plt.savefig(filename, dpi=600)


def visualise_circle_sample_paths_f(dp: process.Diffusion, key, filename, n: int = 5, **kwargs):
    plt.figure()
    import cycler
    plt.rc('axes', prop_cycle=cycler.cycler(color=plt.colormaps.get_cmap('tab20').colors))

    k = dp.diffusion.shape[0] // 2

    for _ in range(n):
        key, subkey = jax.random.split(key)

        _, ys, n = diffusion.get_paths(
            dp=dp,
            key=subkey,
            **kwargs
        )

        for i in range(k):
            # print(i, *ys[0, [i, k + i]])
            plt.plot(*ys[:n, [i, k + i]].T, linewidth=1, alpha=0.6, color=f'C{i}')
            plt.scatter(*ys[n - 1, [i, k + i]], alpha=1, color=f'C{i}', marker='+')

        for i in range(k):
            plt.scatter(*ys[0, [i, k + i]], alpha=1, color=f'C{i}', marker='x')

        polygon_T = patches.Polygon(jnp.vstack((ys[0, :k], ys[0, k:])).T, fill=False)
        polygon_0 = patches.Polygon(jnp.vstack((ys[n - 1, :k], ys[n - 1, k:])).T, fill=False, linestyle='--')

        plt.gca().add_patch(polygon_0)
        plt.gca().add_patch(polygon_T)

    plt.gca().set_aspect('equal')
    plt.savefig(filename, dpi=600)


def visualise_circle_sample_paths_f_factorised(dp: process.Diffusion, score, key, filename, n: int = 5, **kwargs):
    plt.figure()
    import cycler
    plt.rc('axes', prop_cycle=cycler.cycler(color=plt.colormaps.get_cmap('tab20').colors))

    k = dp.d

    y0 = kwargs['y0']
    kwargs_x = kwargs.copy()
    kwargs_y = kwargs.copy()
    kwargs_x['y0'] = y0[:, 0]
    kwargs_y['y0'] = y0[:, 1]

    if isinstance(dp.drift, partial) and 'y0' in dp.drift.keywords:
        dp_x = process.Diffusion(
            d=dp.d,
            drift=partial(dp.drift, y0=dp.drift.keywords['y0'][:, 0]),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )
        dp_y = process.Diffusion(
            d=dp.d,
            drift=partial(dp.drift, y0=dp.drift.keywords['y0'][:, 1]),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )
    else:
        def w(f):
            def us(t, q):
                v = f(t, q[:, None])
                return v
            return us
        dp_x = process.Diffusion(
            d=dp.d,
            drift=w(dp.drift),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )
        dp_y = process.Diffusion(
            d=dp.d,
            drift=w(dp.drift),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )

    for _ in range(n):
        key, subkey_x, subkey_y = jax.random.split(key, 3)

        ts_x, ys_x, n_x = diffusion.get_paths(
            dp=dp_x,
            key=subkey_x,
            **kwargs_x
        )

        ts_y, ys_y, n_y = diffusion.get_paths(
            dp=dp_y,
            key=subkey_y,
            **kwargs_y
        )

        assert n_x == n_y
        assert jnp.all(ts_x[:n_x] == ts_y[:n_y])

        for i in range(k):
            plt.plot(ys_x[:n_x, i], ys_y[:n_y, i], linewidth=1, alpha=0.6, color=f'C{i}')
            plt.scatter(ys_x[n_x - 1, i], ys_y[n_y - 1, i], alpha=1, color=f'C{i}', marker='+')

        # p = jnp.hstack((ys_x[0], ys_y[0]))
        # vector = score(ts_x[0][None], ys_x[0][None])
        # plt.arrow(*p, *vector)

        for i in range(k):
            plt.scatter(ys_x[0, i], ys_y[0, i], alpha=1, color=f'C{i}', marker='x')

        polygon_T = patches.Polygon(jnp.vstack((ys_x[0], ys_y[0])).T, fill=False)
        polygon_0 = patches.Polygon(jnp.vstack((ys_x[n_x - 1], ys_y[n_x - 1])).T, fill=False, linestyle='--')

        plt.gca().add_patch(polygon_0)
        plt.gca().add_patch(polygon_T)

    plt.gca().set_aspect('equal')
    plt.savefig(filename, dpi=600)


def visualise_circle_sample_paths_f_3d(dp: process.Diffusion, score, key, filename, n: int = 5, **kwargs):
    ax = plt.figure().add_subplot(projection='3d')
    import cycler
    plt.rc('axes', prop_cycle=cycler.cycler(color=plt.colormaps.get_cmap('tab20').colors))

    k = dp.diffusion.shape[0] // 3

    for _ in range(n):
        key, subkey = jax.random.split(key)

        _, ys, n = diffusion.get_paths(
            dp=dp,
            key=subkey,
            **kwargs
        )

        for i in range(k):
            plt.plot(*ys[:n, [i, k + i, 2 * k + i]].T, linewidth=1, alpha=0.6, color=f'C{i}')
            # plt.scatter(*ys[n - 1, [i, k + i, 2 * k + i]], alpha=1, color=f'C{i}', marker='+')
            # plt.scatter(*ys[0, [i, k + i, 2 * k + i]], alpha=1, color=f'C{i}', marker='x')

        ax.plot(jnp.hstack((ys[0, :k], ys[0, 0])), jnp.hstack((ys[0, k:2*k], ys[0, k])), jnp.hstack((ys[0, 2*k:], ys[0, 2*k])), color='black')
        ax.plot(jnp.hstack((ys[n - 1, :k], ys[n - 1, 0])), jnp.hstack((ys[n - 1, k:2*k], ys[n - 1, k])), jnp.hstack((ys[n - 1, 2*k:], ys[n - 1, 2*k])), color='black', linestyle='--')

    ax.set_aspect('equalxy')
    plt.savefig(filename, dpi=600)


def visualise_circle_sample_paths_f_factorised_3d(dp: process.Diffusion, score, key, filename, n: int = 5, **kwargs):
    ax = plt.figure().add_subplot(projection='3d')
    import cycler
    plt.rc('axes', prop_cycle=cycler.cycler(color=plt.colormaps.get_cmap('tab20').colors))

    k = dp.d

    y0 = kwargs['y0']
    kwargs_x = kwargs.copy()
    kwargs_y = kwargs.copy()
    kwargs_z = kwargs.copy()
    kwargs_x['y0'] = y0[:, 0]
    kwargs_y['y0'] = y0[:, 1]
    kwargs_z['y0'] = y0[:, 2]

    if isinstance(dp.drift, partial) and 'y0' in dp.drift.keywords:
        dp_x = process.Diffusion(
            d=dp.d,
            drift=partial(dp.drift, y0=dp.drift.keywords['y0'][:, 0]),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )
        dp_y = process.Diffusion(
            d=dp.d,
            drift=partial(dp.drift, y0=dp.drift.keywords['y0'][:, 1]),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )
        dp_z = process.Diffusion(
            d=dp.d,
            drift=partial(dp.drift, y0=dp.drift.keywords['y0'][:, 2]),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )
    else:
        def w(f):
            def us(t, q):
                v = f(t, q[:, None])
                return v
            return us
        dp_x = process.Diffusion(
            d=dp.d,
            drift=w(dp.drift),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )
        dp_y = process.Diffusion(
            d=dp.d,
            drift=w(dp.drift),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )
        dp_z = process.Diffusion(
            d=dp.d,
            drift=w(dp.drift),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )

    for _ in range(n):
        key, subkey_x, subkey_y, subkey_z = jax.random.split(key, 4)

        ts_x, ys_x, n_x = diffusion.get_paths(
            dp=dp_x,
            key=subkey_x,
            **kwargs_x
        )

        ts_y, ys_y, n_y = diffusion.get_paths(
            dp=dp_y,
            key=subkey_y,
            **kwargs_y
        )

        ts_z, ys_z, n_z = diffusion.get_paths(
            dp=dp_z,
            key=subkey_z,
            **kwargs_z
        )

        assert n_x == n_y == n_z
        assert jnp.all(ts_x[:n_x] == ts_y[:n_y])
        assert jnp.all(ts_y[:n_y] == ts_z[:n_z])

        for i in range(k):
            ax.plot(ys_x[:n_x, i], ys_y[:n_y, i], ys_z[:n_z, i], linewidth=1, alpha=0.6, color=f'C{i}')
            ax.scatter(ys_x[n_x - 1, i], ys_y[n_y - 1, i], ys_z[n_z - 1, i], alpha=1, color=f'C{i}', marker='+')
            ax.scatter(ys_x[0, i], ys_y[0, i], ys_z[0, i], alpha=1, color=f'C{i}', marker='x')

        ax.plot(jnp.hstack((ys_x[0], ys_x[0, 0])), jnp.hstack((ys_y[0], ys_y[0, 0])), jnp.hstack((ys_z[0], ys_z[0, 0])), color='black')
        ax.plot(jnp.hstack((ys_x[n_x - 1], ys_x[n_x - 1, 0])), jnp.hstack((ys_y[n_y - 1], ys_y[n_y - 1, 0])), jnp.hstack((ys_z[n_z - 1], ys_z[n_z - 1, 0])), color='black', linestyle='--')

    ax.set_aspect('equalxy')
    plt.savefig(filename, dpi=600)


def visualise_circle_sample_paths_f_factorised_3d_ball(dp: process.Diffusion, score, key, filename, n: int = 5, **kwargs):
    import cycler
    plt.rc('axes', prop_cycle=cycler.cycler(color=plt.colormaps.get_cmap('tab20').colors))

    fig = plt.figure()
    axT = fig.add_subplot(1, 2, 1, projection='3d')
    ax0 = fig.add_subplot(1, 2, 2, projection='3d', sharex=axT, sharey=axT, sharez=axT)

    k = dp.d

    y0 = kwargs['y0']
    kwargs_x = kwargs.copy()
    kwargs_y = kwargs.copy()
    kwargs_z = kwargs.copy()
    kwargs_x['y0'] = y0[:, 0]
    kwargs_y['y0'] = y0[:, 1]
    kwargs_z['y0'] = y0[:, 2]

    if isinstance(dp.drift, partial) and 'y0' in dp.drift.keywords:
        dp_x = process.Diffusion(
            d=dp.d,
            drift=partial(dp.drift, y0=dp.drift.keywords['y0'][:, 0]),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )
        dp_y = process.Diffusion(
            d=dp.d,
            drift=partial(dp.drift, y0=dp.drift.keywords['y0'][:, 1]),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )
        dp_z = process.Diffusion(
            d=dp.d,
            drift=partial(dp.drift, y0=dp.drift.keywords['y0'][:, 2]),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )
    else:
        def w(f):
            def us(t, q):
                v = f(t, q[:, None])
                return v
            return us
        dp_x = process.Diffusion(
            d=dp.d,
            drift=w(dp.drift),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )
        dp_y = process.Diffusion(
            d=dp.d,
            drift=w(dp.drift),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )
        dp_z = process.Diffusion(
            d=dp.d,
            drift=w(dp.drift),
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )

    for _ in range(n):
        key, subkey_x, subkey_y, subkey_z = jax.random.split(key, 4)

        ts_x, ys_x, n_x = diffusion.get_paths(
            dp=dp_x,
            key=subkey_x,
            **kwargs_x
        )

        ts_y, ys_y, n_y = diffusion.get_paths(
            dp=dp_y,
            key=subkey_y,
            **kwargs_y
        )

        ts_z, ys_z, n_z = diffusion.get_paths(
            dp=dp_z,
            key=subkey_z,
            **kwargs_z
        )

        assert n_x == n_y == n_z
        assert jnp.all(ts_x[:n_x] == ts_y[:n_y])
        assert jnp.all(ts_y[:n_y] == ts_z[:n_z])

        for i in range(k):
            axT.scatter(ys_x[0, i], ys_y[0, i], ys_z[0, i], alpha=1, color=f'C{i}', marker='+')
            ax0.scatter(ys_x[n_x - 1, i], ys_y[n_y - 1, i], ys_z[n_z - 1, i], alpha=1, color=f'C{i}', marker='+')

    ax0.set_aspect('equal')
    axT.set_aspect('equal')
    plt.savefig(filename, dpi=600)
