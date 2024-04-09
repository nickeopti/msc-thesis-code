import argparse
import pathlib
from functools import partial

import jax
import jax.numpy as jnp

import thesis.lightning.trainer as trainer
import thesis.models.baseline
import thesis.processes.diffusion as diffusion
import thesis.processes.process as process
from thesis.lightning import loggers
from thesis.visualisations import illustrations


class Dataset:
    def __init__(self, key, y0: jax.Array, dp: process.Diffusion, n: int) -> None:
        self.key = key
        self.y0 = y0
        self.dp = dp
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)

        # covariance = jax.random.uniform(subkey1, minval=-0.8, maxval=0.8)
        covariance = jax.random.uniform(subkey1, minval=0.001, maxval=10)
        covariance_matrix = jnp.array(
            [
                [covariance, 0],
                [0, covariance]
            ]
        )
        dp = process.brownian_motion(covariance_matrix)

        ts, ys, n = diffusion.get_data(dp=dp, y0=self.y0, key=subkey2)

        return dp, ts[:n], ys[:n], self.y0, covariance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    dp = process.brownian_motion(jnp.eye(2))

    if args.checkpoint:
        model, state = thesis.models.baseline.Model.load_from_checkpoint(
            args.checkpoint,
            dp=dp,
        )
        
        plots_path = pathlib.Path(args.checkpoint).parent / 'plots'
    else:
        key = jax.random.PRNGKey(0)
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

        model = thesis.models.baseline.Model(dp, learning_rate=1e-3)

        logger = loggers.CSVLogger(name='2d_likelihood')

        t = trainer.Trainer(500, logger=logger)
        state = t.fit(
            subkey1,
            model,
            Dataset(subkey2, jnp.zeros(dp.d), dp, 16),
            Dataset(subkey3, jnp.zeros(dp.d), dp, 4),
        )

        plots_path = logger.path / 'plots'
    
    # visualise
    # this is in common, whether the model is loaded from a checkpoint,
    # or just has been trained.
        
    plots_path.mkdir(parents=True, exist_ok=True)

    # Deliberately reuse random key
    key = jax.random.PRNGKey(1)
    y0 = jnp.zeros(dp.d)
    yT = jnp.ones(dp.d) * 0.5

    results = []

    # for covariance in (-0.5, -0.3, 0, 0.3, 0.5):
    # for covariance in (0.001, 0.01, 0.1, 1, 2, 5, 10):
    for covariance in jnp.linspace(0.1, 1.9, 10):
        dp = process.brownian_motion(
            jnp.array(
                [
                    [covariance, 0],
                    [0, covariance]
                ]
            )
        )

        def f_bar_analytical(t, y):
            s = -dp.inverse_diffusion @ (y - y0) / t
            return dp.drift - dp.diffusion @ s - dp.diffusion_divergence

        def f_bar_learned(t, y):
            s = state.apply_fn(state.params, t[None], y[None], covariance)[0]
            return dp.drift - dp.diffusion @ s - dp.diffusion_divergence

        for f_bar, name in ((f_bar_analytical, 'analytical'), (f_bar_learned, 'learned')):
            dp_bar = process.Diffusion(
                d=dp.d,
                drift=f_bar,
                diffusion=dp.diffusion,
                inverse_diffusion=dp.inverse_diffusion,
                diffusion_divergence=dp.diffusion_divergence,
            )

            illustrations.visualise_sample_paths_f(
                dp=dp_bar,
                key=key,
                filename=plots_path / f'c_{covariance:.2f}_{name}_bridge.png',
                y0=yT,
                t0=1.,
                t1=0.001,
                dt=-0.001,
            )

        illustrations.visualise_sample_paths(
            dp=dp,
            key=key,
            filename=plots_path / f'c_{covariance:.2f}_unconditional.png',
            y0=y0,
            t0=0,
            t1=1,
            dt=0.001,
        )

        s = lambda t, y: - dp.inverse_diffusion @ (y - y0) / t
        illustrations.visualise_vector_field(
            score=jax.vmap(s),
            filename=plots_path / f'c_{covariance:.2f}_analytical_score_vector_field.png',
        )

        illustrations.visualise_vector_field(
            score=partial(state.apply_fn, state.params, c=covariance),
            filename=plots_path / f'c_{covariance:.2f}_learned_score_vector_field.png',
        )

        # perform approximate likelihood computations
        dp_bar = process.Diffusion(
            d=dp.d,
            drift=f_bar_analytical,
            diffusion=dp.diffusion,
            inverse_diffusion=dp.inverse_diffusion,
            diffusion_divergence=dp.diffusion_divergence,
        )

        dp0 = process.brownian_motion(
            jnp.array(
                [
                    [1, 0.3],
                    [0.3, 1]
                ]
            )
        )

        # compute log-likelihoods, according to original/unconditioned diffusion process
        # of each step in the sampled path
        # and sum those

        def logdet(v):
            r = jnp.linalg.slogdet(v)
            return r.sign * r.logabsdet

        def ll(y, y_next, dt):
            c = -dp_bar.d / 2 * jnp.log(2 * jnp.pi) - logdet(1 * dp_bar.diffusion) / 2
            return c - (y_next - y).T @ dp_bar.inverse_diffusion @ (y_next - y) / dt / 2
        
        import matplotlib.pyplot as plt
        import numpy as np
        a = -3
        b = 3
        val = 15
        nv = 51
        n = 20

        score = jax.vmap(s)

        plt.figure()

        xs = np.linspace(a, b, n)
        ys = np.linspace(a, b, n)
        xx, yy = np.meshgrid(xs, ys)

        s = np.stack((xx.flatten(), yy.flatten())).T
        u, v = score(jnp.ones(n**2) / 1., s).T.reshape(2, n, n)

        plt.contourf(xx, yy, np.sqrt(u**2 + v**2), levels=jnp.linspace(0, val, nv))
        plt.colorbar()
        plt.quiver(xs, ys, u, v)

        likelihoods = []
        _, *point_subkeys = jax.random.split(key, 11)
        for point_key in point_subkeys:
            point = jax.random.multivariate_normal(point_key, jnp.zeros(2), jnp.array([[0.8, 0], [0, 0.8]]))
            print(point)

            _, *path_subkeys = jax.random.split(key, 11)
            for path_subkey in path_subkeys:
                ts, ys, n = diffusion.get_paths(
                    dp=dp_bar,
                    key=path_subkey,
                    y0=point,
                    t0=1.,
                    t1=0.001,
                    dt=-0.01,
                )

                plt.plot(*ys[:n].T, linewidth=0.5, c='black', alpha=0.2)

                lls = jax.jit(jax.vmap(ll))(ys[:n-1], ys[1:n], ts[1:n] - ts[:n-1])
                likelihoods.append(jnp.sum(lls))
                print(covariance, jnp.sum(lls))

            plt.scatter(*point, s=10, alpha=1)
        
        r = sum(likelihoods) / len(likelihoods)
        results.append((covariance, r))
        plt.title(f'Variance: {covariance:.2f}, log-likelihood: {r:.2f}')

        plt.xlim(a, b)
        plt.ylim(a, b)
        plt.savefig(plots_path / f'{covariance:.2f}_likelihood_investigation.png', dpi=600)
    
    print(results)

    plt.figure()
    plt.plot(*zip(*results))
    plt.savefig(plots_path / 'likelihood.png', dpi=600)

if __name__ == '__main__':
    main()
