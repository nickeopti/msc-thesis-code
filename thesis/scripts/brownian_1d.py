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

        self.key, subkey = jax.random.split(self.key)
        ts, ys, n = diffusion.get_data(dp=self.dp, y0=self.y0, key=subkey)

        return self.dp, ts[:n], ys[:n], self.y0, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variance', type=float, default=1)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    dp = process.brownian_motion(
        jnp.array(
            [
                [args.variance]
            ]
        )
    )

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

        logger = loggers.CSVLogger(name='1d')

        t = trainer.Trainer(1500, logger=logger)
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
    yT = jnp.ones(dp.d) * 1

    def f_bar_analytical(t, y):
        s = -dp.inverse_diffusion @ (y - y0) / t
        return dp.drift - dp.diffusion @ s - dp.diffusion_divergence

    def f_bar_learned(t, y):
        s = state.apply_fn(state.params, t[None], y[None], 0)[0]
        return dp.drift - dp.diffusion @ s - dp.diffusion_divergence

    for f_bar, name in ((f_bar_analytical, 'analytical'), (f_bar_learned, 'learned')):
        dp_bar = process.Diffusion(
            d=dp.d,
            drift=f_bar,
            diffusion=dp.diffusion,
            inverse_diffusion=None,
            diffusion_divergence=None,
        )

        illustrations.visualise_sample_paths_f_1d(
            dp=dp_bar,
            key=key,
            filename=plots_path / f'{name}_bridge.png',
            y0=yT,
            t0=1.,
            t1=0.001,
            dt=-0.001,
        )

    illustrations.visualise_sample_paths_1d(
        dp=dp,
        key=key,
        filename=plots_path / 'unconditional.png',
        y0=y0,
        t0=0,
        t1=1,
        dt=0.001,
    )

    s = lambda t, y: - dp.inverse_diffusion @ (y - y0) / t
    illustrations.visualise_vector_field_1d(
        score=jax.vmap(s),
        filename=plots_path / 'analytical_score_vector_field.png',
        t0=0.1,
        a=-1,
        b=1,
    )

    illustrations.visualise_vector_field_1d(
        score=partial(state.apply_fn, state.params, c=0),
        filename=plots_path / 'learned_score_vector_field.png',
        t0=0.1,
        a=-1,
        b=1,
    )


if __name__ == '__main__':
    main()
