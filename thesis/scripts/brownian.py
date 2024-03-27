
import jax
import jax.numpy as jnp

import thesis.lightning.trainer as trainer
import thesis.models.baseline
import thesis.processes.diffusion as diffusion
import thesis.processes.process as process


class Dataset:
    def __init__(self, key, dp: process.Diffusion, n: int) -> None:
        self.key = key
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)

        covariance = jax.random.uniform(subkey1, minval=-1, maxval=1)
        covariance_matrix = jnp.array(
            [
                [1, covariance],
                [covariance, 1]
            ]
        )
        dp = process.brownian_motion(covariance_matrix)

        match index % 2:
            case 0:
                y0 = -jnp.ones(2)
            case 1:
                y0 = jnp.ones(2)
            case 2:
                y0 = jnp.array((2, -1))

        ts, ys, n = diffusion.get_data(dp=dp, y0=y0, key=subkey2)

        return dp, ts[:n], ys[:n], y0, covariance


def main():
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

    dp = process.brownian_motion(jnp.eye(2))
    model = thesis.models.baseline.Model(dp, learning_rate=1e-3)

    t = trainer.Trainer(500)
    t.fit(
        subkey1,
        model,
        Dataset(subkey2, dp, 16),
        Dataset(subkey3, dp, 4),
    )


if __name__ == '__main__':
    main()
