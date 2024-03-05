import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
import torch.utils.data
from diffrax import (ControlTerm, Euler, MultiTerm, ODETerm, SaveAt,
                     VirtualBrownianTree, diffeqsolve)

key = jr.PRNGKey(0)


def get_data(y0 = jnp.zeros(2), t0: float = 0, t1: float = 1, n: int = 100, dt: float = 0.01):
    global key
    key, subkey = jr.split(key)

    brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(2,), key=subkey)
    terms = ControlTerm(lambda *_: jnp.eye(2), brownian_motion)
    solver = Euler()
    saveat = SaveAt(steps=True)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat, max_steps=int(1 / dt))

    return torch.tensor(np.asarray(sol.ts)), torch.tensor(np.asarray(sol.ys))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, i, **kwargs) -> None:
        super().__init__()
        y0 = jnp.zeros(2) if i % 2 else jnp.ones(2)
        self.ts, self.xs = get_data(y0=y0, **kwargs)

    def __len__(self) -> int:
        return len(self.ts)

    def __getitem__(self, index):
        return self.ts[index], self.xs[index]


def get_dataset(n: int = 50, **kwargs):
    return torch.utils.data.ConcatDataset([Dataset(i, **kwargs) for i in range(n)])
