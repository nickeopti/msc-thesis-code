import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
import torch.utils.data
from diffrax import (ControlTerm, Euler, MultiTerm, ODETerm, SaveAt,
                     VirtualBrownianTree, diffeqsolve)

key = jr.PRNGKey(0)


def get_data(y0: float = 0, t0: float = 0, t1: float = 1, n: int = 100, dt: float = 0.01):
    global key
    key, subkey = jr.split(key)

    brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=subkey)
    terms = ControlTerm(lambda *_: 1, brownian_motion)
    solver = Euler()
    saveat = SaveAt(dense=True)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat)

    ts = jnp.linspace(t0, t1, n)
    xs = sol.evaluate(ts)

    return torch.tensor(np.asarray(ts)), torch.tensor(np.asarray(xs))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.ts, self.xs = get_data(**kwargs)

    def __len__(self) -> int:
        return len(self.ts)

    def __getitem__(self, index):
        return self.ts[index], self.xs[index]


def get_dataset(n: int = 50, **kwargs):
    return torch.utils.data.ConcatDataset([Dataset(**kwargs) for _ in range(n)])
