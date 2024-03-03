import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from diffrax import (ControlTerm, Euler, MultiTerm, ODETerm, SaveAt,
                     VirtualBrownianTree, diffeqsolve)

import score


def f(xs, ts):
    return jnp.zeros_like(xs)

def predict(xs, ts):
    with torch.no_grad():
        s, _ = model.net(
            (
                torch.tensor(xs).view(-1, 1),
                torch.tensor(1 - ts).view(-1)
            )
        )
        print(s.item())
    return -s.numpy()

def analytical(xs, ts):
    # TODO: Verify this expression!
    return sigma_inverse(xs, ts) * xs / (1 - ts)

def f_bar(xs, ts, use_analytical: bool = False):
    if use_analytical:
        s = analytical(xs, ts)
    else:
        result_shape = jax.ShapeDtypeStruct((1, 1), xs.dtype)
        s = jax.pure_callback(predict, result_shape, xs, ts).squeeze()
    return f(xs, ts) - sigma(xs, ts) * s - sigma_divergence(xs, ts)

def sigma(xs, ts):
    return jnp.ones_like(xs)

def sigma_inverse(xs, ts):
    return jnp.ones_like(xs)

def sigma_divergence(xs, ts):
    return jnp.zeros_like(xs)


model = score.Model.load_from_checkpoint(sys.argv[1])

key = jr.PRNGKey(0)
for _ in range(25):
    key, subkey = jr.split(key)

    brownian_motion = VirtualBrownianTree(0, 1, tol=1e-3, shape=(), key=subkey)
    # terms = ControlTerm(lambda *_: 1, brownian_motion)
    terms = MultiTerm(
        ODETerm(lambda t, x, _: f_bar(x, t, use_analytical=False)),
        ControlTerm(lambda t, x, _: sigma(x, t), brownian_motion)
    )
    solver = Euler()
    saveat = SaveAt(dense=True)
    sol = diffeqsolve(terms, solver, 0.01, 1, dt0=0.01, y0=0, saveat=saveat)

    ts = jnp.linspace(0, 1, 100)
    xs = sol.evaluate(ts)

    plt.plot(ts, xs)

plt.savefig('bbb_analytical.png', dpi=600)
