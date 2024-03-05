import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from diffrax import (ControlTerm, Euler, MultiTerm, ODETerm, SaveAt,
                     VirtualBrownianTree, diffeqsolve)

import score


def f(xs, ts):
    return jnp.zeros_like(xs)

def predict(xs, ts):
    with torch.no_grad():
        s = model(torch.tensor(xs, dtype=torch.float).unsqueeze(0), torch.tensor(1 - ts, dtype=torch.float).unsqueeze(0))
    return -s.numpy()

def analytical(xs, ts):
    # TODO: Verify this expression!
    return sigma_inverse(xs, ts) @ xs / (1 - ts)

def f_bar(xs, ts, use_analytical: bool = False):
    if use_analytical:
        s = analytical(xs, ts)
    else:
        result_shape = jax.ShapeDtypeStruct((1, 2), xs.dtype)
        s = jax.pure_callback(predict, result_shape, xs, ts).squeeze()
    return f(xs, ts) - sigma(xs, ts) @ s - sigma_divergence(xs, ts)

def sigma(xs, ts):
    # return jnp.ones_like(xs)
    return jnp.eye(xs.shape[-1])

def sigma_inverse(xs, ts):
    # return jnp.ones_like(xs)
    return jnp.eye(xs.shape[-1])

def sigma_divergence(xs, ts):
    return jnp.zeros_like(xs)


model = score.Model.load_from_checkpoint(sys.argv[1])

learned = True

key = jr.PRNGKey(0)
for _ in range(5):
    key, subkey = jr.split(key)
    dt = 0.001

    brownian_motion = VirtualBrownianTree(0, 1, tol=1e-3, shape=(2,), key=subkey)
    # terms = ControlTerm(lambda *_: 1, brownian_motion)
    terms = MultiTerm(
        ODETerm(lambda t, x, _: f_bar(x, t, use_analytical=not learned)),
        ControlTerm(lambda t, x, _: sigma(x, t), brownian_motion)
    )
    solver = Euler()
    saveat = SaveAt(steps=True)
    sol = diffeqsolve(terms, solver, 0, 1, dt0=dt, y0=jnp.ones(2) * 2, saveat=saveat, max_steps=int(1 / dt) + 1)

    plt.plot(*sol.ys.T, linewidth=1, alpha=0.6)
    plt.scatter(*sol.ys[-1], alpha=1)
    print(sol.ys[-1])

plt.savefig(f'bbb_{"learned" if learned else "analytical"}.png', dpi=600)

plt.figure()
n = 20
xs = np.linspace(-2, 4, n)
ys = np.linspace(-2, 4, n)
xx, yy = np.meshgrid(xs, ys)
s = np.stack((xx.flatten(), yy.flatten())).T
with torch.no_grad():
    u, v = model(torch.tensor(s).float(), torch.ones(n**2).float()).T.reshape(2, n, n)
plt.contourf(xx, yy, np.sqrt(u**2 + v**2), levels=50)
plt.colorbar()
plt.quiver(xs, ys, u, v)
plt.savefig('vector.png', dpi=600)
