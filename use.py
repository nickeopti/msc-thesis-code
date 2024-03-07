import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diffrax import (ControlTerm, Euler, MultiTerm, ODETerm, SaveAt,
                     VirtualBrownianTree, diffeqsolve)

import process
import score


def f_bar(t, y, dp: process.Diffusion, use_analytical: bool = False):
    if use_analytical:
        s = -dp.inverse_diffusion(t, y) @ (y - jnp.ones(2) * 2) / t
    else:
        s = state.apply_fn(state.params, t[None], y[None])[0]

    return dp.drift(t, y) - dp.diffusion(t, y) @ s - dp.diffusion_divergence(t, y)


dp = process.brownian_motion(jnp.eye(2))
state = score.Model.load_from_checkpoint('saved_models', dp=dp)

learned = True

key = jax.random.PRNGKey(0)
for _ in range(5):
    key, subkey = jax.random.split(key)
    dt = -0.001

    brownian_motion = VirtualBrownianTree(0, 1, tol=1e-3, shape=(2,), key=subkey)
    terms = MultiTerm(
        ODETerm(lambda t, y, _: f_bar(t, y, dp, use_analytical=not learned)),
        ControlTerm(lambda t, y, _: dp.diffusion(t, y), brownian_motion)
    )
    solver = Euler()
    saveat = SaveAt(steps=True)
    sol = diffeqsolve(terms, solver, 1., 0.001, dt0=dt, y0=jnp.ones(2) * 2, saveat=saveat, max_steps=int(-1 / dt))

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
u, v = state.apply_fn(state.params, jnp.ones(n**2), s).T.reshape(2, n, n)
plt.contourf(xx, yy, np.sqrt(u**2 + v**2), levels=50)
plt.colorbar()
plt.quiver(xs, ys, u, v)
plt.savefig('vector.png', dpi=600)
