import sys
import os.path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import diffusion
import process
import score


def _dp(dp: process.Diffusion, use_analytical: bool = False):
    if use_analytical:
        def f_bar(t, y):
            s = -dp.inverse_diffusion(t, y) @ (y - jnp.ones(2) * 2) / t
            return dp.drift(t, y) - dp.diffusion(t, y) @ s - dp.diffusion_divergence(t, y)
    else:
        def f_bar(t, y):
            s = state.apply_fn(state.params, t[None], y[None])[0]
            return dp.drift(t, y) - dp.diffusion(t, y) @ s - dp.diffusion_divergence(t, y)

    return process.Diffusion(
        d=dp.d,
        drift=f_bar,
        diffusion=dp.diffusion,
        inverse_diffusion=None,
        diffusion_divergence=None,
    )


dp = process.brownian_motion(jnp.array([[1, 0.6], [0.6, 1]]))
# dp = process.brownian_motion(jnp.eye(2))
state = score.Model.load_from_checkpoint(
    os.path.join('logs', 'default', f'version_{sys.argv[1]}', 'checkpoints'),
    dp=dp
)

learned = True

key = jax.random.PRNGKey(1)

plt.figure()
for _ in range(5):
    key, subkey = jax.random.split(key)

    ts, ys, n = diffusion.get_data(
        dp=_dp(dp, use_analytical=not learned),
        y0=jnp.ones(dp.d) * 2,
        key=subkey,
        t0=1.,
        t1=0.001,
        dt=-0.001,
    )

    plt.plot(*ys[:n].T, linewidth=1, alpha=0.6)
    plt.scatter(*ys[n-1], alpha=1)
    print(ys[n-1])
plt.savefig(f'bbb_{"learned" if learned else "analytical"}.png', dpi=600)

plt.figure()
for _ in range(5):
    key, subkey = jax.random.split(key)

    ts, ys, n = diffusion.get_data(
        dp=dp,
        y0=jnp.ones(dp.d) * 2,
        key=subkey,
        t0=0,
        t1=1,
        dt=0.001,
    )

    plt.plot(*ys[:n].T, linewidth=1, alpha=0.6)
    plt.scatter(*ys[n-1], alpha=1)
    print(ys[n-1])
plt.savefig('unconditioned.png', dpi=600)

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
