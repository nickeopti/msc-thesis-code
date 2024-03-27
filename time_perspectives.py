import jax
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt

import diffusion
import process

key = jax.random.PRNGKey(1)

original = process.brownian_motion(jnp.ones((1, 1)))
ts, ys, n = diffusion.get_data(
    dp=original,
    y0=jnp.zeros((1, 1)),
    key=key,
    t0=0,
    t1=1,
    dt=0.001
)

plt.figure()
plt.plot(ts[:n].flatten(), ys[:n].flatten())
plt.savefig('perspectives/original.png')

with open('perspectives/values.csv', 'w') as f:
    f.write('process,i,t,y\n')
    f.writelines(
        (f'original,{i},{t},{y.item()}\n' for i, (t, y) in enumerate(zip(ts[:n], ys[:n])))
    )
    

y_n = ys[n]


backwards = process.Diffusion(
    d=1,
    drift=lambda t, y: y / t,
    diffusion=jnp.ones((1, 1)),
    inverse_diffusion=jnp.ones((1, 1)),
    diffusion_divergence=jnp.zeros((1, 1))
)
ts, ys, n = diffusion.get_paths(
    dp=backwards,
    y0=y_n,
    key=key,
    t0=1,
    t1=0,
    dt=-0.001
)

plt.figure()
plt.plot(ts[:n].flatten(), ys[:n].flatten())
plt.savefig('perspectives/backwards.png')

with open('perspectives/values.csv', 'a') as f:
    f.writelines(
        (f'backwards,{i},{t},{y.item()}\n' for i, (t, y) in enumerate(zip(ts[:n], ys[:n])))
    )


forwards = process.Diffusion(
    d=1,
    drift=lambda t, y: -y / (1 - t),
    diffusion=jnp.ones((1, 1)),
    inverse_diffusion=jnp.ones((1, 1)),
    diffusion_divergence=jnp.zeros((1, 1))
)
ts, ys, n = diffusion.get_paths(
    dp=forwards,
    y0=y_n,
    key=key,
    t0=0,
    t1=1,
    dt=0.001,
    brownian_tree_class=diffusion.ReverseVirtualBrownianTree
)

plt.figure()
plt.plot(ts[:n].flatten(), ys[:n].flatten())
plt.savefig('perspectives/forwards.png')

with open('perspectives/values.csv', 'a') as f:
    f.writelines(
        (f'forwards,{i},{t},{y.item()}\n' for i, (t, y) in enumerate(zip(ts[:n], ys[:n])))
    )
