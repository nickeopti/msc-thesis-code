import jax
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt

import thesis.processes.diffusion as diffusion
import thesis.processes.process as process

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
    drift=lambda t, y: y / t,
    diffusion=original.diffusion,
    inverse_diffusion=original.inverse_diffusion,
    diffusion_divergence=original.diffusion_divergence,
)
ts, ys, n = diffusion.get_data(
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
    drift=lambda t, y: -y / (1 - t),
    diffusion=original.diffusion,
    inverse_diffusion=original.inverse_diffusion,
    diffusion_divergence=original.diffusion_divergence,
)
ts, ys, n = diffusion.get_data(
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
