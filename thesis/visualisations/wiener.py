import jax
import jax.numpy as jnp

from thesis.experiments import constraints, diffusion_processes, simulators

key = jax.random.key(1)

c = constraints.PointConstraints(initial=jnp.zeros(1), terminal=None)
d = diffusion_processes.Brownian1D(1)
s = simulators.LongSimulator()

with open('wiener.csv', 'w') as f:
    f.write('i,t,y\n')
    for i in range(10):
        key, subkey = jax.random.split(key)
        f.write(f'{i},0,{c.initial.item()}\n')
        f.writelines(
            (
                f'{i},{t},{y.item()}\n'
                for t, y
                in zip(*s.simulate_sample_path(subkey, d.dp, c.initial, 0, 1, 0.001))
            )
        )
