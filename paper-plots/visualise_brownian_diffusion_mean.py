import jax.numpy as jnp
import matplotlib.pyplot as plt

with open('brownian_2d_points_collection.npy', 'rb') as f:
    paths = jnp.load(f)

with open('diffusion_mean_estimates.npy', 'rb') as f:
    estimates = jnp.load(f)

plt.figure()
for path in paths:
    plt.plot(*path.T, linewidth=0.2, color='#666666', zorder=-1)

plt.scatter([0], [0], color='black', marker='o')
plt.scatter(*paths[:, -1].T, color='black', marker='*')

print(estimates.shape)
plt.plot(*estimates[:, 0].T, linewidth=2, color='#901A1E')
plt.scatter(*estimates[1:-1, 0].T, color='#901A1E', s=20, marker='x', zorder=0)
plt.scatter(*estimates[0, 0].T, color='#901A1E', marker='o')
plt.scatter(*estimates[-1, 0].T, color='#901A1E', s=40, marker='*', zorder=2)
plt.scatter(*paths[:, -1].mean(axis=0).T, color='blue', s=60, marker='^', zorder=1)


plt.gca().set_aspect('equal')
plt.axis('off')
# plt.box(False)
plt.tight_layout()
plt.savefig('brownian_diffusion_mean.png', dpi=600)
plt.close()
