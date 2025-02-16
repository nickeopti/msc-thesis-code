import argparse
import os.path
import sys

import cycler
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import selector

plt.rc('axes', prop_cycle=cycler.cycler(color=plt.colormaps.get_cmap('tab10').colors))

parser = argparse.ArgumentParser(allow_abbrev=False)

data_path = selector.get_argument(parser, 'data_path', type=str)
metadata = pd.read_csv(os.path.join(data_path, 'metadata.txt'), sep=';')
landmarks = pd.read_csv(os.path.join(data_path, 'aligned.txt'), sep=',', header=None)

species = selector.get_argument(parser, 'species', type=str, default=None)

if species is None:
    print(metadata['species'])
    sys.exit(0)

every = selector.get_argument(parser, 'every', type=int, default=1)
butterflies = [
    jnp.array(landmarks.loc[metadata['species'] == s.strip()])[0].reshape((-1, 2))[::every] * 50
    for s in species.split(',')
]

# a = jnp.array(landmarks.loc[metadata['species'] == species])[0].reshape((-1, 2)) * 50


hide_axes = selector.get_argument(parser, 'hide_axes', type=bool, default=False)

for butterfly in butterflies:
    plt.scatter(*butterfly.T, s=5)
    # for k in range(len(a)):
    #     # plt.scatter(a[k, 0], a[k, 1], color=f'C{2*k}')
    #     plt.scatter(a[k, 0], a[k, 1])
if hide_axes:
    plt.axis('off')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig('butterflies.png', dpi=600)
