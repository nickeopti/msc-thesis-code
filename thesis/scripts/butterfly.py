import argparse
import os.path
import sys

import cycler
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import selector

plt.rc('axes', prop_cycle=cycler.cycler(color=plt.colormaps.get_cmap('tab20').colors))

parser = argparse.ArgumentParser(allow_abbrev=False)

data_path = selector.get_argument(parser, 'data_path', type=str)
metadata = pd.read_csv(os.path.join(data_path, 'metadata.txt'), sep=';')
landmarks = pd.read_csv(os.path.join(data_path, 'aligned.txt'), sep=',', header=None)

species = selector.get_argument(parser, 'species', type=str, default=None)

if species is None:
    print(metadata['species'])
    sys.exit(0)

a = jnp.array(landmarks.loc[metadata['species'] == species])[0].reshape((-1, 2)) * 50

every = selector.get_argument(parser, 'every', type=int, default=1)

hide_axes = selector.get_argument(parser, 'hide_axes', type=bool, default=False)

for k in range(len(a[::every])):
    plt.scatter(a[::every][k, 0], a[::every][k, 1], color=f'C{2*k}')
if hide_axes:
    plt.axis('off')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig('butterfly.png', dpi=600)
