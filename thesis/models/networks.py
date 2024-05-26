import itertools
from typing import Callable

import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

State = train_state.TrainState


class Network(nn.Module):
    activation: Callable


class Linear(Network):
    dim: int
    max_hidden_size: int

    def setup(self) -> None:
        layer_sizes = [2**i for i in range(4, 12)]
        initial = max(range(len(layer_sizes)), key=lambda i: self.dim < layer_sizes[i])
        terminal = min(range(len(layer_sizes)), key=lambda i: layer_sizes[i] <= self.max_hidden_size)

        self.network = nn.Sequential(
            list(
                itertools.chain(
                    *[(nn.Dense(dim), self.activation) for dim in layer_sizes[initial:terminal]],
                    *[(nn.Dense(dim), self.activation) for dim in layer_sizes[terminal-1:(initial-1 if initial > 0 else None):-1]],
                    [nn.Dense(self.dim)]
                )
            )
        )

    def __call__(self, x):
        return self.network(x)


class UNet(Network):
    dim: int
    reductions: int

    def setup(self) -> None:
        self.down_layers = [nn.Dense(self.dim // 2**r) for r in range(self.reductions + 1)]
        self.up_layers = [nn.Dense(self.dim // 2**r) for r in range(self.reductions - 1, -1, -1)]
        self.final_layer = nn.Dense(self.dim)

    def __call__(self, x):
        y = x
        down_values = []
        for layer in self.down_layers:
            y = nn.gelu(layer(y))
            down_values.append(y)

        z = jnp.zeros_like(y)
        for y, layer in zip(reversed(down_values), self.up_layers):
            z = nn.gelu(layer(z + y))

        return self.final_layer(z + down_values[0])


class InverseUNet(Network):
    dim: int
    max_hidden_size: int

    def setup(self) -> None:
        layer_sizes = [2**i for i in range(4, 12)]
        initial = max(range(len(layer_sizes)), key=lambda i: self.dim < layer_sizes[i])
        terminal = min(range(len(layer_sizes)), key=lambda i: layer_sizes[i] <= self.max_hidden_size)

        self.up_layers = [(nn.Dense(dim), self.activation) for dim in layer_sizes[initial:terminal]]
        self.down_layers = [(nn.Dense(dim), self.activation) for dim in layer_sizes[terminal-1:(initial-1 if initial > 0 else None):-1]]
        self.final_layer = nn.Dense(self.dim)

    def __call__(self, x):
        y = x
        up_values = []
        for layer in self.up_layers:
            y = nn.gelu(layer(y))
            up_values.append(y)

        z = jnp.zeros_like(y)
        for y, layer in zip(reversed(up_values), self.down_layers):
            z = nn.gelu(layer(z + y))

        return self.final_layer(z + up_values[0])
