import itertools
import math
from typing import Callable

import jax
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
        self.final_layer = nn.Dense(self.dim, use_bias=False)

    def __call__(self, x):
        y = x
        down_values = []
        for layer in self.down_layers:
            y = self.activation(layer(y))
            down_values.append(y)

        z = jnp.zeros_like(y)
        for y, layer in zip(reversed(down_values), self.up_layers):
            z = self.activation(layer(z + y))

        return self.final_layer(z + down_values[0])


class InverseUNet(Network):
    dim: int
    max_hidden_size: int

    def setup(self) -> None:
        layer_sizes = [2**i for i in range(4, 16)]
        initial = max(range(len(layer_sizes)), key=lambda i: self.dim < layer_sizes[i])
        terminal = min(range(len(layer_sizes)), key=lambda i: layer_sizes[i] <= self.max_hidden_size)

        self.up_layers = [(nn.Dense(dim), self.activation) for dim in layer_sizes[initial:terminal]]
        self.down_layers = [(nn.Dense(dim), self.activation) for dim in layer_sizes[terminal-2:(initial-1 if initial > 0 else None):-1]]
        self.final_layer = nn.Dense(self.dim, use_bias=False)

    def __call__(self, x):
        y = x
        up_values = []
        for layer, activation in self.up_layers:
            y = activation(layer(y))
            up_values.append(y)

        z = jnp.zeros_like(y)
        for y, (layer, activation) in zip(reversed(up_values), self.down_layers):
            z = activation(layer(z + y))

        return self.final_layer(z + up_values[0])


def get_time_embedding(
    embedding_dim: int, max_period: float = 128.0, scaling: float = 100.0
):
    div_term = jnp.exp(
        jnp.arange(0, embedding_dim, 2, dtype=jnp.float32)
        * (-math.log(max_period) / embedding_dim)
    )

    def time_embedding(t: float) -> jnp.ndarray:
        """Embed scalar time steps into a vector of size `embedding_dim`"""
        emb = jnp.empty((embedding_dim,), dtype=jnp.float32)
        emb = emb.at[0::2].set(jnp.sin(scaling * t * div_term))
        emb = emb.at[1::2].set(jnp.cos(scaling * t * div_term))
        return emb

    return time_embedding


class TimeEmbeddingMLP(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, t_emb: jnp.ndarray) -> tuple[jax.Array, jax.Array]:
        scale_shift = nn.Dense(
            2 * self.output_dim, kernel_init=nn.initializers.xavier_normal()
        )(t_emb)
        scale, shift = jnp.array_split(scale_shift, 2, axis=-1)
        return scale, shift


class InverseUNetVarianceEmbedding(Network):
    dim: int
    max_hidden_size: int
    variance_embedding_dim: int = 32

    def setup(self) -> None:
        layer_sizes = [2**i for i in range(4, 16)]
        initial = max(range(len(layer_sizes)), key=lambda i: self.dim < layer_sizes[i])
        terminal = min(range(len(layer_sizes)), key=lambda i: layer_sizes[i] <= self.max_hidden_size)

        self.up_layers = [(nn.Dense(dim), self.activation, TimeEmbeddingMLP(dim)) for dim in layer_sizes[initial:terminal]]
        self.down_layers = [(nn.Dense(dim), self.activation, TimeEmbeddingMLP(dim)) for dim in layer_sizes[terminal-2:(initial-1 if initial > 0 else None):-1]]
        self.final_layer = nn.Dense(self.dim, use_bias=False)

    def __call__(self, x, sigma):
        variance_embedding = get_time_embedding(self.variance_embedding_dim)
        sigma_embedding = jax.vmap(variance_embedding, in_axes=0)(sigma)

        y = x
        up_values = []
        for layer, activation, embedder in self.up_layers:
            scale, shift = embedder(sigma_embedding)
            y = activation(layer(y) * (1.0 + scale) + shift)
            up_values.append(y)

        z = jnp.zeros_like(y)
        for y, (layer, activation, embedder) in zip(reversed(up_values), self.down_layers):
            scale, shift = embedder(sigma_embedding)
            z = activation(layer(z + y) * (1.0 + scale) + shift)

        return self.final_layer(z + up_values[0])
