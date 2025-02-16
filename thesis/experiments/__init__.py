import copy
from typing import Callable, Optional

import jax
import jax.numpy as jnp

PathsVisualiser = Callable
FieldVisualiser = Callable


class Constraints:
    initial: jax.Array
    terminal: jax.Array

    visualise_paths: Optional[PathsVisualiser] = None
    visualise_field: Optional[FieldVisualiser] = None
    visualise_combination: Optional[Callable] = None

    def __init__(self, initial: jax.Array, terminal: jax.Array) -> None:
        assert initial.shape == terminal.shape

        self.initial = initial
        self.terminal = terminal

    @property
    def shape(self) -> tuple[int, ...]:
        return self.initial.shape

    def reversed(self):
        c = copy.copy(self)
        c.initial, c.terminal = c.terminal, c.initial
        return c


class ConstraintsCollection(Constraints):
    def __init__(self, *initials: jax.Array) -> None:
        initials: jax.Array = jnp.array(initials)
        self.mean = initials.mean(axis=0, keepdims=True)
        self.sd = initials.std(axis=0)

        initials = jnp.vstack((initials, self.mean))

        super().__init__(initials[0], self.mean[0])

        self.initials = initials

    def __len__(self) -> int:
        return len(self.initials)
