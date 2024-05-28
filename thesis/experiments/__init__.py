import copy
from typing import Callable, Optional

import jax

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
