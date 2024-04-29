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

    def __init__(self, initial: jax.Array, terminal: jax.Array) -> None:
        self.initial = initial
        self.terminal = terminal

    def reversed(self):
        c = copy.copy(self)
        c.initial, c.terminal = c.terminal, c.initial
        return c
