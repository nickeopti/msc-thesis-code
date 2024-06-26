import abc
import pathlib
import typing
from typing import Any, Generic, Protocol, Self, TypeVar

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
from flax import linen as nn
from flax.training import train_state

State = TypeVar('State', bound=train_state.TrainState)


class Evaluation(Protocol):
    """https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols"""

    def __call__(self, state: State, *args: Any) -> jax.Array: ...


def _get_class_from_type(cls: Generic[State]) -> State:
    # Pure Python magic; get the concrete class of a generic type
    state_class: State = typing.get_args(cls.__orig_bases__[0])[0]
    return state_class


class Module(nn.Module, abc.ABC, Generic[State]):
    @abc.abstractmethod
    def initialise_params(self, rng): ...

    @abc.abstractmethod
    def make_training_step(self) -> Evaluation: ...

    def make_validation_step(self) -> Evaluation: ...

    def on_fit_end(self, state: State, log_path: pathlib.Path) -> None: ...

    @property
    @abc.abstractmethod
    def init_params(self) -> Any: ...

    @abc.abstractmethod
    def configure_optimizers(self) -> optax.GradientTransformation: ...

    @classmethod
    def load_from_checkpoint(cls, path, /, **kwargs) -> tuple[Self, State]:
        # Don't really care about values nor randomness here,
        # as the values will anyways be overriden from the checkpoint
        key = jax.random.PRNGKey(0)

        model = cls(**kwargs)
        params = model.initialise_params(key)
        tx = model.configure_optimizers()

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint_manager = orbax.checkpoint.CheckpointManager(
            pathlib.Path(path).resolve(),
            orbax_checkpointer,
        )

        state_class: State = _get_class_from_type(cls)
        empty_state = state_class.create(
            apply_fn=model.apply,
            params=jax.tree_map(jnp.zeros_like, params),
            tx=tx,
        )

        state: State = checkpoint_manager.restore(
            checkpoint_manager.latest_step(),
            items=empty_state,
        )

        return model, state
