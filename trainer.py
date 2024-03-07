import abc
import csv
import datetime
import itertools
import pathlib
import warnings

import clu.metrics
import jax
import jax.numpy as jnp
import orbax.checkpoint
import tqdm
from flax import linen as nn
from flax.training import orbax_utils, train_state


class Module(nn.Module, abc.ABC):
    @abc.abstractmethod
    def initialise_params(self, rng):
        ...

    @staticmethod
    @abc.abstractmethod
    def training_step(params, state, *args):
        ...

    @staticmethod
    @abc.abstractmethod
    def validation_step(params, state, *args):
        ...

    @abc.abstractmethod
    def configure_optimizers(self):
        ...

    @classmethod
    def load_from_checkpoint(cls, path, **kwargs) -> train_state.TrainState:
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

        empty_state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=jax.tree_map(jnp.zeros_like, params),
            tx=tx,
        )

        state: train_state.TrainState = checkpoint_manager.restore(
            checkpoint_manager.latest_step(),
            items=empty_state,
        )

        return state


class Trainer:
    def __init__(self, epochs: int) -> None:
        if epochs == -1:
            self.counter = lambda: tqdm.tqdm(itertools.count(), total=float('inf'), desc='Epoch', unit='')
        elif epochs >= 0:
            self.counter = lambda: tqdm.trange(epochs, desc='Epoch', unit='')
        else:
            raise ValueError(f'{epochs=} must be either -1 or nonnegative')

        self.log_file = 'log.csv'

    def fit(
        self,
        rng,
        model: Module,
        train_data,
        val_data=None,
        state: train_state.TrainState = None,
        **state_args
    ):
        if state is None:
            state = train_state.TrainState.create(
                apply_fn=model.apply,
                params=model.initialise_params(rng),
                tx=model.configure_optimizers(),
            )

        with open(self.log_file, 'w') as f:
            writer = csv.DictWriter(f, ['epoch', 'train', 'val'])
            writer.writeheader()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(state)
            path = pathlib.Path('saved_models')
            checkpoint_manager = orbax.checkpoint.CheckpointManager(
                path.resolve(),
                orbax_checkpointer,
                orbax.checkpoint.CheckpointManagerOptions(
                    save_interval_steps=5,
                    max_to_keep=5,
                    create=True,
                ),
            )

        print(datetime.datetime.now())

        try:
            t = self.counter()
            for epoch in t:
                train_loss = clu.metrics.Average.empty()
                val_loss = clu.metrics.Average.empty()

                for batch in tqdm.tqdm(train_data, desc='Training', leave=False):
                    def loss_fn(params):
                        return model.training_step(params, state, *batch)

                    grad_fn = jax.value_and_grad(loss_fn)
                    loss, grad = grad_fn(state.params)
                    state = state.apply_gradients(grads=grad)

                    train_loss = train_loss.merge(
                        train_loss.from_model_output(loss)
                    )

                if val_data is not None:
                    for batch in tqdm.tqdm(val_data, desc='Validating', leave=False):
                        loss = model.validation_step(state.params, state, *batch)

                        val_loss = val_loss.merge(
                            val_loss.from_model_output(loss)
                        )

                losses = {
                    'train': train_loss.compute(),
                    'val': val_loss.compute(),
                }
                with open(self.log_file, 'a') as f:
                    writer = csv.DictWriter(f, ['epoch', 'train', 'val'])
                    writer.writerow({'epoch': epoch} | losses)

                t.set_postfix(losses)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    checkpoint_manager.save(epoch, state, save_kwargs={'save_args': save_args})
        except KeyboardInterrupt:
            print('Detected keyboard interrupt. Attempting graceful shutdown...')

        print(datetime.datetime.now())

        return state
