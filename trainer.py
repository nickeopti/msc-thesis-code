import abc
import csv
import datetime
import itertools
import pathlib

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
    def validation_step(params, state, *args):
        ...

    def on_fit_end(self, params, state, log_path):
        ...

    @abc.abstractmethod
    def configure_optimizers(self):
        ...

    @classmethod
    def load_from_checkpoint(cls, path, /, **kwargs) -> train_state.TrainState:
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


class Logger:
    def __init__(self, save_dir: str = 'logs', name: str = 'default', version: int = None) -> None:
        self.save_dir = pathlib.Path(save_dir)
        self.name = name
        self._version = version

        self._has_written_header = False

    def log(self, epoch: int, train_loss, validation_loss):
        with open(self.path / 'metrics.csv', 'a' if self._has_written_header else 'w') as f:
            writer = csv.DictWriter(f, ('epoch', 'train', 'validation'))

            if not self._has_written_header:
                writer.writeheader()
                self._has_written_header = True

            writer.writerow(
                {
                    'epoch': epoch,
                    'train': train_loss,
                    'validation': validation_loss,
                }
            )

    @property
    def path(self) -> pathlib.Path:
        return self.save_dir / self.name / f'version_{self.version}'

    @property
    def version(self):
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self) -> int:
        root_dir = self.save_dir / self.name

        if not root_dir.is_dir():
            return 0

        existing_versions = []
        for d in root_dir.iterdir():
            if d.is_dir() and d.name.startswith('version_'):
                existing_versions.append(int(d.name.split('_')[-1]))

        return max(existing_versions, default=-1) + 1


class Checkpointer:
    def __init__(self, path: pathlib.Path, state=None) -> None:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        if state is not None:
            self.save_args = {
                'save_args': orbax_utils.save_args_from_target(state)
            }
        else:
            self.save_args = {}

        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            path.resolve(),
            orbax_checkpointer,
            orbax.checkpoint.CheckpointManagerOptions(
                save_interval_steps=5,
                max_to_keep=5,
                create=True,
            ),
        )

    def save(self, epoch, state, **kwargs) -> None:
        self.checkpoint_manager.save(epoch, state, save_kwargs=self.save_args, **kwargs)


class Trainer:
    def __init__(self, epochs: int, logger: Logger = None) -> None:
        if epochs == -1:
            self.counter = lambda: tqdm.tqdm(itertools.count(), total=float('inf'), desc='Epoch', unit='')
        elif epochs >= 0:
            self.counter = lambda: tqdm.trange(epochs, desc='Epoch', unit='')
        else:
            raise ValueError(f'{epochs=} must be either -1 or nonnegative')

        self.logger = logger if logger is not None else Logger()

    def fit(
        self,
        rng,
        model: Module,
        train_data,
        val_data=None,
        state: train_state.TrainState = None,
    ):
        if state is None:
            state = train_state.TrainState.create(
                apply_fn=model.apply,
                params=model.initialise_params(rng),
                tx=model.configure_optimizers(),
            )

        checkpointer = Checkpointer(self.logger.path / 'checkpoints', state)

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

                tl = train_loss.compute()
                vl = val_loss.compute()
                self.logger.log(epoch, tl, vl)
                t.set_postfix(train=tl, val=vl, version=self.logger.version)

                checkpointer.save(epoch, state)
        except KeyboardInterrupt:
            print('Detected keyboard interrupt. Attempting graceful shutdown...')

        checkpointer.save(epoch, state, force=True)

        print(datetime.datetime.now())

        model.on_fit_end(state.params, state, self.logger.path)

        return state
