import pathlib

import absl.logging
import orbax.checkpoint
from flax.training import orbax_utils

# TODO: Remove once orbax has been updated
absl.logging.set_verbosity(absl.logging.ERROR)


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
        try:
            self.checkpoint_manager.save(epoch, state, save_kwargs=self.save_args, **kwargs)
        except ValueError as e:
            if 'already exists' in str(e):
                pass
            else:
                raise e
