import datetime
import itertools

import clu.metrics
import jax
import tqdm

import thesis.lightning
from thesis.lightning import State, checkpointers, loggers


class Trainer:
    def __init__(self, epochs: int, logger: loggers.Logger = None) -> None:
        if epochs == -1:
            self.counter = lambda: tqdm.tqdm(itertools.count(), total=float('inf'), desc='Epoch', unit='')
        elif epochs >= 0:
            self.counter = lambda: tqdm.trange(epochs, desc='Epoch', unit='')
        else:
            raise ValueError(f'{epochs=} must be either -1 or nonnegative')

        self.logger = logger if logger is not None else loggers.CSVLogger()

    def fit(
        self,
        rng,
        model: thesis.lightning.Module[State],
        train_data,
        val_data=None,
        state: State = None,
    ) -> State:
        if state is None:
            state_class: State = thesis.lightning._get_class_from_type(model.__class__)
            state = state_class.create(
                apply_fn=model.apply,
                params=model.initialise_params(rng),
                tx=model.configure_optimizers(),
            )

        checkpointer = checkpointers.Checkpointer(self.logger.path / 'checkpoints', state)

        print(datetime.datetime.now())

        try:
            t = self.counter()
            for epoch in t:
                train_loss = clu.metrics.Average.empty()
                val_loss = clu.metrics.Average.empty()

                for batch in tqdm.tqdm(train_data, desc='Training', leave=False):
                    def loss_fn(params):
                        local_state = state.replace(params=params)
                        return model.training_step(local_state, *batch)

                    grad_fn = jax.value_and_grad(loss_fn)
                    loss, grad = grad_fn(state.params)
                    state = state.apply_gradients(grads=grad)

                    train_loss = train_loss.merge(
                        train_loss.from_model_output(loss)
                    )

                if val_data is not None:
                    for batch in tqdm.tqdm(val_data, desc='Validating', leave=False):
                        loss = model.validation_step(state, *batch)

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

        model.on_fit_end(state, self.logger.path)

        return state
