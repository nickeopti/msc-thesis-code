import datetime
import itertools
import sys
from functools import partial

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

        print(model.tabulate(jax.random.key(0), *model.init_params))

        checkpointer = checkpointers.Checkpointer(self.logger.path / 'checkpoints', state)

        with open(self.logger.path / 'arguments.txt', 'x') as f:
            f.write(' '.join(sys.argv))

        training_step = model.make_training_step()
        validation_step = model.make_validation_step()

        print(datetime.datetime.now())
        key = rng

        @partial(jax.jit, static_argnames=('n',))
        def compute(key: jax.dtypes.prng_key, state: State, n: int):
            batch = jax.vmap(train_data.__getitem__)(jax.random.split(key, n))

            def loss_fn(params):
                local_state = state.replace(params=params)
                return training_step(local_state, *batch)

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grad = grad_fn(state.params)
            state = state.apply_gradients(grads=grad)

            return state, train_loss.from_model_output(loss)

        try:
            t = self.counter()
            for epoch in t:
                train_loss = clu.metrics.Average.empty()
                val_loss = clu.metrics.Average.empty()

                key, subkey = jax.random.split(key)

                state, loss = compute(subkey, state, len(train_data))
                train_loss = train_loss.merge(loss)

                if val_data is not None:
                    for batch in tqdm.tqdm(val_data, desc='Validating', leave=False):
                        loss = validation_step(state, *batch)

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
