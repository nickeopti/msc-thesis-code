import jax.numpy as jnp
import lightning
import numpy as np
import torch
import torch.utils.data

import diffusion


def f(xs, ts):
    return torch.zeros_like(xs)

def sigma_inverse(xs, ts):
    # return torch.ones_like(xs)
    return torch.eye(xs.shape[-1]).unsqueeze(0).repeat(xs.shape[0], 1, 1)


class Model(lightning.LightningModule):
    def __init__(self, delta_t: float = 0.01) -> None:
        super().__init__()

        self.delta_t = delta_t

        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 2)
        )

    def forward(self, xs: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        stacked = torch.vstack((xs.T, ts)).T
        score_prediction = self.net(stacked)
        return score_prediction / ts.reshape(-1, 1)

    def training_step(self, batch, _):
        ts, xs, v = batch
        score_prediction = self(xs[1:], ts[1:])
        loss = (score_prediction + torch.bmm(sigma_inverse(xs[:-1], ts[1:]), (xs[1:] - xs[:-1] - f(xs[:-1], ts[1:]) * self.delta_t).unsqueeze(-1)).squeeze() / self.delta_t)**2
        loss = loss.mean()

        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, _):
        ts, xs, v = batch
        score_prediction = self(xs[1:], ts[1:])
        score = -torch.bmm(sigma_inverse(xs[1:], ts[1:]), (xs[1:] - v).unsqueeze(-1)).squeeze() / ts[1:].reshape(-1, 1)

        loss = torch.mean((score_prediction - score)**2)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class Dataset:
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n
    
    def __getitem__(self, index):
        match index % 3:
            case 0:
                y0 = jnp.zeros(2)
            case 1:
                y0 = jnp.ones(2) * 2
            case 2:
                y0 = jnp.array((2, -1))

        return *diffusion.get_data(y0=y0), torch.tensor(np.asarray(y0))


if __name__ == '__main__':
    model = Model()
    train_data = Dataset(16)
    val_data = Dataset(4)

    trainer = lightning.Trainer(max_epochs=-1, log_every_n_steps=1)
    trainer.fit(
        model=model,
        train_dataloaders=train_data,
        val_dataloaders=val_data,
    )
