import math

import lightning
import torch
import torch.utils.data

import diffusion


def f(xs, ts):
    return torch.zeros_like(xs)

def sigma_inverse(xs, ts):
    return torch.ones_like(xs)


class Wrapper(torch.nn.Module):
    def __init__(self, module) -> None:
        super().__init__()

        self.module = module

    def forward(self, batch):
        xs, ts = batch
        ys = self.module(xs)
        return ys, ts


class PositionalEncoding(torch.nn.Module):
    def forward(self, batch) -> torch.Tensor:
        xs, ts = batch
        _, dim = xs.shape

        self.encoding = torch.zeros_like(xs)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(100.0) / dim))
        self.encoding[:, 0::2] = torch.sin(ts.unsqueeze(1) * div_term)
        if self.encoding.shape[1] > 1:
            self.encoding[:, 1::2] = torch.cos(ts.unsqueeze(1) * div_term)

        return xs + self.encoding, ts


class Model(lightning.LightningModule):
    def __init__(self, delta_t: float = 0.01) -> None:
        super().__init__()

        self.delta_t = delta_t

        self.net = torch.nn.Sequential(
            Wrapper(torch.nn.Linear(1, 16)),
            Wrapper(torch.nn.ELU()),
            PositionalEncoding(),
            Wrapper(torch.nn.Linear(16, 32)),
            Wrapper(torch.nn.ELU()),
            PositionalEncoding(),
            Wrapper(torch.nn.Linear(32, 16)),
            Wrapper(torch.nn.ELU()),
            PositionalEncoding(),
            Wrapper(torch.nn.Linear(16, 1))
        )

    def training_step(self, batch, _):
        ts, xs = batch
        score_prediction, _ = self.net((xs[1:].view(-1, 1), ts[1:]))
        loss = (score_prediction.view(-1) + sigma_inverse(xs[:-1], ts[1:]) * (xs[1:] - xs[:-1] - f(xs[:-1], ts[1:]) * self.delta_t) / self.delta_t)**2
        loss = loss.mean()

        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, _):
        ts, xs = batch
        score_prediction, _ = self.net((xs[1:].view(-1, 1), ts[1:]))
        score = -sigma_inverse(xs[1:], ts[1:]) * xs[1:] / ts[1:]

        loss = torch.mean((score_prediction.view(-1) - score)**2)
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
        return diffusion.get_data()


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
