from typing import Type
import torch
import torch.nn
import pytorch_lightning as pl
import torch.nn.functional as F

from . import losses, types, normalizers


def _make_layer(
    n_in: int,
    n_out: int,
    bn: bool = False,
    dp: float = 0.0,
    act: Type[torch.nn.Module] = None,
) -> torch.nn.Sequential:
    layers = [torch.nn.Linear(n_in, n_out)]
    if act is not None:
        layers.append(act())
    if bn:
        layers.append(torch.nn.BatchNorm1d(n_out))
    if dp > 0:
        layers.append(torch.nn.Dropout(p=dp))
    return torch.nn.Sequential(*layers)


class ToyScoreModel(pl.LightningModule):
    def __init__(
        self,
        loss: losses.ScoreMatchingLoss,
        norm: types.DataNormalizer = None,
        n_input: int = 2,
        n_hidden: int = 64,
        n_hidden_layers: int = 2,
        batch_norm: bool = False,
        lr: float = 1e-3,
        activation: Type[torch.nn.Module] = torch.nn.Softplus,
    ):
        super().__init__()
        layers = [
            _make_layer(n_input, n_hidden, bn=batch_norm, dp=0.01, act=activation)
        ]
        layers += [
            _make_layer(n_hidden, n_hidden, bn=batch_norm, dp=0.01, act=activation)
            for _ in range(n_hidden_layers - 1)
        ]
        layers += [_make_layer(n_hidden, n_input)]
        self.layers = torch.nn.Sequential(*layers)
        if norm is None:
            norm = normalizers.NoOp()
        self.norm = norm
        self.loss_fn = loss
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        return self.layers(self.norm(x))

    def training_step(self, batch, batch_idx):
        del batch_idx
        x = batch
        loss = self.loss_fn(self, x)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ConditionalToyScoreModel(pl.LightningModule):
    def __init__(
        self,
        loss: losses.ScoreMatchingLoss,
        n_input: int,
        n_cond: int,
        n_hidden: int = 64,
        n_hidden_layers: int = 2,
        batch_norm: bool = False,
        lr: float = 1e-3,
        activation: Type[torch.nn.Module] = torch.nn.Softplus,
        norm: types.DataNormalizer = None,
    ):
        super().__init__()
        self.cond_embed = torch.nn.Embedding(n_cond, 2)

        layers = [
            _make_layer(n_input + 2, n_hidden, bn=batch_norm, dp=0.00, act=activation)
        ]
        layers += [
            _make_layer(n_hidden + 2, n_hidden, bn=batch_norm, dp=0.01, act=activation)
            for _ in range(n_hidden_layers - 1)
        ]
        layers += [_make_layer(n_hidden + 2, n_input)]

        self.layers = torch.nn.ModuleList(layers)
        if norm is None:
            norm = normalizers.NoOp()
        self.norm = norm
        self.n_cond = n_cond
        self.loss_fn = loss
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor, c: torch.LongTensor):
        x = self.norm(x)
        e = self.cond_embed(c)
        if e.shape[0] == 1:
            e = e.repeat(x.shape[0], 1)
        for layer in self.layers:
            x = layer(torch.cat((x, e), -1))
        return x

    def training_step(self, batch, batch_idx):
        del batch_idx
        x = batch
        loss = self.loss_fn(self, x)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)