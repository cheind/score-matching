from typing import Type
import torch
import torch.nn
import torch.nn.functional as F
import pytorch_lightning as pl

from . import losses


class ToyScoreModel(pl.LightningModule):
    def __init__(
        self,
        loss: losses.ScoreMatchingLoss,
        n_input: int = 2,
        n_hidden: int = 64,
        n_hidden_layers: int = 2,
    ):
        super().__init__()
        layers = [self._layer(n_input, n_hidden, bn=False, dp=0.01, act=torch.nn.ELU)]
        layers += [
            self._layer(n_hidden, n_hidden, bn=False, dp=0.01, act=torch.nn.ELU)
            for _ in range(n_hidden_layers - 1)
        ]
        layers += [self._layer(n_hidden, n_input)]
        self.layers = torch.nn.Sequential(*layers)
        print(layers)
        self.loss_fn = loss
        self.save_hyperparameters()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        del batch_idx
        x = batch
        loss = self.loss_fn(self, x)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def _layer(
        self,
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
