import torch
import torch.nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.parameter import Parameter
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
        layers = [torch.nn.Linear(n_input, n_hidden)]
        layers += [
            torch.nn.Linear(n_hidden, n_hidden) for _ in range(n_hidden_layers - 1)
        ]
        self.head = torch.nn.Linear(n_hidden, n_input)
        self.layers = torch.nn.ModuleList(layers)
        self.loss_fn = loss
        self.save_hyperparameters()

    def forward(self, x):
        for lay in self.layers:
            x = F.softplus(lay(x))
        return self.head(x)

    def training_step(self, batch, batch_idx):
        del batch_idx
        x = batch
        loss = self.loss_fn(self, x)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)