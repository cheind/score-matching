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
    ):
        super().__init__()
        self.w1 = torch.nn.Linear(n_input, n_hidden)
        self.w2 = torch.nn.Linear(n_hidden, n_hidden)
        self.w3 = torch.nn.Linear(n_hidden, n_input)
        self.loss_fn = loss
        self.save_hyperparameters()

    def forward(self, x):
        x = F.softplus(self.w1(x))
        x = F.softplus(self.w2(x))
        return self.w3(x)

    def training_step(self, batch, batch_idx):
        del batch_idx
        x = batch
        loss = self.loss_fn(self, x)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
