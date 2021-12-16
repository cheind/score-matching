import torch
from torch import autograd
import pytorch_lightning as pl
import torch.distributions as D
from torch.distributions.distribution import Distribution
import torch.nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
from torch.nn.modules.activation import ReLU
from torch.utils.data.dataloader import DataLoader


def create_gt_distribution():
    mix = D.Categorical(torch.tensor([1 / 5, 4 / 5]))
    comp = D.MultivariateNormal(
        loc=torch.tensor([[-2.0, -2.0], [2.0, 2.0]]),
        covariance_matrix=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ),
    )
    pi = D.MixtureSameFamily(mix, comp)
    return pi


def get_batch_jacobian(net, x, noutputs):
    # Faster https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
    # Bot only suitable for certain architectures.
    # returns (B,n_in,n_out)
    x = x.unsqueeze(1)  # b, 1 ,in_dim
    n = x.size()[0]
    x = x.repeat(1, noutputs, 1)  # b, out_dim, in_dim
    x.requires_grad_(True)
    y = net(x)
    input_val = (
        torch.eye(noutputs, device=x.device)
        .reshape(1, noutputs, noutputs)
        .repeat(n, 1, 1)
    )
    bj = torch.autograd.grad(y, x, grad_outputs=input_val, create_graph=True)[0]
    return bj


def get_full_jacobian(net: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Slower
    # returns (B,n_in,B,n_out)
    x.requires_grad_()
    j = torch.autograd.functional.jacobian(net, x, create_graph=True, vectorize=True)
    x.requires_grad_(False)
    del x.grad
    return j


def score_matching_loss(score_model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    B, _ = x.shape

    j = get_batch_jacobian(score_model, x, 2)
    tr = j[range(B), 0, 0] + j[range(B), 1, 1]

    # j = get_full_jacobian(score_model, x)
    # tr = j[range(B), 0, range(B), 0] + j[range(B), 1, range(B), 1]

    y = score_model(x)
    return (tr + 0.5 * (y ** 2).sum(-1)).mean()


class ScoreModel(pl.LightningModule):
    def __init__(self, n_input: int = 2, n_hidden: int = 64):
        super().__init__()
        self.w1 = torch.nn.Linear(n_input, n_hidden)
        self.w2 = torch.nn.Linear(n_hidden, n_hidden)
        self.w3 = torch.nn.Linear(n_hidden, n_input)
        self.save_hyperparameters()

    def forward(self, x):
        x = F.softplus(self.w1(x))
        x = F.softplus(self.w2(x))
        return self.w3(x)

    def training_step(self, batch, batch_idx):
        x = batch
        loss = score_matching_loss(self, x)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class DistributionDataset(torch.utils.data.IterableDataset):
    def __init__(self, pi: D.Distribution) -> None:
        self.pi = pi

    def __iter__(self):
        while True:
            yield self.pi.sample()


def train(pi: Distribution):
    ds = DistributionDataset(pi)
    dl = DataLoader(ds, batch_size=128)
    trainer = pl.Trainer(
        gpus=1, limit_train_batches=1000, max_epochs=1, checkpoint_callback=False
    )
    model = ScoreModel(n_input=2, n_hidden=64)
    trainer.fit(model, train_dataloaders=dl)
    trainer.save_checkpoint("tmp/score_model.ckpt")
    return model


def load(path):
    return ScoreModel.load_from_checkpoint(path)


def main():
    pi = create_gt_distribution()
    model = train(pi)
    model = load("tmp/score_model.ckpt")
    model = model.cuda().eval()

    fig, axs = plt.subplots(1, 2)
    N = 20
    X = torch.linspace(-3, 3, N)
    Y = torch.linspace(-3, 3, N)
    U, V = torch.meshgrid(X, Y)
    UV = torch.stack((V, U), -1)
    x = UV.view(-1, 2).requires_grad_()
    S_gt = torch.autograd.grad(pi.log_prob(x).sum(), x)[0]
    S_gt = S_gt.view(N, N, 2)
    axs[0].quiver(X, Y, S_gt[..., 0], S_gt[..., 1])
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_xlim([-3, 3])
    axs[0].set_ylim([-3, 3])

    samples = pi.sample((5000,)).cuda()
    samplesnp = samples.detach().cpu().numpy()
    axs[1].hist2d(
        samplesnp[:, 0],
        samplesnp[:, 1],
        cmap="viridis",
        rasterized=False,
        bins=128,
        alpha=0.8,
    )
    with torch.no_grad():
        model = model.cuda().eval()
        S_pred = model(UV.view(-1, 2).cuda()).view(N, N, 2).cpu()
    axs[1].quiver(
        X,
        Y,
        S_pred[..., 0],
        S_pred[..., 1],
        color=(1, 1, 1, 1),
    )

    axs[1].set_aspect("equal", adjustable="box")
    axs[1].set_xlim([-3, 3])
    axs[1].set_ylim([-3, 3])

    plt.show()

    # ------------------------------------------------------

    fig, axs = plt.subplots(1, 2)
    from ..langevin import ula

    x0 = torch.rand(5000, 2) * 6 - 3.0
    with torch.no_grad():
        samples = ula(model, x0.cuda(), n_steps=10000, tau=1e-2, n_burnin=9999)
    samplesnp = samples[-1].detach().cpu().numpy()
    axs[1].hist2d(
        samplesnp[:, 0],
        samplesnp[:, 1],
        cmap="viridis",
        rasterized=False,
        bins=128,
        alpha=0.8,
    )
    axs[1].set_aspect("equal", adjustable="box")
    axs[1].set_xlim([-3, 3])
    axs[1].set_ylim([-3, 3])

    samples = pi.sample((5000,))
    samplesnp = samples.cpu().numpy()
    axs[0].hist2d(
        samplesnp[:, 0],
        samplesnp[:, 1],
        cmap="viridis",
        rasterized=False,
        bins=128,
        alpha=0.8,
    )
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_xlim([-3, 3])
    axs[0].set_ylim([-3, 3])
    plt.show()


if __name__ == "__main__":
    main()