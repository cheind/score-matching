from typing import Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn
import torch.optim
import torch.utils.data
from torch.distributions.distribution import Distribution
from torch.utils.data.dataloader import DataLoader

from .. import losses, models, utils, normalizers


def create_gt_distribution():
    mix = D.Categorical(torch.tensor([1 / 5, 4 / 5]))
    comp = D.MultivariateNormal(
        loc=torch.tensor([[-5.0, -5.0], [5.0, 5.0]]),
        covariance_matrix=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ),
    )
    pi = D.MixtureSameFamily(mix, comp)
    return pi


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
        gpus=1, limit_train_batches=1000, max_epochs=1, enable_checkpointing=False
    )
    model = models.ToyScoreModel(
        loss=losses.ISMLoss(enable_fast=True),
        n_input=2,
        n_hidden=128,
        n_hidden_layers=2,
        activation=torch.nn.Softplus,
        batch_norm=False,
        lr=1e-3,
    )
    trainer.fit(model, train_dataloaders=dl)
    trainer.save_checkpoint("tmp/score_model.ckpt")
    return model


def load(path):
    return models.ToyScoreModel.load_from_checkpoint(path)


def main():
    pi = create_gt_distribution()
    # model = train(pi)
    model = load("tmp/score_model.ckpt")
    model = model.cuda().eval()

    fig, axs = plt.subplots(1, 2)
    N = 20
    X = torch.linspace(-10, 10, N)
    Y = torch.linspace(-10, 10, N)
    U, V = torch.meshgrid(X, Y)
    UV = torch.stack((V, U), -1)
    x = UV.view(-1, 2).requires_grad_()
    S_gt = torch.autograd.grad(pi.log_prob(x).sum(), x)[0]
    S_gt = S_gt.view(N, N, 2)
    print(S_gt[-1, -1])
    axs[0].quiver(
        X,
        Y,
        S_gt[..., 0],
        S_gt[..., 1],
        angles="xy",
        scale_units="xy",
    )
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_xlim([-10, 10])
    axs[0].set_ylim([-10, 10])

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
    print(S_pred[-1, -1])
    axs[1].quiver(
        X,
        Y,
        S_pred[..., 0],
        S_pred[..., 1],
        color=(1, 1, 1, 1),
        angles="xy",
        scale_units="xy",
    )

    axs[1].set_aspect("equal", adjustable="box")
    axs[1].set_xlim([-10, 10])
    axs[1].set_ylim([-10, 10])

    plt.show()

    # ------------------------------------------------------

    fig, axs = plt.subplots(1, 2)
    from ..langevin import ula

    x0 = torch.rand(5000, 2) * 20 - 10.0
    n_steps = 10000
    with torch.no_grad():
        samples = ula(model, x0.cuda(), n_steps=n_steps, tau=1e-2, n_burnin=n_steps - 1)
    samplesnp = samples[-1].detach().cpu().numpy()
    axs[1].hist2d(
        samplesnp[:, 0],
        samplesnp[:, 1],
        cmap="viridis",
        rasterized=False,
        range=[[-10, 10], [-10, 10]],
        bins=128,
        alpha=0.8,
    )
    axs[1].set_aspect("equal", adjustable="box")
    axs[1].set_xlim([-10, 10])
    axs[1].set_ylim([-10, 10])

    samples = pi.sample((5000,))
    samplesnp = samples.cpu().numpy()
    axs[0].hist2d(
        samplesnp[:, 0],
        samplesnp[:, 1],
        cmap="viridis",
        rasterized=False,
        range=[[-10, 10], [-10, 10]],
        bins=128,
        alpha=0.8,
    )
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_xlim([-10, 10])
    axs[0].set_ylim([-10, 10])
    plt.show()

    # -----------------------------------------------------
    fig, axs = plt.subplots(1, 3)
    u = utils.integrate_scores_rect2d(
        model,
        (-10, 10),
        (-10, 10),
        100,
        100,
        pi.log_prob(torch.tensor([-10, -10])),
        torch.device("cuda"),
    )
    axs[2].imshow(u.cpu(), extent=(-10, 10, -10, 10), origin="lower")

    N = 100
    M = 100
    X = torch.linspace(-10, 10, N)
    Y = torch.linspace(-10, 10, M)
    U, V = torch.meshgrid(Y, X)

    UV = torch.stack((V, U), -1)
    print(UV.shape)  # is MxNx2
    print(UV[0, 1])  # coordinate order is [x,y]
    print(UV[1, 0])

    x = UV.view(-1, 2)
    u_gt = pi.log_prob(x).view(M, N)
    axs[0].imshow(u_gt, extent=(-10, 10, -10, 10), origin="lower")

    X = torch.linspace(-10, 10, N)
    Y = torch.linspace(-10, 10, N)
    U, V = torch.meshgrid(X, Y)
    UV = torch.stack((V, U), -1)
    x = x.clone().requires_grad_()
    scores_gt = torch.autograd.grad(pi.log_prob(x).sum(), x)[0]
    scores_gt = scores_gt.view(N, N, 2)

    u_gt_int = utils.integrate_scores_rect2d(
        scores_gt,
        (-10, 10),
        (-10, 10),
        100,
        100,
        pi.log_prob(torch.tensor([-3, -3])),
        scores_gt.device,
    )
    axs[1].imshow(u_gt_int, extent=(-10, 10, -10, 10), origin="lower")

    plt.show()


if __name__ == "__main__":
    main()
