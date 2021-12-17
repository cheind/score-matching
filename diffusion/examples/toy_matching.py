import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn
import torch.optim
import torch.utils.data
from torch.distributions.distribution import Distribution
from torch.utils.data.dataloader import DataLoader

from .. import losses, models


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
    model = models.ToyScoreModel(loss=losses.ISMLoss(), n_input=2, n_hidden=64)
    trainer.fit(model, train_dataloaders=dl)
    trainer.save_checkpoint("tmp/score_model.ckpt")
    return model


def load(path):
    return models.ToyScoreModel.load_from_checkpoint(path)


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
    axs[1].set_xlim([-3, 3])
    axs[1].set_ylim([-3, 3])

    plt.show()

    # ------------------------------------------------------

    fig, axs = plt.subplots(1, 2)
    from ..langevin import ula

    x0 = torch.rand(5000, 2) * 6 - 3.0
    n_steps = 20000
    with torch.no_grad():
        samples = ula(model, x0.cuda(), n_steps=n_steps, tau=1e-2, n_burnin=n_steps - 1)
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