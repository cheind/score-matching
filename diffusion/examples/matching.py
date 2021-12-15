import torch
from torch import autograd
import torch.distributions as D
import torch.nn
import torch.nn.functional as F
import torch.optim
import matplotlib.pyplot as plt
from torch.nn.modules.activation import ReLU


def create_gt_distribution():
    mix = D.Categorical(torch.tensor([0.5, 0.5]))
    comp = D.MultivariateNormal(
        loc=torch.tensor([[-2.0, -2.0], [2.0, 2.0]]),
        covariance_matrix=torch.tensor(
            [
                [[0.5, 0.0], [0.0, 0.5]],
                [[0.5, 0.0], [0.0, 0.5]],
            ]
        ),
    )
    pi = D.MixtureSameFamily(mix, comp)
    return pi


class ScoreFunctionModel(torch.nn.Module):
    def __init__(self, n_hidden: int = 64, dropout: float = 0.05):
        super().__init__()
        self.w1 = torch.nn.Linear(2, n_hidden)
        self.w2 = torch.nn.Linear(n_hidden, n_hidden)
        self.w3 = torch.nn.Linear(n_hidden, 2)
        self.dp1 = torch.nn.Dropout(dropout)
        self.dp2 = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.dp1(F.softplus(self.w1(x)))
        x = self.dp2(F.softplus(self.w2(x)))
        return self.w3(x)


def score_matching_loss(score_model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    B, D = x.shape
    x.requires_grad_()
    y = score_model(x)
    # faster? https://github.com/pytorch/pytorch/issues/41530
    j = torch.autograd.functional.jacobian(
        score_model, x, create_graph=True, vectorize=False
    )
    tr = j[range(B), 0, range(B), 0] + j[range(B), 1, range(B), 1]
    x.requires_grad_(False)
    del x.grad

    return (tr + 0.5 * (y ** 2).sum(-1)).mean()


def main():
    pi = create_gt_distribution()
    model = ScoreFunctionModel(n_hidden=128, dropout=0.05).cuda()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for e in range(1000):
        samples = pi.sample((128,)).cuda()
        samples = samples / 5  # ca. [-1, 1]
        opt.zero_grad()
        loss = score_matching_loss(model, samples)
        loss.backward()
        opt.step()
        if e % 100 == 0:
            print(loss.item())

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
        S_pred = model(UV.view(-1, 2).cuda() / 5).view(N, N, 2).cpu()
    axs[1].quiver(X, Y, S_pred[..., 0], S_pred[..., 1], color=(1, 1, 1, 1))

    axs[1].set_aspect("equal", adjustable="box")
    axs[1].set_xlim([-3, 3])
    axs[1].set_ylim([-3, 3])

    plt.show()


if __name__ == "__main__":
    main()