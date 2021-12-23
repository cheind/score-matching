import dataclasses
from typing import Union

import torch

from . import types


@dataclasses.dataclass
class Rect2dCoords:
    xlim: tuple[int, int]
    ylim: tuple[int, int]
    n_x: int
    n_y: int
    X: torch.Tensor = dataclasses.field(init=False)
    Y: torch.Tensor = dataclasses.field(init=False)
    XY: torch.Tensor = dataclasses.field(init=False)

    def __post_init__(self):
        self.X = torch.linspace(self.xlim[0], self.xlim[1], self.n_x)
        self.Y = torch.linspace(self.ylim[0], self.ylim[1], self.n_y)
        U, V = torch.meshgrid(self.Y, self.X)
        self.XY = torch.stack((V, U), -1)


def scores_rect2d(
    score_model: types.DataScoreModel,
    xlim: tuple[int, int],
    ylim: tuple[int, int],
    n_x: int,
    n_y: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Compute the data scores sampled at a discrete rectangular field."""
    X = torch.linspace(xlim[0], xlim[1], n_x, device=device)
    Y = torch.linspace(ylim[0], ylim[1], n_y, device=device)
    U, V = torch.meshgrid(Y, X)
    UV = torch.stack((V, U), -1)
    scores = score_model(UV.view(-1, 2)).view(n_y, n_x, 2)
    return scores  # NxMx2


@torch.no_grad()
def integrate_scores_rect2d(
    scores: Union[torch.Tensor, types.DataScoreModel],
    xlim: tuple[int, int],
    ylim: tuple[int, int],
    n_x: int,
    n_y: int,
    c: float = 0.0,
    device: torch.device = None,
) -> torch.tensor:
    """Integrate the gradient field on a rectangular domain to reconstruct the potential function.

    This uses direct gradient integration using the trapezoidal rule, but may can be rewritten
    terms of solution to a Poisson equation if needed. The method assumes that the field is conservative. That is a potential function phi exists, such that the path integral is
    given by
        \int_a^b scores(x)dx = phi(b)-phi(a)
    independent how the path is shaped between a and b.
    """

    if not torch.is_tensor(scores):
        scores = scores_rect2d(scores, xlim, ylim, n_x, n_y, device)

    # uses direct gradient approximation. hope that u is conservative field.

    u = scores.new_zeros((n_y, n_x))
    hx = (xlim[1] - xlim[0]) / (n_x - 1)
    hy = (ylim[1] - ylim[0]) / (n_y - 1)

    tx = 0.5 * (scores[:, 1:, 0] + scores[:, :-1, 0]) * hx
    ty = 0.5 * (scores[1:, :, 1] + scores[:-1, :, 1]) * hy

    # seed
    u[0, 0] = c
    for ix in range(n_x):
        if ix > 0:
            u[0, ix] = tx[0, ix - 1] + u[0, ix - 1]
        for iy in range(1, n_y):
            u[iy, ix] = ty[iy - 1, ix] + u[iy - 1, ix]
    return u
