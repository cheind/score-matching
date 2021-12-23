import torch
import matplotlib.pyplot as plt

from .types import DataScoreModel
from . import utils


def draw_scores_rect2d(
    scores: torch.Tensor,
    rect2d: utils.Rect2dCoords,
    ax=None,
    **quiver_kwargs,
):
    ax = ax or plt.gca()
    scores = scores.detach().cpu()

    kwargs = dict(
        angles="xy",
        scale_units="xy",
    )
    kwargs.update(quiver_kwargs)

    ax.quiver(
        rect2d.X,
        rect2d.Y,
        scores[..., 0],
        scores[..., 1],
        **kwargs,
    )


def set_axis_aspect(ax, rect2d: utils.Rect2dCoords):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(rect2d.xlim)
    ax.set_ylim(rect2d.ylim)


def draw_samples_rect2d(
    samples: torch.Tensor,
    rect2d: utils.Rect2dCoords,
    ax=None,
    **hist2d_kwargs,
):
    ax = ax or plt.gca()
    samplesnp = samples.detach().cpu().numpy()

    kwargs = dict(
        cmap="viridis",
        rasterized=False,
        bins=128,
        alpha=0.8,
        range=[rect2d.xlim, rect2d.ylim],
    )
    kwargs.update(hist2d_kwargs)

    ax.hist2d(
        samplesnp[:, 0],
        samplesnp[:, 1],
        **kwargs,
    )
