from typing import Callable
import torch
import torch.nn


DataScoreModel = Callable[[torch.Tensor], torch.Tensor]
"""Signature to evaluate the score function `\nabla_x \log p(x)` 
with respect to the data."""


class DataNormalizer(torch.nn.Module):
    """Pre-normalize data before any other feature transform."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ScoreMatchingLoss(torch.nn.Module):
    """Base class for losses computing a score matching loss."""

    def forward(self, score_model: DataScoreModel, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
