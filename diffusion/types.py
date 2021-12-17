from typing import Callable
import torch


DataScoreModel = Callable[[torch.Tensor], torch.Tensor]
"""Signature to evaluate the score function `\nabla_x \log p(x)` 
with respect to the data."""