import torch
import torch.nn

from . import types


class NoOp(types.DataNormalizer):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MinMaxNormalizer(types.DataNormalizer):
    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        super().__init__()
        self.register_buffer("low", torch.atleast_2d(torch.as_tensor(low)))
        self.register_buffer("high", torch.atleast_2d(torch.as_tensor(high)))
        self.register_buffer("scale", 1.0 / (self.high - self.low))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.low) * self.scale
        return x * 2.0 - 1.0


class Standardizer(types.DataNormalizer):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", torch.atleast_2d(torch.as_tensor(mean)))
        self.register_buffer("inv_std", 1.0 / torch.atleast_2d(torch.as_tensor(std)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * self.inv_std

    @staticmethod
    def estimate(x: torch.Tensor) -> "Standardizer":
        return Standardizer(x.mean(0, keepdim=True), x.std(0, keepdim=True))
