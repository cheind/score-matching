from typing import Iterator, Callable
import itertools
import torch


def iterate_langevin(
    log_data_grad: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    tau: float = 1e-2,
    burnin: int = 1000,
) -> Iterator[torch.Tensor]:
    x = x0.requires_grad_(True)
    tau = torch.as_tensor(tau)
    sqrt_2tau = torch.sqrt(2 * tau)

    for i in itertools.count(0):
        if i > burnin:
            yield x.detach()
        eps = torch.empty_like(x0).normal_(mean=0.0, std=1.0)
        x = x + tau * log_data_grad(x) + sqrt_2tau * eps
