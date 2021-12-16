from typing import Iterator, Callable
import itertools
import torch


def iterate_ula(
    log_data_grad: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    tau: float = 1e-2,
    n_burnin: int = 1000,
) -> Iterator[torch.Tensor]:
    """Generates samples using the Unadjusted Langevin Algorithm (ULA).

    See: https://www.icts.res.in/sites/default/files/paap-2019-08-09-Eric%20Moulines.pdf
    """
    x = x0.requires_grad_(True)
    tau = torch.as_tensor(tau)
    sqrt_2tau = torch.sqrt(2 * tau)

    eps = torch.empty_like(x0)
    for i in itertools.count(0):
        if i > n_burnin:
            yield x.detach()
        eps.normal_(mean=0.0, std=1.0)
        x = x + tau * log_data_grad(x) + sqrt_2tau * eps

    del x.grad


def ula(
    log_data_grad: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    n_steps: int,
    tau: float = 1e-2,
    n_burnin: int = 1000,
) -> torch.Tensor:
    """Returns samples generated from the Unadjusted Langevin Algorithm.

    Args:
        log_data_grad: Function that computes the log-probability gradient
            tensor (M,D) with respect to samples (M,D)
        x0: Starting seed tensor (M,D) for the discrete Langevin process
        n_steps: Number of steps to perform
        n_burnin: Number of steps to consider burnin
        tau: Step size

    Returns:
        samples: (N,M,D) tensor of samples.
    """
    g = iterate_ula(log_data_grad, x0, tau=tau, n_burnin=n_burnin)
    samples = torch.stack(list(itertools.islice(g, n_steps)), 0)
    return samples
