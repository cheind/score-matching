from typing import Iterator, Callable
import itertools
import torch
from functools import partial

from .types import DataScoreModel, ConditionalDataScoreModel


def iterate_ula(
    score_model: DataScoreModel,
    x0: torch.Tensor,
    tau: float = 1e-2,
    n_burnin: int = 1000,
    max_score: float = 5.0,
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
        x = (
            x
            + tau * torch.clamp(score_model(x), -max_score, max_score)
            + sqrt_2tau * eps
        )

    del x.grad


def ula(
    score_model: DataScoreModel,
    x0: torch.Tensor,
    n_steps: int,
    n_burnin: int = 0,
    tau: float = 1e-2,
    max_score: float = 5.0,
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
    g = iterate_ula(score_model, x0, tau=tau, n_burnin=n_burnin, max_score=max_score)
    samples = torch.stack(list(itertools.islice(g, n_steps)), 0)
    return samples


def annealed_ula(
    score_model: ConditionalDataScoreModel,
    x0: torch.Tensor,
    sigmae: torch.Tensor,
    n_steps: int,
    tau: float = 1e-2,
    max_score: float = 5.0,
):
    x = x0
    x = x.unsqueeze(0)  # (1,M,D)
    L = len(sigmae)
    steps_per_level = torch.tensor([n_steps // L] * L)
    steps_per_level[-1] += n_steps % L
    print(steps_per_level)
    for level in range(len(sigmae)):
        tau_level = tau * sigmae[level] / sigmae[-1]
        steps_level = steps_per_level[level]
        cond = torch.tensor([level]).to(x0.device)
        sm = partial(score_model, c=cond)
        x = ula(
            sm,
            x[-1],
            n_steps=steps_level,
            n_burnin=0,
            tau=tau_level,
            max_score=max_score,
        )
    return x