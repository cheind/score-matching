import torch
import torch.nn


def batched_jacobian(net, x, noutputs):
    # Faster https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
    # Bot only suitable for certain architectures.
    # returns (B,n_in,n_out)
    x = x.unsqueeze(1)  # b, 1 ,in_dim
    n = x.size()[0]
    x = x.repeat(1, noutputs, 1)  # b, out_dim, in_dim
    x.requires_grad_(True)
    y = net(x)
    input_val = (
        torch.eye(noutputs, device=x.device)
        .reshape(1, noutputs, noutputs)
        .repeat(n, 1, 1)
    )
    bj = torch.autograd.grad(y, x, grad_outputs=input_val, create_graph=True)[0]
    return bj


def full_jacobian(
    net: torch.nn.Module, x: torch.Tensor, vectorize: bool = False
) -> torch.Tensor:
    # Slower
    # returns (B,n_in,B,n_out)
    x.requires_grad_()
    j = torch.autograd.functional.jacobian(
        net, x, create_graph=True, vectorize=vectorize
    )
    x.requires_grad_(False)
    del x.grad
    return j
