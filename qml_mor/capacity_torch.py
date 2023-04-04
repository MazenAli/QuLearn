from typing import List, Tuple, Optional, TypeAlias
import warnings
import torch
import numpy as np

from .train import train_torch

Tensor: TypeAlias = torch.Tensor
Optimizer: TypeAlias = torch.optim.Optimizer


def capacity(
    Nmin: int,
    Nmax: int,
    sizex: int,
    num_samples: int,
    opt: Optimizer,
    qnn_model,
    params,
    Nstep: int = 1,
    opt_steps: int = 300,
    opt_stop: float = 1e-16,
    early_stop: bool = True,
    seed: Optional[int] = None,
    cuda: bool = False,
) -> List[int]:
    """
    Estimates the memory capacity of a quantum neural network model
    over a range of values of N.
    See arXiv:1908.01364.

    Args:
        Nmin (int): The minimum value of N.
        Nmax (int): The maximum value of N, included.
        sizex (int): The size of the input feature vector.
        num_samples (int): The number of samples to use for computing the capacity.
        opt: The torch optimizer.
        qnn_model: The quantum neural network model.
        params: The model parameters.
        Nstep (int, optional): Step size for N. Defaults to 1.
        opt_steps (int, optional): The number of optimization steps.
            Defaults to 300.
        opt_stop (float, optional): The convergence threshold for the optimization.
            Defaults to 1e-16.
        early_stop (bool, optional): Stops iterations early if previous
            capacity at least as large. Defaults to True.
        seed (int, optional): The random seed to use for generating the dataset.
            Defaults to None.
        cuda (bool, optional): Set True if run on GPU. Defaults to False.

    Returns:
        List[int]: List of capacities over the range of N.
    """

    capacities = []
    Cprev = 0
    for N in range(Nmin, Nmax + 1, Nstep):
        mre = fit_labels(
            N,
            sizex,
            num_samples,
            opt,
            qnn_model,
            params,
            opt_steps,
            opt_stop,
            seed,
            cuda,
        )
        m = max(int(np.log2(1.0 / mre)), 0)
        C = N * m
        capacities.append(C)

        if C <= Cprev and N != Nmax and early_stop:
            warnings.warn("Stopping early, capacity not improving.")
            break

    return capacities


def fit_labels(
    N: int,
    sizex: int,
    num_samples: int,
    opt: Optimizer,
    qnn_model,
    params,
    opt_steps: int = 300,
    opt_stop: float = 1e-16,
    seed: Optional[int] = None,
    cuda: bool = False,
) -> float:
    """
    Fits labels to a quantum neural network model and returns the mean relative error.

    Args:
        N (int): The number of inputs.
        sizex (int): The size of the input feature vector.
        num_samples (int): The number of samples for random output labels.
        opt: The torch optimizer.
        qnn_model: The quantum neural network model.
        params: The model parameters.
        opt_steps (int, optional): The number of optimization steps.
            Defaults to 300.
        opt_stop (float, optional): The convergence threshold for the optimization.
            Defaults to 1e-16.
        seed (int, optional): The random seed to use for generating the dataset.
            Defaults to None.
        cuda (bool, optional): Set True if run on GPU. Defaults to False.

    Returns:
        float: The mean relative error.
    """

    x, y = gen_dataset(N, sizex, num_samples, seed, cuda)

    mre_sample = []
    for s in range(num_samples):

        params = train_torch(opt, qnn_model, params, x, y[s], opt_steps, opt_stop)

        y_pred = torch.stack([qnn_model(x[k], params) for k in range(N)])
        mre = torch.mean(torch.abs((y[s] - y_pred) / y_pred))
        mre_sample.append(mre)

    mre_N = torch.mean(torch.tensor(mre_sample))
    return mre_N.item()


def gen_dataset(
    N: int,
    sizex: int,
    num_samples: int = 10,
    seed: Optional[int] = None,
    scale: float = 2.0,
    shift: float = -1.0,
    cuda: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Generates a dataset of inputs x and outputs y for a QNN.

    Args:
        N (int): The number of inputs for the QNN.
        sizex (int): The size of each input.
        num_samples (int): The number of output samples to generate. Defaults to 10.
        seed (int, optional): The random seed to use for generating the dataset.
            Defaults to None.
        scale (float, optional): The re-scaling factor for uniform random numbers
            in [0,1]. Defaults to 2.0.
        shift (float, optional): The shift value for uniform random numbers [0,1].
            Defaults to -1.0.
        cuda (bool, optional): Set True if run on GPU. Defaults to False.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the input
            tensor x of shape (N, sizex) and the output
        tensor y of shape (num_samples, N).
    """

    scale = 2.0
    shift = -1.0

    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda")

    if seed is not None:
        torch.manual_seed(seed)
    x = (
        scale
        * torch.rand(N, sizex, dtype=torch.float64, device=device, requires_grad=False)
        + shift
    )
    y = (
        scale
        * torch.rand(
            num_samples, N, dtype=torch.float64, device=device, requires_grad=False
        )
        + shift
    )

    return x, y
