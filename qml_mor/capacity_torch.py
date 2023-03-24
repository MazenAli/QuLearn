from typing import Tuple, Optional, TypeAlias
import torch
import numpy as np

Tensor: TypeAlias = torch.Tensor


def capacity(
    Nmin: int,
    Nmax: int,
    sizex: int,
    num_samples: int,
    qnn_model,
    params,
    Nstep: int = 1,
    opt_steps: int = 300,
    opt_stop: float = 1e-16,
    seed: Optional[int] = None,
) -> int:
    """
    Estimates the memory capacity of a quantum neural network model
    over a range of values of N.
    See arXiv:1908.01364.

    Args:
        Nmin (int): The minimum value of N.
        Nmax (int): The maximum value of N.
        sizex (int): The size of the input feature vector.
        num_samples (int): The number of samples to use for computing the capacity.
        qnn_model: The quantum neural network model.
        params: The model parameters.
        Nstep (int, optional): Step size for N. Defaults to 1.
        opt_steps (int, optional): The number of optimization steps.
            Defaults to 300.
        opt_stop (float, optional): The convergence threshold for the optimization.
            Defaults to 1e-16.

    Returns:
        int: The maximum capacity over the range of N.
    """

    capacities = []
    for N in range(Nmin, Nmax, Nstep):
        mre = fit_labels(
            N, sizex, num_samples, qnn_model, params, opt_steps, opt_stop, seed
        )
        m = int(np.log2(1.0 / mre))
        C = N * m
        capacities.append(C)

    C = max(capacities)

    return C


def fit_labels(
    N: int,
    sizex: int,
    num_samples: int,
    qnn_model,
    params,
    opt_steps: int = 300,
    opt_stop: float = 1e-16,
    seed: Optional[int] = None,
) -> float:
    """
    Fits labels to a quantum neural network model and returns the mean relative error.

    Args:
        N (int): The number of inputs.
        sizex (int): The size of the input feature vector.
        num_samples (int): The number of samples for random output labels.
        qnn_model: The quantum neural network model.
        params: The model parameters.
        opt_steps (int, optional): The number of optimization steps.
            Defaults to 300.
        opt_stop (float, optional): The convergence threshold for the optimization.
            Defaults to 1e-16.

    Returns:
        float: The mean relative error.
    """

    x, y = gen_dataset(N, sizex, num_samples, seed)

    mre_sample = []
    for s in range(num_samples):

        def cost(params):
            pred = torch.tensor(
                [qnn_model(x[k], params) for k in range(N)],
                dtype=torch.float64,
                requires_grad=True,
            )
            loss = torch.nn.MSELoss()(y[s], pred)
            return loss

        opt = torch.optim.Adam(params, lr=0.1, amsgrad=True)
        for n in range(opt_steps):
            opt.zero_grad()
            loss = cost(params)
            loss.backward()
            opt.step()

            if loss <= opt_stop:
                break

        y_pred = torch.tensor(
            [qnn_model(x[k], params) for k in range(N)],
            requires_grad=False,
        )
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

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the input
            tensor x of shape (N, sizex) and the output
        tensor y of shape (num_samples, N).
    """

    scale = 2.0
    shift = -1.0

    if seed is not None:
        torch.manual_seed(seed)
    x = scale * torch.rand(N, sizex, requires_grad=False) + shift
    y = (
        scale * torch.rand(num_samples, N, dtype=torch.float64, requires_grad=False)
        + shift
    )

    return x, y
