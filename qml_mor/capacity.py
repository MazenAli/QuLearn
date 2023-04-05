from typing import List, TypeAlias
import warnings
import torch
import numpy as np
import pennylane as qml

from .optimize import Optimizer
from .datagen import DataGenTorch

Tensor: TypeAlias = torch.Tensor
Model: TypeAlias = qml.QNode


def capacity(
    Nmin: int,
    Nmax: int,
    num_samples: int,
    datagen: DataGenTorch,
    opt: Optimizer,
    model: Model,
    Nstep: int = 1,
    early_stop: bool = True,
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
        model: The model.
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
        mre = fit_rand_labels(N, num_samples, datagen, opt, model)
        m = max(int(np.log2(1.0 / mre)), 0)
        C = N * m
        capacities.append(C)

        if C <= Cprev and N != Nmax and early_stop:
            warnings.warn("Stopping early, capacity not improving.")
            break

    return capacities


def fit_rand_labels(
    N: int,
    num_samples: int,
    datagen: DataGenTorch,
    opt: Optimizer,
    model: Model,
) -> float:
    """
    Fits random labels to a model and returns the mean relative error.

    Args:
        N (int): The number of inputs.
        num_samples (int): The number of samples for random output labels.
        datagen (DataGenTorch): The data generation class.
        opt (Optimizer): The optimizer.
        model: The model.

    Returns:
        float: The mean relative error.
    """

    data = datagen.gen_data(N=N, num_samples=num_samples)
    X = data["X"]
    Y = data["Y"]

    mre_sample = []
    for s in range(num_samples):
        data_opt = {"X": X, "Y": Y[s]}
        params = opt.optimize(model, data_opt)
        y_pred = torch.stack([model(X[k], params) for k in range(N)])
        mre = torch.mean(torch.abs((Y[s] - y_pred) / y_pred))
        mre_sample.append(mre)

    mre_N = torch.mean(torch.tensor(mre_sample))
    return mre_N.item()
