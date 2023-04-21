from typing import List, TypeAlias
import warnings
import torch
import numpy as np
import pennylane as qml

from .optimize import Optimizer
from .datagen import DataGenCapacity

Tensor: TypeAlias = torch.Tensor
Model: TypeAlias = qml.QNode
Datagen: TypeAlias = DataGenCapacity
Opt: TypeAlias = Optimizer


def capacity(
    model: Model,
    datagen: Datagen,
    opt: Opt,
    Nmin: int,
    Nmax: int,
    Nstep: int = 1,
    early_stop: bool = True,
) -> List[int]:
    """
    Estimates the memory capacity of a model over a range of values of N.
    See arXiv:1908.01364.

    Args:
        model (Model): The model.
        datagen (Datagen): The (synthetic) data generator.
        opt (Opt): The optimizer.
        Nmin (int): The minimum value of N.
        Nmax (int): The maximum value of N, included.
        Nstep (int, optional): Step size for N. Defaults to 1.
        early_stop (bool, optional): Stops iterations early if previous
            capacity at least as large. Defaults to True.

    Returns:
        List[int]: List of capacities over the range of N.
    """

    capacities = []
    Cprev = 0
    for N in range(Nmin, Nmax + 1, Nstep):
        mre = fit_rand_labels(model, datagen, opt, N)
        m = max(int(np.log2(1.0 / mre)), 0)
        C = N * m
        capacities.append(C)

        if C <= Cprev and N != Nmax and early_stop:
            warnings.warn("Stopping early, capacity not improving.")
            break

    return capacities


def fit_rand_labels(model: Model, datagen: Datagen, opt: Opt, N: int) -> float:
    """
    Fits random labels to a model and returns the mean relative error.

    Args:
        model (Model): The model.
        datagen (Datagen): The data generation class.
        opt (Opt): The optimizer.
        N (int): The number of inputs.

    Returns:
        float: The mean relative error.
    """

    data = datagen.gen_data(N)
    X = data["X"]
    Y = data["Y"]

    mre_sample = []
    for s in range(datagen.num_samples):
        data_opt = {"X": X, "Y": Y[s]}
        params = opt.optimize(model, data_opt)
        y_pred = torch.stack([model(X[k], params) for k in range(N)])
        mre = torch.mean(torch.abs((Y[s] - y_pred) / y_pred))
        mre_sample.append(mre)

    mre_N = torch.mean(torch.tensor(mre_sample))
    return mre_N.item()
