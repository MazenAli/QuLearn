from typing import List, Tuple

# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

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
Capacity = List[Tuple[int, float, int, int]]


def capacity(
    model: Model,
    datagen: Datagen,
    opt: Opt,
    Nmin: int,
    Nmax: int,
    Nstep: int = 1,
    early_stop: bool = True,
    stop_count: int = 1,
) -> Capacity:
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
        early_stop (bool, optional): Stops early if previous stop_count iterations
            capacity at least as large. Defaults to True.
        stop_count (int, optional): See early_stop. Defaults to 1.

    Returns:
        Capacity: List of tuples (N, mre=2^(-m), m, N*m).
    """

    capacities = []
    Cprev = 0
    count = 0
    for N in range(Nmin, Nmax + 1, Nstep):
        mre = fit_rand_labels(model, datagen, opt, N)
        m = max(int(np.log2(1.0 / mre)), 0)
        C = N * m

        capacities.append((N, mre, m, C))

        if C <= Cprev and N != Nmax and early_stop:
            count += 1
            if count >= stop_count:
                warnings.warn("Stopping early, capacity not improving.")
                break
        else:
            count = 0

        Cprev = C

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
        loader = datagen.data_to_loader(data, s)
        params = opt.optimize(model, loader)
        y_pred = torch.stack([model(X[k], params) for k in range(N)])
        mre = torch.mean(torch.abs((Y[s] - y_pred) / y_pred))
        mre_sample.append(mre.item())

    mre_N = torch.mean(torch.tensor(mre_sample))
    return mre_N.item()
