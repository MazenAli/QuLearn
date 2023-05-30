from typing import List, Tuple

# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import warnings
import torch
from torch.nn import Module
import numpy as np

from .trainer import Trainer
from .datagen import DataGenCapacity

Tensor: TypeAlias = torch.Tensor
Model: TypeAlias = Module
Datagen: TypeAlias = DataGenCapacity
Tr: TypeAlias = Trainer
Capacity = List[Tuple[int, float, int, int]]


def memory(
    model: Model,
    datagen: Datagen,
    trainer: Tr,
    Nmin: int,
    Nmax: int,
    Nstep: int = 1,
    early_stop: bool = True,
    stop_count: int = 1,
) -> Capacity:
    """
    Estimates the memory capacity of a model over a range of values of N.

    :param model: The model.
    :type model: Model
    :param datagen: The (synthetic) data generator.
    :type datagen: Datagen
    :param trainer: The trainer.
    :type trainer: Tr
    :param Nmin: The minimum value of N.
    :type Nmin: int
    :param Nmax: The maximum value of N, included.
    :type Nmax: int
    :param Nstep: Step size for N. Defaults to 1.
    :type Nstep: int, optional
    :param early_stop: Stops early if previous stop_count iterations
            capacity at least as large. Defaults to True.
    :type early_stop: bool, optional
    :param stop_count: See early_stop. Defaults to 1.
    :type stop_count: int, optional
    :return: List of tuples (N, mre=2^(-m), m, N*m).
    :rtype: Capacity

    .. seealso::
        arXiv:1908.01364.
    """

    capacities = []
    Cprev = 0
    count = 0
    for N in range(Nmin, Nmax + 1, Nstep):
        mre = fit_rand_labels(model, datagen, trainer, N)
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


def fit_rand_labels(model: Model, datagen: Datagen, trainer: Tr, N: int) -> float:
    """
    Fits random labels to a model and returns the mean relative error.

    :param model: The model.
    :type model: Model
    :param datagen: The data generation class.
    :type datagen: Datagen
    :param trainer: The trainer.
    :type trainer: Tr
    :param N: The number of inputs.
    :type N: int
    :return: The mean relative error.
    :rtype: float
    """

    data = datagen.gen_data(N)
    X = data["X"]
    Y = data["Y"]

    path = None
    if trainer.best_loss:
        path = f"{trainer.file_name}_bestmre"

    mre_sample = []
    for s in range(datagen.num_samples):
        loader = datagen.data_to_loader(data, s)
        trainer.train(model, loader, loader)

        if path is not None:
            state = torch.load(path)
            model.load_state_dict(state)

        y_pred = model(X)
        mre = torch.mean(torch.abs((Y[s] - y_pred) / y_pred))
        mre_sample.append(mre.item())

    mre_N = torch.mean(torch.tensor(mre_sample))
    return mre_N.item()
