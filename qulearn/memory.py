import logging

import numpy as np
import torch

from .datagen import DataGenCapacity as Datagen
from .trainer import SupervisedTrainer as Trainer
from .types import Capacity, Model


def memory(
    model: Model,
    datagen: Datagen,
    trainer: Trainer,
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
    :type trainer: Trainer
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
                logging.basicConfig(level=logging.WARNING)
                logging.warning("Stopping early, capacity not improving.")
                break
        else:
            count = 0

        Cprev = C

    return capacities


def fit_rand_labels(model: Model, datagen: Datagen, trainer: Trainer, N: int) -> float:
    """
    Fits random labels to a model and returns the mean relative error.

    :param model: The model.
    :type model: Model
    :param datagen: The data generation class.
    :type datagen: Datagen
    :param trainer: The trainer.
    :type trainer: Trainer
    :param N: The number of inputs.
    :type N: int
    :return: The mean relative error.
    :rtype: float
    """

    data = datagen.gen_data(N)
    X = data["X"]
    Y = data["Y"]
    mre_sample = []
    for s in range(datagen.num_samples):
        loader = datagen.data_to_loader(data, s)
        trainer.train(model, loader, loader)
        y_pred = model(X)
        mre = torch.mean(torch.abs((Y[s] - y_pred) / y_pred))
        mre_sample.append(mre.item())

    mre_N = torch.mean(torch.tensor(mre_sample))
    return mre_N.item()
