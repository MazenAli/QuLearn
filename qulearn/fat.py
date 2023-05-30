# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import warnings
import torch
import pennylane as qml

from .datagen import DataGenFat
from .trainer import Trainer

Tensor: TypeAlias = torch.Tensor
Model: TypeAlias = qml.QNode
Datagen: TypeAlias = DataGenFat
Tr: TypeAlias = Trainer


def fat_shattering_dim(
    model: Model,
    datagen: Datagen,
    trainer: Tr,
    dmin: int,
    dmax: int,
    gamma: float = 0.0,
    dstep: int = 1,
) -> int:
    """
    Estimate the fat-shattering dimension for a model with a given architecture.

    :param model: The model.
    :type model: Model
    :param datagen: The (synthetic) data generator.
    :type datagen: Datagen
    :param trainer: The trainer.
    :type trainer: Tr
    :param dmin: Iteration start for dimension check.
    :type dmin: int
    :param dmax: Iteration stop for dimension check (including).
    :type dmax: int
    :param gamma: The margin value. Defaults to 0.0 (pseudo-dim).
    :type gamma: float, optional
    :param dstep: Dimension iteration step size. Defaults to 1.
    :type dstep: int
    :return: The estimated fat-shattering dimension.
    :rtype: int
    """

    for d in range(dmin, dmax + 1, dstep):
        shattered = check_shattering(model, datagen, trainer, d, gamma)

        if not shattered:
            if d == dmin:
                warnings.warn(f"Stopped at dmin = {dmin}.")
                return 0

            return d - 1

    warnings.warn(f"Reached dmax = {dmax}.")
    return dmax


def check_shattering(
    model: Model, datagen: Datagen, trainer: Tr, d: int, gamma: float
) -> bool:
    """
    Check if the model shatters a given dimension d with margin value gamma.

    :param model: The model.
    :type model: Model
    :param datagen: The (synthetic) data generator.
    :type datagen: Datagen
    :param trainer: The trainer.
    :type trainer: Tr
    :param d: Size of data set to shatter.
    :type d: int
    :param gamma: The margin value.
    :type gamma: float
    :return: True if the model shatters a random data set of size d, False otherwise.
    :rtype: bool
    """

    data = datagen.gen_data(d)
    X = data["X"]
    b = data["b"]
    r = data["r"]

    path = None
    if trainer.best_loss:
        path = f"{trainer.file_name}_bestmre"

    for sr in range(len(r)):
        shattered = True
        for sb in range(len(b)):
            loader = datagen.data_to_loader(data, sr, sb)
            trainer.train(model, loader, loader)

            if path is not None:
                state = torch.load(path)
                model.load_state_dict(state)

            predictions = model(X)

            for i, pred in enumerate(predictions):
                if b[sb, i] == 1 and not (pred >= r[sr, i] + gamma):
                    shattered = False
                    break
                if b[sb, i] == 0 and not (pred <= r[sr, i] - gamma):
                    shattered = False
                    break

            if not shattered:
                break

        if shattered:
            return True

    return False


def normalize_const(weights: Tensor, gamma: float, Rx: float) -> float:
    """
    Compute a normalization constant given a tensor of weights and
    the margin parameter gamma.

    Rationale: the fat-shattering dimension of a linear classifier,
    with weights bounded by Rw and data bounded by Rx, is bounded by
    <= Rw^2*Rx^2/gamma^2. Hence, normalizing the fat-shattering dimension
    of a model with unbounded weights compares it to the best linear classifier
    with the same weight norm.

    :param weights: Tensor of weights
    :type weights: Tensor
    :param gamma: Margin parameter.
    :type gamma: float
    :param Rx: Estimated 2-radius of input data.
    :type Rx: float
    :return: A positive real-valued normalization constant.
    :rtype: float
    """

    V = torch.norm(weights, p=2)
    V = V.item()
    C = V**2 * Rx**2 / gamma**2

    return C
