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

    Args:
        model (Model): The model.
        datagen (Datagen): The (synthetic) data generator.
        trainer (Tr): The trainer.
        dmin (int): Iteration start for dimension check.
        dmax (int): Iteration stop for dimension check (including).
        gamma (float, optional): The margin value.
            Defaults to 0.0 (pseudo-dim).
        gamma_fac (float, optional): Additional multiplicative factor
            to increase margin. Defaults to 1.0.
        dstep (int, optional): Dimension iteration step size.
            Defaults to 1.

    Returns:
        int: The estimated fat-shattering dimension.
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

    Args:
        model (Model): The model.
        datagen (Datagen): The (synthetic) data generator.
        trainer (Tr): The trainer.
        d (int): Size of data set to shatter.
        gamma (float): The margin value.

    Returns:
        bool: True if the model shatters a random data set of size d,
            False otherwise.
    """

    data = datagen.gen_data(d)
    X = data["X"]
    b = data["b"]
    r = data["r"]

    for sr in range(len(r)):
        shattered = True
        for sb in range(len(b)):
            loader = datagen.data_to_loader(data, sr, sb)
            trainer.train(model, loader, loader)
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
    Compute a normalization constant given a tensor of weights
    and the margin parameter gamma.

    Rationale: the fat-shattering dimension of a linear classifier,
    with weights bounded by Rw and data bounded by Rx, is bounded by
    <= Rw^2*Rx^2/gamma^2. Hence, normalizing the fat-shattering dimension
    of a model with unbounded weights compares it to the best linear classifier
    with the same weight norm.

    Args:
        weights (Tensor): Tensor of weights
        gamma (float): Margin parameter.
        boundx (float): Estimated 2-radius of input data.

    Returns:
        float: A positive real-valued normalization constant.
    """

    V = torch.norm(weights, p=2)
    V = V.item()
    C = V**2 * Rx**2 / gamma**2

    return C
