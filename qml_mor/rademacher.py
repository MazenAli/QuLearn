# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import torch
import pennylane as qml

from .datagen import DataGenRademacher
from .loss import RademacherLoss
from .trainer import Trainer

Model: TypeAlias = qml.QNode
Tensor: TypeAlias = torch.Tensor
Tr: TypeAlias = Trainer
Datagen: TypeAlias = DataGenRademacher


def rademacher(
    model: Model, trainer: Tr, X: Tensor, sigmas: Tensor, datagen: Datagen
) -> Tensor:
    """
    Estimate Rademacher complexity of a given model.

    Args:
        model (Model): Prediction model.
        trainer (Tr): The trainer.
        X (Tensor): Data tensor of size (num_data_samples, size_data_set, dim_feature)
        sigmas (Tensor): Sigmas tensor of size (num_sigma_samples, size_data_set)
        datagen (Datagen): Datagen object for converting to loader.

    Returns:
        Tensor: Scalar-valued tensor with Rademacher complexity.
    """

    num_data_samples = X.shape[0]
    num_sigma_samples = sigmas.shape[0]
    data = {"X": X, "sigmas": sigmas}

    sum = torch.tensor([0.0], requires_grad=False, device=X.device)
    for m in range(num_data_samples):
        for s in range(num_sigma_samples):
            X_ = X[m]
            sigmas_ = sigmas[s]
            loss_fn = RademacherLoss(sigmas_)
            trainer.loss_fn = loss_fn
            loader = datagen.data_to_loader(data, m)

            trainer.train(model, loader, loader)
            predictions = model(X_)

            sum += -loss_fn(predictions)

    sum /= num_data_samples * num_sigma_samples

    return sum
