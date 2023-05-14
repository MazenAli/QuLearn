# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import torch
import pennylane as qml

from .loss import RademacherLoss
from .optimize import Optimizer

Model: TypeAlias = qml.QNode
Tensor: TypeAlias = torch.Tensor
Opt: TypeAlias = Optimizer


def rademacher(model: Model, opt: Opt, X: Tensor, sigmas: Tensor) -> Tensor:
    """
    Estimate Rademacher complexity of a given model.

    Args:
        model (Model): Prediction model.
        opt (Opt): Optimization class.
        X (Tensor): Data tensor of size (num_data_samples, size_data_set, dim_feature)
        sigmas (Tensor): Sigmas tensor of size (num_sigma_samples, size_data_set)

    Returns:
        Tensor: Scalar-valued tensor with Rademacher complexity.
    """

    num_data_samples = X.shape[0]
    num_sigma_samples = sigmas.shape[0]
    d = len(X[0])
    Y = torch.zeros(d, device=X.device)

    sum = torch.tensor([0.0], requires_grad=False, device=X.device)
    for m in range(num_data_samples):
        for s in range(num_sigma_samples):
            X_ = X[m]
            sigmas_ = sigmas[s]
            loss_fn = RademacherLoss(sigmas_)
            opt.loss_fn = loss_fn
            data_opt = {"X": X_, "Y": Y}

            params = opt.optimize(model, data_opt)
            predictions = torch.stack([model(X_[k], params) for k in range(d)])

            sum += -loss_fn(predictions)

    sum /= num_data_samples * num_sigma_samples

    return sum
