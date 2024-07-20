import torch

from .datagen import DataGenRademacher as Datagen
from .loss import RademacherLoss
from .trainer import SupervisedTrainer as Trainer
from .types import QModel, Tensor


def rademacher(
    model: QModel, trainer: Trainer, X: Tensor, sigmas: Tensor, datagen: Datagen
) -> Tensor:
    """
    Estimate Rademacher complexity of a given model.

    :param model: Prediction model.
    :type model: QModel
    :param trainer: The trainer.
    :type trainer: Trainer
    :param X: Data tensor of size (num_data_samples, size_data_set, dim_feature)
    :type X: Tensor
    :param sigmas: Sigmas tensor of size (num_sigma_samples, size_data_set)
    :type sigmas: Tensor
    :param datagen: Datagen object for converting to loader.
    :type datagen: Datagen
    :return: Scalar-valued tensor with Rademacher complexity.
    :rtype: Tensor
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
            with torch.no_grad():
                predictions = model(X_)
                sum += -loss_fn(predictions)

    sum /= num_data_samples * num_sigma_samples

    return sum
