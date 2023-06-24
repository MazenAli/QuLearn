from typing import Optional, Callable, Dict

# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

from enum import Enum

import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pennylane as qml


Optimizer: TypeAlias = torch.optim.Optimizer
Loss: TypeAlias = torch.nn.Module
Metric: TypeAlias = Callable
Writer: TypeAlias = SummaryWriter
Logger: TypeAlias = logging.Logger
Model: TypeAlias = qml.QNode
Loader: TypeAlias = DataLoader
Tensor: TypeAlias = torch.Tensor


class EpochType(Enum):
    """
    Enum to denote the type of epoch in the training process.

    :cvar Train: Indicates the epoch is a training epoch.
    :cvar Validate: Indicates the epoch is a validation epoch.
    """

    Train = 1
    Validate = 2


class SupervisedTrainer:
    """
    Class to handle the training of a supervised learning model.

    :param optimizer: The optimizer to be used in the training process.
    :type optimizer: Optimizer
    :param loss_fn: The loss function used for optimization.
    :type loss_fn: Loss
    :param metrics: A dictionary mapping metric names to the metric functions to be evaluated.
    :type metrics: Dict[str, Metric]
    :param num_epochs: The number of epochs to train for.
    :type num_epochs: int
    :param writer: An optional writer for logging purposes. Default is None.
    :type writer: Optional[Writer]
    :param logger: An optional logger for logging purposes. Default is None.
    :type logger: Optional[Logger]
    """

    def __init__(
        self,
        optimizer: Optimizer,
        loss_fn: Loss,
        metrics: Dict[str, Metric],
        num_epochs: int,
        writer: Optional[Writer] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.writer = writer
        self.logger = logger

    def train(self, model: Model, train_data: Loader, valid_data: Loader) -> None:
        """
        Train the given model using the provided data loaders.

        :param model: The model to be trained.
        :type model: Model
        :param train_data: The DataLoader for the training data.
        :type train_data: Loader
        :param valid_data: The DataLoader for the validation data.
        :type valid_data: Loader
        """

        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch(model, train_data, epoch)
            self.validate_epoch(model, valid_data, epoch)

    def train_epoch(self, model: Model, train_data: Loader, epoch: int = 0) -> None:
        """
        Train the model for one epoch.

        :param model: The model to be trained.
        :type model: Model
        :param train_data: The DataLoader for the training data.
        :type train_data: Loader
        :param epoch: The current epoch number. Default is 0.
        :type epoch: int
        """
        epoch_type = EpochType.Train
        self._epoch(epoch_type, model, train_data, epoch)

    def validate_epoch(self, model: Model, valid_data: Loader, epoch: int = 0) -> None:
        """
        Validate the model after an epoch of training.

        :param model: The model to be validated.
        :type model: Model
        :param valid_data: The DataLoader for the validation data.
        :type valid_data: Loader
        :param epoch: The current epoch number. Default is 0.
        :type epoch: int
        """
        epoch_type = EpochType.Validate
        self._epoch(epoch_type, model, valid_data, epoch)

    def _epoch(
        self, epoch_type: EpochType, model: Model, data: Loader, epoch: int = 0
    ) -> None:
        running_loss = 0.0
        running_metrics = {}
        for metric in self.metrics:
            running_metrics[metric] = 0.0

        if epoch_type == EpochType.Train:
            for inputs, labels in data:
                self._train_step(model, inputs, labels)

        for inputs, labels in data:
            with torch.no_grad():
                predicted = model(inputs)
                loss = self.loss_fn(predicted, labels)
                running_loss += loss.item() * len(inputs)
                for metric in self.metrics:
                    metric_val = self.metrics[metric](predicted, labels)
                    running_metrics[metric] += metric_val.item() * len(inputs)

        running_loss /= float(len(data.dataset))  # type: ignore
        for metric in self.metrics:
            running_metrics[metric] /= float(len(data.dataset))  # type: ignore

        phase = epoch_type.name
        self._log_metrics(phase, running_loss, running_metrics, epoch)

    def _train_step(self, model: Model, inputs: Tensor, labels: Tensor) -> None:
        self.optimizer.zero_grad()
        predicted = model(inputs)
        loss = self.loss_fn(predicted, labels)
        loss.backward()
        self.optimizer.step()

    def _log_metrics(
        self, phase: str, loss: float, metrics: Dict[str, float], epoch: int
    ) -> None:
        if self.writer is not None:
            self.writer.add_scalar(f"Loss/{phase}", loss, epoch)
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(
                    f"Metrics/{phase}/{metric_name}", metric_value, epoch
                )

        if self.logger is not None:
            metrics_strs = [
                f"{metric_name}: {metric_value:.4f}"
                for metric_name, metric_value in metrics.items()
            ]
            self.logger.info(
                f"{phase} - Epoch: {epoch}, Loss: {loss:.4f}, Metrics: {', '.join(metrics_strs)}"
            )
