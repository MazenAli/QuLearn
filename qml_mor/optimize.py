from typing import Iterable, Generic, TypeVar, Tuple

# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

from abc import ABC, abstractmethod
import warnings
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pennylane as qml


Tensor: TypeAlias = torch.Tensor
Loss: TypeAlias = torch.nn.Module
Optimizer: TypeAlias = torch.optim.Optimizer
Writer: TypeAlias = SummaryWriter
Model: TypeAlias = qml.QNode
Params = Iterable[Tensor]
Data: TypeAlias = DataLoader
T = TypeVar("T")
W = TypeVar("W")
L = TypeVar("L")
M = TypeVar("M")
D = TypeVar("D")


class Optimize(ABC, Generic[T, L, W, M, D]):
    """
    Abstract base class for optimizing model parameters.

    Args:
        optimizer (T): Optimizer for updating parameters.
        loss_fn (L): Loss function to minimize.
        writer (W): Writer for logging.
        num_epochs (int): Maximum number of epochs.
        opt_stop (float): Stop optimization if loss is smaller
            than this value.
        stagnation_threshold (float): If relative reduction in loss
            smaller than this for stagnation_count times, stop.
        stagnation_count (int): See stagnation_threshold.
        best_loss (bool): Return parameters corresponding to best loss value.
    """

    def __init__(
        self,
        optimizer: T,
        loss_fn: L,
        writer: W,
        num_epochs: int,
        opt_stop: float,
        stagnation_threshold: float,
        stagnation_count: int,
        best_loss: bool,
    ) -> None:
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.writer = writer
        self.num_epochs = num_epochs
        self.opt_stop = opt_stop
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_count = stagnation_count
        self.best_loss = best_loss

    @abstractmethod
    def train(self, model: M, train_data: D, valid_data: D) -> float:
        """
        Optimize model parameters using the given data.

        Args:
            model (M): The model to optimize.
            train_data (D): Data used for training.
            valid_data (D): Data used for validation.

        Returns:
            float: The final loss value.
        """
        pass


class VanillaTorchOptimize(Optimize[Optimizer, Loss, Writer, Model, Data]):
    """
    Wrapper for torch Adam.

    Args:
        optimizer (Optimizer): Torch optimizer.
        loss_fn (Loss): Loss function to minimize.
        writer (Writer): Tensorboard writer for logging.
        kwargs: Additional keyword arguments for base class.
            For Optimizer:
                - num_epochs (int, optional): Defaults to 300.
                - opt_stop (float, optional): Defaults to 1e-16.
                - stagnation_threshold (float, optional): Defaults to 0.01.
                - stagnation_count (int, optional): Defaults to 100.
                - best_loss (bool, optional): Defaults to True.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        loss_fn: Loss,
        writer: Writer,
        **kwargs,
    ) -> None:
        base_kwargs = {
            "num_epochs": kwargs.pop("num_epochs", 300),
            "opt_stop": kwargs.pop("opt_stop", 1e-16),
            "stagnation_threshold": kwargs.pop("stagnation_threshold", 1e-02),
            "stagnation_count": kwargs.pop("stagnation_count", 100),
            "best_loss": kwargs.pop("best_loss", True),
        }
        super().__init__(
            optimizer=optimizer,
            loss_fn=loss_fn,
            writer=writer,
            **base_kwargs,
        )
        self.kwargs = kwargs

    def train(self, model: Model, train_data: Data, valid_data: Data) -> float:
        """
        Optimize the parameters of a model using the Adam/Amsgrad algorithm.

        Args:
            model (Model): The model to optimize.
            train_data (Data): The data used to train the model.
            valid_data (Data): The data used to validate the model.

        Returns:
            float: Final loss.
        """

        self._model = Model
        self._train_data = train_data

        prev_loss = None
        stag_counter = 0
        best_params = model.parameters()
        best_loss = float("inf")
        for epoch in range(self.num_epochs):
            total_loss, total_size = self._train_epoch(epoch)
            train_loss = total_loss / total_size

            vloss_total = 0.0
            vsize_total = 0
            for vdata in valid_data:
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vsize = len(vinputs)
                vloss = self.loss_fn(voutputs, vlabels)
                vloss_total += vloss.item() * vsize
                vsize_total += vsize
            vloss = vloss_total / vsize_total

            if self.best_loss:
                if vloss < best_loss:
                    best_loss = vloss
                    best_params = model.state_dict().copy()

            if vloss <= self.opt_stop:
                break

            if prev_loss is not None:
                loss_delta = (prev_loss - vloss) / prev_loss

                if loss_delta < self.stagnation_threshold:
                    stag_counter += 1
                else:
                    stag_counter = 0

                if stag_counter >= self.stagnation_count:
                    warnings.warn(
                        f"Stopping early at epoch {epoch}, loss not improving.\n"
                        f"loss_delta ({loss_delta}) smaller "
                        f"than threshold ({self.stagnation_threshold}) "
                        f"for more than {self.stagnation_count} iterations."
                    )

                    break

            prev_loss = vloss

            self.writer.add_scalars(
                "Training vs. validation loss",
                {"Training": train_loss, "Validation": vloss},
                epoch,
            )

        final = vloss
        if self.best_loss:
            final = best_loss
            model.load_state_dict(best_params)

        return final

    def _train_epoch(self, epoch: int) -> Tuple[float, int]:
        running_loss = 0.0
        total_size = 0
        for i, data in enumerate(self._train_data):
            inputs, labels = data
            self.optimizer.zero_grad()
            outputs = self._model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            Nx = len(inputs)
            running_loss += loss.item() * Nx
            total_size += Nx
            idx = epoch * len(self._train_data) + i + 1
            self.writer.add_scalar("Loss/train", loss.item(), idx)

        return running_loss, total_size
