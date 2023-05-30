from typing import Iterable, Generic, TypeVar, Optional

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


class Trainer(ABC, Generic[T, L, W, M, D]):
    """
    Abstract base class for optimizing model parameters.

    :param optimizer: Optimizer for updating parameters.
    :type optimizer: T
    :param loss_fn: Loss function to minimize.
    :type loss_fn: L
    :param writer: Writer for logging.
    :type writer: W
    :param num_epochs: Maximum number of epochs.
    :type num_epochs: int
    :param opt_stop: Stop optimization if loss is smaller
            than this value.
    :type opt_stop: float
    :param stagnation_threshold: If relative reduction in loss
        smaller than this for stagnation_count times, stop.
    :type stagnation_threshold: float
    :param stagnation_count: See stagnation_threshold.
    :type stagnation_count: int
    :param best_loss: Save parameters corresponding to best loss value.
    :type best_loss: bool
    :param file_name: Name of file to save best parameters.
    :type file_name: str
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
        file_name: str,
    ) -> None:
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.writer = writer
        self.num_epochs = num_epochs
        self.opt_stop = opt_stop
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_count = stagnation_count
        self.best_loss = best_loss
        self.file_name = file_name

    @abstractmethod
    def train(self, model: M, train_data: D, valid_data: D) -> float:
        """
        Optimize model parameters using the given data.

        :param model: The model to optimize.
        :type model: M
        :param train_data: Data used for training.
        :type train_data: D
        :param valid_data: Data used for validation.
        :type valid_data: D
        :return: The final loss value.
        :rtype: float
        """
        pass


class RegressionTrainer(Trainer[Optimizer, Loss, Optional[Writer], Model, Data]):
    """
    Training on labeled real-valued input and output data.

    :param optimizer: Optimizer.
    :type optimizer: Optimizer
    :param loss_fn: Loss function to minimize.
    :type loss_fn: Loss
    :param writer: Writer for logging. Defaults to None.
    :type writer: Optional[Writer]
    :param num_epochs: Defaults to 300.
    :type num_epochs: int, optional
    :param opt_stop: Training loss stopping criterion. Defaults to 1e-16.
    :type opt_stop: float, optional
    :param stagnation_threshold: Stagnation relative change
            stopping criterion. Defaults to 0.01.
    :type stagnation_threshold: float, optional
    :param stagnation_count: Stagnation length for stopping.
            Defaults to 100.
    :type stagnation_count: int, optional
    :param best_loss: Save best training loss parameters.
            Defaults to True.
    :type best_loss: bool, optional
    :param file_name: Name of file prefix for best parameters. Defaults to "model".
    :type file_name: str, optional
    """

    def __init__(
        self,
        optimizer: Optimizer,
        loss_fn: Loss,
        writer: Optional[Writer] = None,
        num_epochs: int = 300,
        opt_stop: float = 1e-16,
        stagnation_threshold: float = 1e-02,
        stagnation_count: int = 100,
        best_loss: bool = True,
        file_name: str = "model",
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            loss_fn=loss_fn,
            writer=writer,
            num_epochs=num_epochs,
            opt_stop=opt_stop,
            stagnation_threshold=stagnation_threshold,
            stagnation_count=stagnation_count,
            best_loss=best_loss,
            file_name=file_name,
        )

    def train(self, model: Model, train_data: Data, valid_data: Data) -> float:
        """
        Optimize the parameters of a model, compute training loss,
        validation loss and validation mre.

        :param model: The model to optimize.
        :type model: Model
        :param train_data: The data used to train the model.
        :type train_data: Data
        :param valid_data: The data used to validate the model.
        :type valid_data: Data
        :return: Final (or best) training loss.
        :rtype: float
        """

        self._model = model
        self._train_data = train_data

        prev_loss = None
        stag_counter = 0
        best_tparams = model.state_dict().copy()
        best_vparams = model.state_dict().copy()
        best_mreparams = model.state_dict().copy()
        best_vloss = float("inf")
        best_tloss = float("inf")
        best_mre = float("inf")
        for epoch in range(self.num_epochs):
            tloss = self._train_epoch()
            vloss_total = 0.0
            mre = 0.0
            vsize_total = 0
            for vdata in valid_data:
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vsize = len(vinputs)
                vloss = self.loss_fn(voutputs, vlabels)
                vloss_total += vloss.item() * vsize
                vsize_total += vsize
                mre_ = torch.mean(torch.abs((voutputs - vlabels) / vlabels)) * vsize
                mre += mre_.item()
            vloss = vloss_total / vsize_total
            mre /= vsize_total

            if self.best_loss:
                if vloss < best_vloss:
                    best_vloss = vloss
                    best_vparams = model.state_dict().copy()

                if tloss < best_tloss:
                    best_tloss = tloss
                    best_tparams = model.state_dict().copy()

                if mre < best_mre:
                    best_mre = mre
                    best_mreparams = model.state_dict().copy()

            if tloss <= self.opt_stop:
                break

            if prev_loss is not None:
                loss_delta = (prev_loss - tloss) / prev_loss

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

            prev_loss = tloss

            if self.writer is not None:
                self.writer.add_scalars(
                    "loss", {"training": tloss, "valiadation": vloss}, epoch
                )
                self.writer.add_scalar("mre", mre, epoch)

        final = tloss
        if self.best_loss:
            final = best_tloss

            model.load_state_dict(best_tparams)
            model_patht = f"{self.file_name}_besttrain"
            model_pathv = f"{self.file_name}_bestval"
            model_pathmre = f"{self.file_name}_bestmre"
            torch.save(best_tparams, model_patht)
            torch.save(best_vparams, model_pathv)
            torch.save(best_mreparams, model_pathmre)

        return final

    def _train_epoch(self) -> float:
        """"""
        total_loss = 0.0
        total_size = 0
        for data in self._train_data:
            inputs, labels = data
            self.optimizer.zero_grad()
            outputs = self._model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            Nx = len(inputs)
            total_loss += loss.item() * Nx
            total_size += Nx

        total_loss /= total_size

        return total_loss
