from typing import Iterable, Generic, TypeVar

# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

from abc import ABC, abstractmethod
import warnings
import torch
import pennylane as qml

Tensor: TypeAlias = torch.Tensor
Loss: TypeAlias = torch.nn.Module
Model: TypeAlias = qml.QNode
Params = Iterable[Tensor]
Data: TypeAlias = torch.utils.data.DataLoader
P = TypeVar("P")
L = TypeVar("L")
M = TypeVar("M")
D = TypeVar("D")


class Optimizer(ABC, Generic[P, L, M, D]):
    """
    Abstract base class for optimization algorithms.

    Args:
        params (P): Parameters to optimize.
        loss_fn (L): Loss function to minimize.
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
        params: P,
        loss_fn: L,
        num_epochs: int,
        opt_stop: float,
        stagnation_threshold: float,
        stagnation_count: int,
        best_loss: bool,
    ) -> None:
        self.params = params
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.opt_stop = opt_stop
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_count = stagnation_count
        self.best_loss = best_loss

    @abstractmethod
    def optimize(self, model: M, data: D) -> Params:
        """
        Optimize model parameters using the given data.

        Args:
            model (M): The model to optimize.
            data (D): The data used to optimize the model.

        Returns:
            Params: The optimized parameters.
        """
        pass


class AdamTorch(Optimizer[Params, Loss, Model, Data]):
    """
    Wrapper for torch Adam.

    Args:
        params (Params): Parameters to optimize.
        loss_fn (Loss): Loss function to minimize.
        lr (float, optional): Learning rate. Defaults to 0.1.
        amsgrad (bool, optional): Use AMSGrad variant of Adam. Defaults to False.
        kwargs: Additional keyword arguments for base class and torch.optim.Adam.
            For Optimizer:
                - num_epochs (int, optional): Defaults to 300.
                - opt_stop (float, optional): Defaults to 1e-16.
                - stagnation_threshold (float, optional): Defaults to 0.01.
                - stagnation_count (int, optional): Defaults to 100.
                - best_loss (bool, optional): Defaults to True.
            Other keywords passed to torch.optim.Adam.
    """

    def __init__(
        self,
        params: Params,
        loss_fn: Loss,
        lr=0.1,
        amsgrad=False,
        **kwargs,
    ) -> None:
        base_kwargs = {
            "num_epochs": kwargs.pop("num_epochs", 300),
            "opt_stop": kwargs.pop("opt_stop", 1e-16),
            "stagnation_threshold": kwargs.pop("stagnation_threshold", 1e-02),
            "stagnation_count": kwargs.pop("stagnation_count", 100),
            "best_loss": kwargs.pop("best_loss", True),
        }
        super().__init__(params=params, loss_fn=loss_fn, **base_kwargs)
        self.lr = lr
        self.amsgrad = amsgrad
        self.kwargs = kwargs

    def optimize(self, model: Model, data: Data) -> Params:
        """
        Optimize the parameters of a model using the Adam/Amsgrad algorithm.

        Args:
            model (Model): The model to optimize.
            data (Data): The data used to optimize the model.

        Returns:
            Params: The optimized parameters.
        """

        opt = torch.optim.Adam(
            params=self.params,
            lr=self.lr,
            amsgrad=self.amsgrad,
            **self.kwargs,
        )

        params = self.params
        prev_loss = None
        stag_counter = 0
        best_params = params
        best_loss = float("inf")
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_size = 0
            for batch_X, batch_Y in data:
                opt.zero_grad()
                Nx = batch_X.size(0)
                pred = torch.stack([model(batch_X[k], params) for k in range(Nx)])
                loss = self.loss_fn(pred, batch_Y)
                loss.backward()
                opt.step()

                total_loss += loss.item() * Nx
                total_size += Nx

            loss_val = total_loss / total_size
            if self.best_loss:
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_params = [
                        t.detach().clone().requires_grad_(t.requires_grad)
                        for t in params
                    ]
            if loss_val <= self.opt_stop:
                break

            if prev_loss is not None:
                loss_delta = (prev_loss - loss_val) / prev_loss

                if loss_delta < self.stagnation_threshold:
                    stag_counter += 1
                else:
                    stag_counter = 0

                if stag_counter >= self.stagnation_count:
                    warnings.warn(
                        f"ADAM: Stopping early at epoch {epoch}, loss not improving.\n"
                        f"loss_delta ({loss_delta}) smaller "
                        f"than threshold ({self.stagnation_threshold}) "
                        f"for more than {self.stagnation_count} iterations."
                    )

                    break

            prev_loss = loss_val

        return best_params
