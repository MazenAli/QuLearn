from typing import TypeAlias, Iterable, Dict
from abc import ABC, abstractmethod
import torch
import pennylane as qml

Tensor: TypeAlias = torch.Tensor
Loss: TypeAlias = torch.nn.Module
Model: TypeAlias = qml.QNode
Params = Iterable[Tensor]
Data = Dict[str, Tensor]


class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms.

    Args:
        params (Params): Parameters to optimize.
    """

    def __init__(self, params: Params) -> None:
        self.params = params

    @abstractmethod
    def optimize(self, model: Model, data: Data) -> Params:
        """
        Optimize model parameters using the given data.

        Args:
            model (Model): The model to optimize.
            data (Data): The data used to optimize the model.

        Returns:
            Params: The optimized parameters.
        """
        pass


class AdamTorch(Optimizer):
    """
    Wrapper for torch Adam.

    Args:
        params (Params): Parameters to optimize.
        loss_fn (Loss): Loss function to minimize.
        lr (float, optional): Learning rate. Defaults to 0.1.
        amsgrad (bool, optional): Use AMSGrad variant of Adam. Defaults to False.
        opt_steps (int, optional): Maximum number of optimization steps.
            Defaults to 300.
        opt_stop (float, optional): Stop optimization if loss is smaller
            than this value. Defaults to 1e-16.
        kwargs: Additional keyword arguments for torch.optim.Adam.
    """

    def __init__(
        self,
        params: Params,
        loss_fn: Loss,
        lr=0.1,
        amsgrad=False,
        opt_steps: int = 300,
        opt_stop: float = 1e-16,
        **kwargs,
    ) -> None:

        self.params = params
        self.loss_fn = loss_fn
        self.lr = lr
        self.amsgrad = amsgrad
        self.opt_steps = opt_steps
        self.opt_stop = opt_stop
        self.kwargs = kwargs

    def optimize(self, model: Model, data: Data) -> Params:
        """
        Optimize the parameters of a model using the Adam algorithm.

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

        X = data["X"]
        Y = data["Y"]

        Nx = len(X)
        Ny = len(Y)
        if Nx != Ny:
            raise ValueError(
                f"Number of samples for X ({Nx}) "
                f"not equal to number of samples for Y ({Ny})"
            )

        params = self.params
        for n in range(self.opt_steps):
            opt.zero_grad()
            pred = torch.stack([model(X[k], params) for k in range(Nx)])
            loss = self.loss_fn(pred, Y)
            loss.backward()
            opt.step()

            if loss <= self.opt_stop:
                break

        return params
