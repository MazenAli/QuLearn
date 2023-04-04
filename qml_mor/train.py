from typing import TypeAlias, TypeVar, Iterable
import torch

Tensor: TypeAlias = torch.Tensor
Optimizer: TypeAlias = torch.optim.Optimizer
T = TypeVar("T", bound=Iterable[Tensor])


def train_torch(
    opt: Optimizer,
    qnn_model,
    params: T,
    X: Tensor,
    Y: Tensor,
    opt_steps: int = 300,
    opt_stop: float = 1e-16,
) -> T:
    """
    Fits labels to a quantum neural network model and returns the optimal parameters.

    Args:
        opt: The torch optimizer.
        qnn_model: The quantum neural network model.
        params: The model parameters.
        X (Tensor): The sample of input features of dimension (num_samples, input_dim).
        Y (Tensor): The labels of dimension (num_samples, output_dim).
        opt_steps (int, optional): The number of optimization steps.
            Defaults to 300.
        opt_stop (float, optional): The convergence threshold for the optimization.
            Defaults to 1e-16.

    Returns:
        params: The optimized model parameters.
    """

    mse_loss = torch.nn.MSELoss()
    N = len(X)
    loss = None
    for n in range(opt_steps):
        opt.zero_grad()
        pred = torch.stack([qnn_model(X[k], params) for k in range(N)])
        loss = mse_loss(pred, Y)
        loss.backward()
        opt.step()

        if loss <= opt_stop:
            break

    return params
