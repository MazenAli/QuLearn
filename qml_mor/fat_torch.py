from typing import TypeAlias, Optional, Set, Tuple
import warnings
from itertools import product
import numpy as np
from scipy.stats import qmc
import torch
from .train import train_torch

Tensor: TypeAlias = torch.Tensor
Optimizer: TypeAlias = torch.optim.Optimizer


def fat_shattering_dim(
    dmin: int,
    dmax: int,
    sizex: int,
    Sb: int,
    Sr: int,
    opt: Optimizer,
    qnn_model,
    params,
    gamma: float,
    dstep: int = 1,
    opt_steps: int = 300,
    opt_stop: float = 1e-16,
    seed: Optional[int] = None,
    cuda: bool = False,
):
    """
    Estimate the fat-shattering dimension for a model with a given architecture.

    Args:
        dmin (int): Iteration start for dimension check.
        dmax (int): Iteration stop for dimension check (including).
        sizex (int): Dimension of input features.
        Sb (int): Number of samples for the b vector.
        Sr (int): Number of samples for the r vector.
        opt (Optimizer): Torch optimizer.
        qnn_model: QNN model.
        params: Model parameters.
        gamma (float): The margin value.
        dstep (int, optional): Dimension iteration step size.
            Defaults to 1.
        opt_steps (int, optional): The number of optimization steps.
            Defaults to 300.
        opt_stop (float, optional): The convergence threshold for the optimization.
            Defaults to 1e-16.
        cuda (bool, optional): Set True if run on GPU. Defaults to False.

    Returns:
        int: The estimated fat-shattering dimension.
    """

    for d in range(dmin, dmax + 1, dstep):
        X = gen_synthetic_features(d, sizex, seed, cuda=cuda)
        b = generate_samples_b(d, Sb)
        r = generate_samples_r(d, Sr)
        shattered = check_shattering(
            opt, qnn_model, params, X, b, r, gamma, opt_steps, opt_stop, cuda
        )

        if not shattered:
            if d == dmin:
                warnings.warn(f"Stopped at dmin = {dmin}.")

            return d - 1

    warnings.warn(f"Reached dmax = {dmax}.")
    return dmax


def check_shattering(
    opt: Optimizer,
    qnn_model,
    params,
    X: Tensor,
    b: np.ndarray,
    r: np.ndarray,
    gamma: float,
    opt_steps: int = 300,
    opt_stop: float = 1e-16,
    cuda: bool = False,
) -> bool:
    """
    Check if the model shatters the given set of samples X with margin gamma.

    Args:
        opt: The torch optimizer.
        qnn_model: The QNN model.
        params: The model initial parameters.
        X (Tensor): The set of samples to shatter, dimension (d, sizex).
        b (numpy.ndarray): An array of shape (Sb, d) containing b values.
        r (numpy.ndarray): An array of shape (Sr, d) containing r values.
        gamma (float): The margin value.
        opt_steps (int, optional): The number of optimization steps.
            Defaults to 300.
        opt_stop (float, optional): The convergence threshold for the optimization.
            Defaults to 1e-16.
        cuda (bool, optional): Set True if run on GPU. Defaults to False.

    Returns:
        bool: True if the model shatters the samples, False otherwise.
    """

    Y = gen_synthetic_labels(b, r, gamma, gamma, cuda)
    d = len(X)

    for sr in range(len(r)):

        shattered = True
        for sb in range(len(b)):
            opt_params = train_torch(
                opt, qnn_model, params, X, Y[sr, sb], opt_steps, opt_stop
            )
            predictions = torch.stack([qnn_model(X[k], opt_params) for k in range(d)])

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


def generate_samples_b(d: int, S: int) -> np.ndarray:
    """
    Generate S unique samples of b from {0, 1}^d.

    Args:
        d (int): Dimension of the feature space.
        S (int): Number of samples to generate.

    Returns:
        numpy.ndarray: An array of shape (S, d) containing unique samples of b.
    """

    max_values = 2**d
    if S >= max_values:
        possible_values = list(product([0, 1], repeat=d))
        return np.array(possible_values)
    else:
        samples: Set[Tuple[int, ...]] = set()
        while len(samples) < S:
            b_sample = tuple(np.random.randint(0, 2, size=d))
            samples.add(b_sample)
        return np.array(list(samples))


def generate_samples_r(d: int, S: int) -> np.ndarray:
    """
    Generate S samples of r from [0, 1]^d using Latin Hypercube Sampling.

    Args:
        d (int): Dimension of the feature space.
        S (int): Number of samples to generate.

    Returns:
        numpy.ndarray: An array of shape (S, d) containing samples of r.
    """

    sampler = qmc.LatinHypercube(d=d)
    r_samples = sampler.random(n=S)
    return r_samples


def gen_synthetic_features(
    d: int,
    sizex: int,
    seed: Optional[int] = None,
    scale: float = 2.0,
    shift: float = -1.0,
    cuda: bool = False,
) -> Tensor:
    """
    Generates d inputs x of dimension sizex sampled uniformly from scale*[0,1]+shift.

    Args:
        d (int): The number of inputs x to generate.
        sizex (int): The size of each input.
        seed (int, optional): The random seed to use for generating the features.
            Defaults to None.
        scale (float, optional): The re-scaling factor for uniform random numbers
            in [0,1]. Defaults to 2.0.
        shift (float, optional): The shift value for uniform random numbers [0,1].
            Defaults to -1.0.
        cuda (bool, optional): Set True if run on GPU. Defaults to False.

    Returns:
        Tensor: Tensor X of shape (d, sizex).
    """

    scale = 2.0
    shift = -1.0

    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda")

    if seed is not None:
        torch.manual_seed(seed)

    x = (
        scale
        * torch.rand(d, sizex, dtype=torch.float64, device=device, requires_grad=False)
        + shift
    )

    return x


def gen_synthetic_labels(
    b: np.ndarray,
    r: np.ndarray,
    gamma: float = 0.0,
    c: float = 0.0,
    cuda: bool = False,
) -> Tensor:
    """
    Generate constant label values equal to r_i + gamma + c
    when b_i = 1 and r_i - gamma - c when b_i = 0.

    Args:
        b (numpy.ndarray): An array of shape (Sb,d)
            containing Sb samples of d-dim binary values.
        r (numpy.ndarray): An array of shape (Sr,d)
            containing Sr samples of d-dim real values in [0, 1].
        gamma (float, optional): The fat-shattering margin value.
            Defaults to 0.0.
        c (float, optional): The constant value added to the margin.
            Defaults to 0.0.
        cuda (bool, optional): Set True if run on GPU. Defaults to False.

    Returns:
        Tensor: Y of shape (Sr, Sb, d).

    Raises:
            ValueError: If the length of b[0] is not the same as the length of r[0].
    """

    Sb = len(b)
    Sr = len(r)
    d = d1 = len(b[0])
    d2 = len(r[[0]])

    if d1 != d2:
        raise ValueError(
            f"The length of b[0] and r[0] are {d1} and {d2}. "
            f"Should be constant and the same."
        )

    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda")

    labels = np.zeros((Sr, Sb, d))
    for sr in range(Sr):
        for sb in range(Sb):
            for i in range(d):

                if b[sb][i] == 1:
                    labels[sr, sb, i] = r[sr, i] + (gamma + c)
                else:
                    labels[sr, sb, i] = r[sr, i] - (gamma + c)

    y = torch.tensor(labels, dtype=torch.float64, device=device, requires_grad=False)

    return y


def normalize_const(weights: Tensor, gamma: float) -> float:
    """
    Compute a normalization constant given a tensor of weights
    and the margin parameter gamma.

    Args:
        weights (Tensor): Tensor of weights
        gamma (float): Margin parameter.

    Returns:
        float: A positive real-valued normalization constant.
    """

    V = torch.norm(weights, p=1)
    C = (V / gamma) ** 2 * np.log2(V / gamma)

    return C
