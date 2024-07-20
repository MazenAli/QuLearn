import math
from math import pi
from typing import List

import torch

from .types import Model, ParameterList, Tensor


def compute_effdim(
    model: Model,
    features: Tensor,
    param_list: ParameterList,
    weights: Tensor,
    volume: Tensor,
    gamma: Tensor,
) -> Tensor:
    """
    Compute the effective dimension as proposed in arXiv:2112.04807.

    :param model: The statistical model, which outputs a probability vector.
    :type model: Model
    :param features: The input data (X) used to compute the FIMs, shape (num_samples, num_features).
    :type features: Tensor
    :param param_list: A list of lists of parameters. Each set of parameters is used to
        compute an empirical FIM.
    :param weights: Weights for the Monte Carlo integration. The integral f(theta) is estimated
        as 1/N * sum_k f_k w_k.
    :type weights: Tensor
    :param volume: Volume of the parameter space.
    :type volume: Tensor
    :param gamma: Gamma parameter of the effective dimension.
    :type gamma: Tensor

    :return: Effective dimension.
    :rtype: Tensor

    .. note::
        The effective dimension will be negative for data samples that are too small,
        where n/log(n) < 2pi/gamma.
    """

    num_data_samples = len(features)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    fims = compute_fims(model, features, param_list)
    trace_integral = mc_integrate_fim_trace(fims, weights)
    norm_const = norm_const_fim(trace_integral, num_parameters, volume)
    kappa = const_effdim(num_data_samples, gamma)
    effdim = mc_integrate_fims_effdim(fims, weights, norm_const, kappa, volume)

    return effdim


def mc_integrate_fims_effdim(
    fims: List[Tensor],
    weights: Tensor,
    norm_const: Tensor,
    kappa: Tensor,
    volume: Tensor,
) -> Tensor:
    """
    Performs a weighted Monte Carlo integration of the square root of the determinant
    of the matrix (Identity + c*FIM(theta)) and estimate the effective dimension
    as in arXiv:2112.04807.

    :param fims: List of Fisher Information Matrices estimated at different parameter samples.
    :type fims: List[Tensor]
    :param weights: Weights for each parameter sample, or a single constant weight.
        The integral f(theta) is estimated as 1/N * sum_k f_k w_k.
    :type weights: Tensor
    :param norm_const: The FIM normalization constant.
    :type norm_const: Tensor
    :param kappa: The denominator in effective dimension.
    :type kappa: Tensor
    :param volume: Volume of the parameter space.
    :type volume: Tensor

    :return: A scalar tensor of the estimated effective dimension.
    :rtype: Tensor

    :raises ValueError: If any of the FIMs is not a square matrix, or if the number
        of FIMs is not equal to the number of weights (for non-scalar weights).
    """

    num_samples = len(fims)
    if weights.numel() > 1:
        num_weights = len(weights)
        if num_samples != num_weights:
            raise ValueError(
                f"Number of samples ({num_samples}) "
                "must be equal to number of weights ({num_weights})"
            )

    logdets = torch.zeros(num_samples, device=fims[0].device, dtype=fims[0].dtype)

    for i, fim in enumerate(fims):
        _check_fim(fim)
        logdet = half_log_det(fim, norm_const * kappa)
        logdets[i] = logdet
    zeta = torch.max(logdets)

    sum = torch.zeros(1, device=fims[0].device, dtype=fims[0].dtype)
    for logdet in logdets:
        if weights.numel() > 1:
            weight = weights[i]
        else:
            weight = weights

        sum += weight * torch.exp(logdet - zeta)
    sum /= num_samples

    result = 2.0 * zeta / torch.log(kappa) + 2.0 / torch.log(kappa) * torch.log(1.0 / volume * sum)

    return result


def half_log_det(fim: Tensor, c: Tensor) -> Tensor:
    """
    Estimates half of the logarithm of the determinant of the matrix (Identity + c*FIM).

    This function takes as input a square Fisher Information Matrix (FIM) and a scaling factor,
    and returns half of the logarithm of the determinant of the matrix (Identity + c*FIM).

    :param fim: The square Fisher Information Matrix (FIM).
    :type fim: Tensor
    :param c: The scaling factor.
    :type c: Tensor

    :return: Half of the logarithm of the determinant of the matrix (Identity + c*FIM).
    :rtype: Tensor

    :raises ValueError: If the FIM is not a square matrix.
    """

    _check_fim(fim)

    eigs = torch.linalg.eigvalsh(fim)
    result = torch.tensor(0.5, device=fim.device, dtype=fim.dtype) * sum(torch.log(1.0 + c * eigs))
    return result


def const_effdim(num_samples: int, gamma: Tensor) -> Tensor:
    """
    Computes the constant factor in the effective dimension as per the formula described in
    arXiv:2011.00027.

    This function takes the number of data samples and the effective dimension gamma parameter
    (between 0 and 1), and returns the constant factor in the effective dimension formula.

    :param num_samples: Number of data samples.
    :type num_samples: int
    :param gamma: The effective dimension gamma parameter (should be between 0 and 1, exclusive).
    :type gamma: float

    :return: The constant factor in the effective dimension formula.
    :rtype: float

    :raises ValueError: If the number of samples is less than 2 or the gamma parameter is not in
        the (0, 1] range.
    """

    if num_samples < 2:
        raise ValueError(f"num_samples ({num_samples}) has to be at least 2")
    if gamma <= 0.0 or gamma > 1.0:
        raise ValueError(f"gamma ({gamma}) has to be in (0, 1])")

    const = gamma * num_samples / (2 * pi * math.log(num_samples))
    return const


def norm_const_fim(trace_integral: Tensor, num_parameters: int, volume: Tensor) -> Tensor:
    """
    Computes the normalization constant for the Fisher Information Matrix (FIM).

    The function takes as input the integral over the trace of FIMs, the number
    of trainable parameters (i.e., the dimension of the FIM), and the volume of the
    parameter space.

    :param trace_integral: Integral over the trace of FIMs.
    :type trace_integral: Tensor
    :param num_parameters: Number of trainable parameters, or the dimension of the FIM.
    :type num_parameters: int
    :param volume: Volume of the parameter space.
    :type volume: Tensor

    :return: Normalization constant for the FIM.
    :rtype: Tensor

    :raises ValueError: If the trace integral is less than or equal to zero, the number of
        parameters is less than one, or the volume is negative.
    """

    if trace_integral <= 0.0:
        raise ValueError(f"trace_integral ({trace_integral}) must be positive")
    if num_parameters < 1:
        raise ValueError(f"num_parameters ({num_parameters}) has to be at least 1")
    if volume < 0.0:
        raise ValueError(f"volume ({volume}) has to be non-negative")

    norm_const = num_parameters * volume / trace_integral
    return norm_const


def mc_integrate_fim_trace(
    fims: List[Tensor],
    weights: Tensor,
) -> Tensor:
    """
    Performs a weighted Monte Carlo integration of the traces of the Fisher Information
    Matrices (FIMs), which are estimated at different parameter samples.

    The function takes as input a list of Fisher Information Matrices and weights for each
    parameter sample (or a single constant weight). It returns a scalar tensor of the
    estimated integral of the traces of the FIMs.

    :param fims: List of Fisher Information Matrices estimated at different parameter samples.
    :type fims: List[Tensor]
    :param weights: Weights for each parameter sample, or a single constant weight.
        The integral f(theta) is estimated as 1/N * sum_k f_k w_k.
    :type weights: Tensor

    :return: A scalar tensor of the estimated integral of the trace.
    :rtype: Tensor
    """

    num_samples = len(fims)
    if weights.numel() > 1:
        num_weights = len(weights)
        if num_samples != num_weights:
            raise ValueError(
                f"Number of samples ({num_samples}) "
                f"must be equal to number of weights ({num_weights})"
            )

    sum = torch.zeros(1, device=fims[0].device, dtype=fims[0].dtype)
    for i, fim in enumerate(fims):
        _check_fim(fim)
        trace = torch.trace(fim)

        if weights.numel() > 1:
            weight = weights[i]
        else:
            weight = weights

        sum += weight * trace

    sum /= num_samples

    return sum


def compute_fims(
    model: Model,
    features: Tensor,
    param_list: ParameterList,
) -> List[Tensor]:
    """
    Computes a list of empirical Fisher Information Matrices (FIMs) for a given parameter list.

    This function modifies the parameters of the model temporarily for each set of parameters
    in the parameter list and computes the corresponding empirical FIM.

    :param model: The model returning a discrete probability distribution.
    :type model: Model
    :param features: The input data (X) used to compute the FIMs.
    :type features: Tensor
    :param param_list: A list of lists of parameters. Each set of parameters is used to
        compute an empirical FIM.
    :type param_list: ParameterList

    :return: A list of Fisher information matrices.
    :rtype: List[Tensor]

    .. note::
        This function assumes that the model's parameters are differentiable.
    """

    original_params = [p.clone() for p in model.parameters() if p.requires_grad]

    fims = []
    for param in param_list:
        for model_param, sample_param in zip(model.parameters(), param):
            model_param.data = sample_param

        fim = empirical_fim(model, features)
        fims.append(fim)

    for model_param, original_param in zip(model.parameters(), original_params):
        model_param.data = original_param

    return fims


def empirical_fim(model: Model, features: Tensor) -> Tensor:
    """
    Computes the empirical Fisher information matrix.

    :param model: The model returning a discrete probability distribution.
    :type model: Model
    :param features: The training feature set X to estimate the FIM.
    :type features: Tensor

    :returns: The Fisher information matrix.
    :rtype: Tensor

    :raises ValueError: If invalid features format.

    .. note::
        This function assumes that the output of the model is a differentiable
        tensor of probabilities.
    """

    _check_features(features)

    probs = model(features)
    log_probs = torch.log(probs)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_samples = log_probs.shape[0]
    num_states = log_probs.shape[1]
    FIM = torch.zeros(
        (num_parameters, num_parameters), device=features.device, dtype=features.dtype
    )

    for sample in range(num_samples):
        for state in range(num_states):
            model.zero_grad()
            log_prob = log_probs[sample, state]
            log_prob.backward(retain_graph=True)
            grad_list = [
                p.grad.view(-1)
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            grad = torch.cat(grad_list)
            prod = torch.outer(grad, grad)
            FIM += prod * probs[sample, state]
    FIM /= num_samples

    return FIM


def _check_features(features: Tensor) -> None:
    lshapex = len(features.shape)
    if lshapex != 2:
        raise ValueError(f"features must be a 2-dim tensor (not {lshapex})")


def _check_fim(fim: Tensor) -> None:
    shape = fim.shape
    len_shape = len(shape)
    if len_shape != 2:
        raise ValueError(f"fim (shape={shape}) must be a squared matrix.")
    if shape[0] != shape[1]:
        raise ValueError(f"fim (shape={shape}) must be a squared matrix.")
