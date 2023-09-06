import pytest
import math
import torch
import numpy as np
from qulearn.fim import (
    empirical_fim,
    compute_fims,
    mc_integrate_fim_trace,
    norm_const_fim,
    const_effdim,
    half_log_det,
    mc_integrate_fims_effdim,
    compute_effdim,
)


@pytest.fixture
def model():
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 2, bias=False), torch.nn.Softmax(dim=1)
    )
    # Manually set weights to known values
    model[0].weight.data = torch.tensor([[1.0], [1.0]])
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, num_parameters


def test_output_shape(model):
    # Define input
    X = torch.rand((5, 1))
    # Call function
    fim = empirical_fim(model[0], X)
    # Check output shape
    assert fim.shape == (model[1], model[1])


def test_invalid_input(model):
    # Define invalid input
    X = torch.rand(5)
    # Check if function raises error
    with pytest.raises(ValueError):
        empirical_fim(model, X)


def test_correct_computation(model):
    X = torch.tensor([[1.0], [2.0]])

    # Call function
    fim = empirical_fim(model[0], X)

    # Manually compute expected FIM
    diag = 0.25 * X[0][0] ** 2 + 0.25 * X[1][0] ** 2
    diag *= 0.5
    offdiag = -1.0 * diag
    expected_fim = torch.tensor([[diag, offdiag], [offdiag, diag]])

    # Check that computed FIM is close to expected
    assert torch.allclose(fim, expected_fim, atol=1e-4)


@pytest.fixture
def setup_data():
    fims = [torch.eye(2, dtype=torch.float32) for _ in range(10)]
    weights = torch.tensor([1.0] * 10)
    weights_error = torch.tensor([1.0] * 11)
    return fims, weights, weights_error


def test_mc_integrate_fims_effdim_correct(setup_data):
    integral = mc_integrate_fims_effdim(
        setup_data[0],
        setup_data[1],
        torch.tensor(2.0),
        torch.tensor(3.0),
        torch.tensor(5.0),
    )
    check = 2.0 * math.log(7.0 / 5.0) / math.log(3.0)
    assert check == pytest.approx(integral.item(), abs=1e-4)


def test_mc_integrate_fim_trace_correct(setup_data):
    integral = mc_integrate_fim_trace(setup_data[0], setup_data[1])
    expected = 2.0  # Trace of I_2 is 2, and weights are all 1
    assert expected == pytest.approx(integral.item(), abs=1e-4)


def test_half_log_det(setup_data):
    half_log = half_log_det(setup_data[0][0], torch.tensor(2.0))
    expected = math.log(3)

    assert expected == pytest.approx(half_log.item(), abs=1e-4)


def test_mc_integrate_fim_trace_weights_length_mismatch(setup_data):
    with pytest.raises(ValueError):
        mc_integrate_fim_trace(setup_data[0], setup_data[2])


def test_mc_integrate_fim_trace_weights_scalar(setup_data):
    integral = mc_integrate_fim_trace(setup_data[0], torch.tensor(2.0))
    expected = 4.0  # Weights are doubled
    assert expected == pytest.approx(integral.item(), abs=1e-4)


def test_compute_fims(model):
    # Define some parameter samples
    X = torch.rand(3, 1)
    param_list = [[torch.tensor([[1.5], [-2.0]])], [torch.tensor([[3.0], [4.0]])]]

    # original params
    original_params = [p.clone() for p in model[0].parameters()]

    # Compute FIMs
    fims = compute_fims(model[0], X, param_list)

    # Check the output size is correct
    assert len(fims) == len(param_list)

    # Check each FIM is a 2x2 matrix
    for fim in fims:
        assert fim.shape == (2, 2)

    # Check if reset correctly
    for expected, actual in zip(original_params, model[0].parameters()):
        assert torch.allclose(expected, actual)


def test_const_effdim():
    n = 50
    gamma = 0.9
    expected = gamma * n / (2 * math.pi * math.log(n))
    result = const_effdim(n, gamma)
    assert expected == pytest.approx(result, abs=1e-4)


def test_norm_const():
    trace = torch.tensor(0.5)
    d = 10
    vol = 2.0
    expected = d * vol / trace
    result = norm_const_fim(trace, d, vol)
    assert expected == pytest.approx(result, abs=1e-4)


@pytest.fixture
def setup_effdim():
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 2, bias=False), torch.nn.Softmax(dim=1)
    )
    model[0].weight.data = torch.tensor([[1.0], [1.0]])

    X = torch.tensor([[1.0], [2.0]])
    param_list = [[torch.tensor([[1.0], [1.0]])], [torch.tensor([[1.0], [1.0]])]]
    weights = torch.tensor(2.0)
    volume = torch.tensor(2.0)
    gamma = torch.tensor(0.9)
    return model, X, param_list, weights, volume, gamma


@pytest.fixture
def setup_effdim2():
    dim = 3
    model = torch.nn.Sequential(
        torch.nn.Linear(dim, 5, bias=True), torch.nn.Softmax(dim=1)
    )

    n = 50
    X = torch.randn((n, dim))
    param_list = [[p.clone() for p in model.parameters() if p.requires_grad]] * 2
    weights = torch.tensor(10.0)
    volume = torch.tensor(10.0)
    gamma = torch.tensor(1.0)
    return model, X, param_list, weights, volume, gamma


def test_comp_effdim(setup_effdim):
    effdim = compute_effdim(*setup_effdim)

    num_parameters = sum(
        p.numel() for p in setup_effdim[0].parameters() if p.requires_grad
    )
    X = setup_effdim[1]
    diag = 0.25 * X[0][0] ** 2 + 0.25 * X[1][0] ** 2
    diag *= 0.5
    offdiag = -1.0 * diag
    expected_fim = torch.tensor([[diag, offdiag], [offdiag, diag]])
    traceint = torch.trace(expected_fim) * setup_effdim[4]
    normconst = setup_effdim[4] * num_parameters / traceint
    n = len(X)
    const = setup_effdim[5] * n / (2 * math.pi * math.log(n))
    sqrtdet = math.sqrt(
        torch.det(torch.eye(num_parameters) + const * normconst * expected_fim)
    )

    dgamma = 2 * math.log(sqrtdet) / math.log(const)

    assert dgamma == pytest.approx(effdim.item(), abs=1e-4)


def test_comp_effdim2(setup_effdim2):
    effdim = compute_effdim(*setup_effdim2)
    model = setup_effdim2[0]
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    X = setup_effdim2[1]
    FIM = empirical_fim(model, X)
    trace = torch.trace(FIM)
    nFIM = num_parameters * FIM / trace
    gamma = setup_effdim2[5]
    n = len(X)
    kappa = gamma * n / (2.0 * np.pi * np.log(n))
    M = torch.eye(num_parameters) + nFIM * kappa
    sqrtdet = torch.sqrt(torch.det(M))
    expected = 2.0 * torch.log(sqrtdet) / torch.log(kappa)

    assert expected.item() == pytest.approx(effdim.item(), abs=1e-4)
