import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from qulearn.datagen import (
    DataGenCapacity,
    DataGenFat,
    DataGenRademacher,
    NormalPrior,
    UniformPrior,
    generate_lhs_samples,
    generate_model_lhs_samples,
)


@pytest.fixture(scope="module")
def setup_datagen_capacity():
    sizex = 10
    num_samples = 10
    seed = 0
    datagen = DataGenCapacity(sizex=sizex, num_samples=num_samples, seed=seed)
    return datagen


@pytest.fixture(scope="module")
def setup_datagen_fat():
    sizex = 10
    seed = 0
    prior = UniformPrior(sizex, seed=seed)
    datagen = DataGenFat(prior=prior, Sb=8, Sr=10, seed=seed)
    return datagen


@pytest.fixture(scope="module")
def setup_normal_prior():
    sizex = 10
    scale = 2.0
    shift = -1.0
    normal_prior = NormalPrior(sizex, scale, shift)
    return normal_prior


@pytest.fixture(scope="module")
def setup_uniform_prior():
    sizex = 10
    scale = 2.0
    shift = -1.0
    uniform_prior = UniformPrior(sizex, scale, shift)
    return uniform_prior


@pytest.fixture(scope="module")
def setup_datagen_rademacher():
    sizex = 10
    num_sigma_samples = 10
    num_data_samples = 10
    prior = NormalPrior(sizex)
    datagen = DataGenRademacher(prior, num_sigma_samples, num_data_samples)
    return datagen


def test_gen_dataset_data_gen_capacity(setup_datagen_capacity):
    datagen = setup_datagen_capacity
    N = 3
    sizex = 10
    num_samples = 10
    data = datagen.gen_data(N)
    x = data["X"]
    y = data["Y"]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == torch.Size((N, sizex))
    assert y.shape == torch.Size((num_samples, N, 1))


def test_data_to_loader_raises_value_error_for_invalid_s_data_gen_capacity(
    setup_datagen_capacity,
):
    datagen = setup_datagen_capacity
    data = datagen.gen_data(10)
    with pytest.raises(ValueError):
        datagen.data_to_loader(data, -1)


def test_data_to_loader_returns_correct_loader_data_gen_capacity(
    setup_datagen_capacity,
):
    datagen = setup_datagen_capacity
    data = datagen.gen_data(100)
    loader = datagen.data_to_loader(data, 0)
    assert isinstance(loader, DataLoader)
    assert len(loader.dataset) == 100


def test_data_to_loader_returns_value_error_data_gen_capacity(setup_datagen_capacity):
    datagen = setup_datagen_capacity
    with pytest.raises(ValueError):
        data = {"X": 0}
        datagen.data_to_loader(data, 0)

    with pytest.raises(ValueError):
        data = {"Y": 0}
        datagen.data_to_loader(data, 0)

    with pytest.raises(ValueError):
        data = {"X": torch.zeros(10), "Y": 0}
        datagen.data_to_loader(data, 0)


def test_gen_dataset_data_gen_fat(setup_datagen_fat):
    datagen = setup_datagen_fat
    d = 3
    Sb = 8
    Sr = 10
    sizex = 10

    data = datagen.gen_data(d)
    X = data["X"]
    Y = data["Y"]
    b = data["b"]
    r = data["r"]

    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)
    assert isinstance(b, np.ndarray)
    assert isinstance(r, np.ndarray)
    assert X.shape == (d, sizex)
    assert Y.shape == (Sr, Sb, d, 1)


def test_data_to_loader_raises_value_error_for_invalid_sr_data_gen_fat(
    setup_datagen_fat,
):
    datagen = setup_datagen_fat
    data = datagen.gen_data(10)
    with pytest.raises(ValueError):
        datagen.data_to_loader(data, -1, 0)


def test_data_to_loader_raises_value_error_for_invalid_sb_data_gen_fat(
    setup_datagen_fat,
):
    datagen = setup_datagen_fat
    data = datagen.gen_data(10)
    with pytest.raises(ValueError):
        datagen.data_to_loader(data, 0, -1)


def test_data_to_loader_returns_correct_loader_data_gen_fat(setup_datagen_fat):
    datagen = setup_datagen_fat
    data = datagen.gen_data(100)
    loader = datagen.data_to_loader(data, 0, 0)
    assert isinstance(loader, DataLoader)
    assert len(loader.dataset) == 100


def test_data_to_loader_returns_value_error_data_gen_fat(setup_datagen_fat):
    datagen = setup_datagen_fat
    with pytest.raises(ValueError):
        data = {"X": 0}
        datagen.data_to_loader(data, 0, 0)

    with pytest.raises(ValueError):
        data = {"Y": 0}
        datagen.data_to_loader(data, 0, 0)

    with pytest.raises(ValueError):
        data = {"X": torch.zeros(10), "Y": 0}
        datagen.data_to_loader(data, 0, 0)


def test_normal_prior(setup_normal_prior):
    normal_prior = setup_normal_prior
    sizex = 10
    scale = 2.0
    shift = -1.0

    # Test size of output
    m = 10000
    data = normal_prior.gen_data(m)
    assert data.size()[0] == m
    assert data.size()[1] == sizex

    # Test if scale and shift have been applied correctly
    assert math.isclose(torch.mean(data).item(), shift, abs_tol=1e-1)
    assert math.isclose(torch.std(data).item(), scale, abs_tol=1e-1)


def test_uniform_prior(setup_uniform_prior):
    uniform_prior = setup_uniform_prior
    sizex = 10
    scale = 2.0
    shift = -1.0

    # Test size of output
    m = 10000
    data = uniform_prior.gen_data(m)
    assert data.size()[0] == m
    assert data.size()[1] == sizex

    # Test if scale and shift have been applied correctly
    assert math.isclose(torch.mean(data).item(), shift + scale / 2, abs_tol=1e-1)
    assert math.isclose(torch.std(data).item(), scale / math.sqrt(12), abs_tol=1e-1)


def test_gen_data_data_gen_rademacher(setup_datagen_rademacher):
    datagen = setup_datagen_rademacher
    sizex = 10
    num_sigma_samples = 10
    num_data_samples = 10
    m = 5

    data = datagen.gen_data(m)
    assert isinstance(data, dict)
    assert "X" in data
    assert "sigmas" in data

    # Test shape of X
    X = data["X"]
    assert X.size()[0] == num_data_samples
    assert X.size()[1] == m
    assert X.size()[2] == sizex

    # Test shape of sigmas
    sigmas = data["sigmas"]
    assert sigmas.size()[0] == num_sigma_samples
    assert datagen.seed is None
    assert datagen.prior.seed is None

    # Test values of sigmas
    assert torch.all((sigmas == 1) | (sigmas == -1))


def test_generate_lhs_samples():
    n_samples = 10
    n_dims = 2
    lower_bound = -1.0
    upper_bound = 1.0
    samples = generate_lhs_samples(n_samples, n_dims, lower_bound, upper_bound, seed=0)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (n_samples, n_dims)
    assert (samples >= lower_bound).all()
    assert (samples <= upper_bound).all()


def test_generate_model_lhs_samples():
    model = torch.nn.Linear(3, 1)
    n_samples = 10
    lower_bound = -1.0
    upper_bound = 1.0
    parameter_samples = generate_model_lhs_samples(
        model, n_samples, lower_bound, upper_bound, seed=0
    )
    assert isinstance(parameter_samples, list)
    assert len(parameter_samples) == n_samples
    for sample in parameter_samples:
        assert isinstance(sample, list)
        assert len(sample) == len(list(filter(lambda p: p.requires_grad, model.parameters())))
        for param in sample:
            assert isinstance(param, torch.Tensor)
            assert (param.detach().numpy() >= lower_bound).all()
            assert (param.detach().numpy() <= upper_bound).all()
