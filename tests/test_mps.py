import pytest
import tntorch
import torch

from qulearn.hat_basis import HatBasis
from qulearn.mps import (
    HatBasisMPS,
    MPSQGates,
    compute_max_rank_power,
    embed2unitary,
    zerobit_position_odd,
)


@pytest.fixture
def sample_mps():
    cores = [torch.rand(1, 2, 2)] + [torch.rand(2, 2, 2) for _ in range(2)] + [torch.rand(2, 2, 1)]
    mps = tntorch.Tensor(cores)
    return mps


@pytest.fixture
def sample_hat_basis():
    return HatBasis(a=0, b=1, num_nodes=4)


def test_compute_max_rank_power(sample_mps):
    expected_rank_power = 1
    assert compute_max_rank_power(sample_mps) == expected_rank_power


def test_embed2unitary():
    A = torch.rand(4, 2)
    Q, _ = torch.linalg.qr(A)
    U = embed2unitary(Q)
    assert torch.allclose(U @ U.T, torch.eye(U.shape[0]), atol=1e-6)


def test_zerobit_position_odd():
    assert zerobit_position_odd(1, 4) == 2
    assert zerobit_position_odd(3, 4) == 1


class TestMPSQGates:
    def test_qgates(self, sample_mps):
        mps_qgates = MPSQGates(mps=sample_mps)
        gates = mps_qgates.qgates()
        assert isinstance(gates, list)
        assert all(isinstance(g, torch.Tensor) for g in gates)

    def test_pad_cores(self, sample_mps):
        mps_qgates = MPSQGates(mps=sample_mps)
        padded_mps = mps_qgates.pad_cores()
        assert isinstance(padded_mps, tntorch.tensor.Tensor)


class TestHatBasisMPS:
    def test_eval(self, sample_hat_basis: HatBasis):
        hat_basis_mps = HatBasisMPS(basis=sample_hat_basis)
        x = torch.tensor([0.5])
        mps = hat_basis_mps.eval(x)
        assert isinstance(mps, tntorch.tensor.Tensor)
