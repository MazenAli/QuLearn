import torch
from qulearn.hat_basis import HatBasis


def test_position_left_of_a():
    hat_basis = HatBasis(a=0.0, b=1.0, num_nodes=5)
    x = torch.tensor([-0.1, -1.0])
    expected = torch.tensor([-1, -1])
    assert torch.equal(
        hat_basis.position(x), expected
    ), "Position left of a should be -1"


def test_position_right_of_b():
    hat_basis = HatBasis(a=0.0, b=1.0, num_nodes=5)
    x = torch.tensor([1.1, 2.0])
    expected = torch.tensor([-2, -2])
    assert torch.equal(
        hat_basis.position(x), expected
    ), "Position right of b should be -2"


def test_position_within_range():
    hat_basis = HatBasis(a=0.0, b=1.0, num_nodes=5)
    x = torch.tensor([0.25, 0.5, 0.75])
    expected = torch.tensor([1, 2, 3])
    assert torch.equal(
        hat_basis.position(x), expected
    ), "Position within range should be correct"


def test_grid_points_boundary_conditions():
    hat_basis = HatBasis(a=0.0, b=1.0, num_nodes=5)
    x = torch.tensor([-0.1, 1.1])
    left_expected = torch.tensor([-0.25, 1.0])
    right_expected = torch.tensor([0.0, 1.25])
    left, right = hat_basis.grid_points(x)
    assert torch.equal(left, left_expected) and torch.equal(
        right, right_expected
    ), "Grid points at boundaries should be correct"


def test_nonz_vals():
    hat_basis = HatBasis(a=0.0, b=1.0, num_nodes=5)
    x = torch.tensor([0.25, 0.75])
    first_expected, second_expected = (1.0, 1.0), (0.0, 0.0)
    first, second = hat_basis.nonz_vals(x)
    assert torch.allclose(first, torch.tensor(first_expected)) and torch.allclose(
        second, torch.tensor(second_expected)
    ), "Non-zero values should be correct"


def test_eval_basis_vector():
    hat_basis = HatBasis(a=-1.0, b=1.0, num_nodes=5)
    x = torch.tensor([0.0])
    expected = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    result = hat_basis.eval_basis_vector(x)
    assert torch.allclose(result, expected), "Basis vector evaluation should be correct"
