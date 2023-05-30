import pytest
import torch
from qulearn.loss import RademacherLoss


@pytest.fixture
def sigmas():
    return torch.tensor([1, -1, 1, -1])


class TestRademacherLoss:
    def test_init(self, sigmas):
        # Check if sigmas are set correctly
        loss = RademacherLoss(sigmas)
        assert torch.equal(loss.sigmas, sigmas)

    @pytest.mark.parametrize(
        "invalid_sigmas", [torch.tensor([1, 2, 3]), torch.tensor([[1, -1], [9.0, 1.0]])]
    )
    def test_invalid_sigmas(self, invalid_sigmas):
        # Check if exception is raised for invalid sigmas
        with pytest.raises(ValueError):
            RademacherLoss(invalid_sigmas)

    def test_forward(self, sigmas):
        output = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
        expected_loss = 0.05  # -1/4 * sum([0.1, -0.2, 0.3, -0.4])
        loss = RademacherLoss(sigmas)
        actual_loss = loss(output)
        assert actual_loss.item() == pytest.approx(expected_loss, abs=1e-5)

    @pytest.mark.parametrize(
        "invalid_output",
        [torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.tensor([0.1, 0.2, 0.3])],
    )
    def test_forward_invalid_output(self, sigmas, invalid_output):
        # Check if exception is raised for invalid output
        loss = RademacherLoss(sigmas)
        with pytest.raises(ValueError):
            loss(invalid_output)

    @pytest.mark.parametrize(
        "invalid_sigmas", [torch.tensor([1, 2, 3]), torch.tensor([[1, -1], [1, -1]])]
    )
    def test_check_sigmas(self, sigmas, invalid_sigmas):
        # Check if check_sigmas method is working correctly
        loss = RademacherLoss(sigmas)
        with pytest.raises(ValueError):
            loss._check_sigmas(invalid_sigmas)
