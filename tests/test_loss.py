import unittest
import torch
from qml_mor.loss import RademacherLoss


class TestRademacherLoss(unittest.TestCase):
    def setUp(self):
        self.sigmas = torch.tensor([1, -1, 1, -1])
        self.loss = RademacherLoss(self.sigmas)

    def test_init(self):
        # Check if sigmas are set correctly
        self.assertTrue(torch.equal(self.loss.sigmas, self.sigmas))

    def test_invalid_sigmas(self):
        # Check if exception is raised for invalid sigmas
        with self.assertRaises(ValueError):
            RademacherLoss(torch.tensor([1, 2, 3]))  # values not 1 or -1
        with self.assertRaises(ValueError):
            RademacherLoss(torch.tensor([[1, -1], [9.0, 1.0]]))  # 1D tensor

    def test_forward(self):
        output = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
        expected_loss = 0.05  # -1/4 * sum([0.1, -0.2, 0.3, -0.4])
        actual_loss = self.loss(output)
        self.assertAlmostEqual(actual_loss.item(), expected_loss, places=5)

    def test_forward_invalid_output(self):
        # Check if exception is raised for invalid output
        with self.assertRaises(ValueError):
            self.loss(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))  # 2D tensor
        with self.assertRaises(ValueError):
            self.loss(torch.tensor([0.1, 0.2, 0.3]))  # length does not match sigmas

    def test_check_sigmas(self):
        # Check if check_sigmas method is working correctly
        with self.assertRaises(ValueError):
            self.loss._check_sigmas(torch.tensor([1, 2, 3]))  # values not 1 or -1
        with self.assertRaises(ValueError):
            self.loss._check_sigmas(torch.tensor([[1, -1], [1, -1]]))  # 2D tensor


if __name__ == "__main__":
    unittest.main()
