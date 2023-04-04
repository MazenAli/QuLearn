import numpy as np
import torch
import unittest

from qml_mor.fat_torch import (
    fat_shattering_dim,
    check_shattering,
    generate_samples_b,
    generate_samples_r,
    gen_synthetic_features,
    gen_synthetic_labels,
    normalize_const,
)


class TestFunctions(unittest.TestCase):
    def test_generate_samples_b(self):
        samples = generate_samples_b(2, 4)
        self.assertEqual(samples.shape, (4, 2))
        self.assertEqual(len(np.unique(samples, axis=0)), 4)

    def test_generate_samples_r(self):
        samples = generate_samples_r(2, 4)
        self.assertEqual(samples.shape, (4, 2))

    def test_gen_synthetic_features(self):
        features = gen_synthetic_features(3, 5)
        self.assertEqual(features.shape, (3, 5))

    def test_gen_synthetic_labels(self):
        b = np.array([[0, 1], [1, 0]])
        r = np.array([[0.2, 0.7], [0.4, 0.9]])
        labels = gen_synthetic_labels(b, r, 0.1)
        self.assertEqual(labels.shape, (2, 2, 2))

    def test_normalize_const(self):
        weights = torch.tensor([1.0, 2.0, 3.0])
        gamma = 0.5
        C = normalize_const(weights, gamma)
        self.assertIsInstance(C, float)

    def test_check_shattering(self):
        opt = torch.optim.Adam([torch.tensor(1.0, requires_grad=True)], lr=0.1)

        class DummyModel:
            def __call__(self, x, params):
                return torch.sum(x * params)

        qnn_model = DummyModel()
        params = [torch.tensor(1.0, requires_grad=True)]
        X = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        b = np.array([[0, 1]])
        r = np.array([[0.5, 0.5]])
        gamma = 0.2
        opt_steps = 300
        opt_stop = 1e-16
        cuda = False

        shattered = check_shattering(
            opt, qnn_model, params, X, b, r, gamma, opt_steps, opt_stop, cuda
        )

        self.assertTrue(shattered)

    # Integration test
    def test_fat_shattering_dim(self):
        class DummyModel:
            def __call__(self, x, params):
                return torch.sum(x * params)

        def dummy_optimizer(params, **kwargs):
            return torch.optim.SGD(params, lr=0.01)

        params = [torch.tensor([1.0, 2.0, 3.0])]
        fat_shattering_dimension = fat_shattering_dim(
            1, 3, 3, 2, 2, dummy_optimizer, DummyModel(), params, 0.1
        )
        self.assertIsInstance(fat_shattering_dimension, int)


if __name__ == "__main__":
    unittest.main()
