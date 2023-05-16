import unittest
import math
import torch
from torch.utils.data import DataLoader
import numpy as np
from qml_mor.datagen import (
    DataGenCapacity,
    DataGenFat,
    DataGenRademacher,
    UniformPrior,
    NormalPrior,
)


class TestDataGenCapacity(unittest.TestCase):
    def setUp(self):
        sizex = 10
        num_samples = 10
        seed = 0
        self.datagen = DataGenCapacity(sizex=sizex, num_samples=num_samples, seed=seed)
        self.data = {
            "X": torch.randn(100, sizex),
            "Y": torch.randn(num_samples, 100),
        }

    def test_gen_dataset(self):
        N = 3
        sizex = 2
        num_samples = 10
        seed = 0

        datagen = DataGenCapacity(sizex=sizex, num_samples=num_samples, seed=seed)

        data = datagen.gen_data(N)
        x = data["X"]
        y = data["Y"]

        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape, (N, sizex))
        self.assertEqual(y.shape, (num_samples, N))

    def test_data_to_loader_raises_value_error_for_invalid_s(self):
        with self.assertRaises(ValueError):
            self.datagen.data_to_loader(self.data, -1)

    def test_data_to_loader_returns_correct_loader(self):
        loader = self.datagen.data_to_loader(self.data, 0)
        self.assertIsInstance(loader, DataLoader)
        self.assertEqual(len(loader.dataset), 100)

    def test_data_to_loader_returns_value_error(self):
        with self.assertRaises(ValueError):
            data = {"X": 0}
            self.datagen.data_to_loader(data, 0)

        with self.assertRaises(ValueError):
            data = {"Y": 0}
            self.datagen.data_to_loader(data, 0)

        with self.assertRaises(ValueError):
            data = {"X": torch.zeros(10), "Y": 0}
            self.datagen.data_to_loader(data, 0)


class TestDataGenFat(unittest.TestCase):
    def test_gen_dataset(self):
        d = 3
        Sb = 5
        Sr = 2
        sizex = 2
        gamma = 0.5
        seed = 0

        prior = UniformPrior(sizex, seed=seed)
        datagen = DataGenFat(prior=prior, Sb=Sb, Sr=Sr, gamma=gamma)

        data = datagen.gen_data(d)
        X = data["X"]
        Y = data["Y"]
        b = data["b"]
        r = data["r"]

        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(Y, torch.Tensor)
        self.assertIsInstance(b, np.ndarray)
        self.assertIsInstance(r, np.ndarray)
        self.assertEqual(X.shape, (d, sizex))
        self.assertEqual(Y.shape, (Sr, Sb, d))

    def setUp(self):
        sizex = 10
        seed = 0
        prior = UniformPrior(sizex, seed=seed)
        self.datagen = DataGenFat(prior=prior, Sb=10, Sr=10, seed=seed)
        self.data = {
            "X": torch.randn(100, sizex),
            "Y": torch.randn(10, 10, 100),
        }

    def test_data_to_loader_raises_value_error_for_invalid_sr(self):
        with self.assertRaises(ValueError):
            self.datagen.data_to_loader(self.data, -1, 0)

    def test_data_to_loader_raises_value_error_for_invalid_sb(self):
        with self.assertRaises(ValueError):
            self.datagen.data_to_loader(self.data, 0, -1)

    def test_data_to_loader_returns_correct_loader(self):
        loader = self.datagen.data_to_loader(self.data, 0, 0)
        self.assertIsInstance(loader, DataLoader)
        self.assertEqual(len(loader.dataset), 100)

    def test_data_to_loader_returns_value_error(self):
        with self.assertRaises(ValueError):
            data = {"X": 0}
            self.datagen.data_to_loader(data, 0, 0)

        with self.assertRaises(ValueError):
            data = {"Y": 0}
            self.datagen.data_to_loader(data, 0, 0)

        with self.assertRaises(ValueError):
            data = {"X": torch.zeros(10), "Y": 0}
            self.datagen.data_to_loader(data, 0, 0)


class TestNormalPrior(unittest.TestCase):
    def setUp(self):
        self.sizex = 10
        self.scale = 2.0
        self.shift = -1.0
        self.normal_prior = NormalPrior(self.sizex, self.scale, self.shift)

    def test_normal_prior(self):
        # Test size of output
        m = 1000
        data = self.normal_prior.gen_data(m)
        self.assertEqual(data.size()[0], m)
        self.assertEqual(data.size()[1], self.sizex)

        # Test if scale and shift have been applied correctly
        self.assertAlmostEqual(torch.mean(data).item(), self.shift, places=1)
        self.assertAlmostEqual(torch.std(data).item(), self.scale, places=1)


class TestUniformPrior(unittest.TestCase):
    def setUp(self):
        self.sizex = 10
        self.scale = 2.0
        self.shift = -1.0
        self.normal_prior = UniformPrior(self.sizex, self.scale, self.shift)

    def test_uniform_prior(self):
        # Test size of output
        m = 100
        data = self.normal_prior.gen_data(m)
        self.assertEqual(data.size()[0], m)
        self.assertEqual(data.size()[1], self.sizex)

        # Test if scale and shift have been applied correctly
        self.assertAlmostEqual(
            torch.mean(data).item(), self.shift + self.scale / 2, places=1
        )
        self.assertAlmostEqual(
            torch.std(data).item(), self.scale / math.sqrt(12), places=1
        )


class TestDataGenRademacher(unittest.TestCase):
    def setUp(self):
        sizex = 10
        num_sigma_samples = 10
        num_data_samples = 10
        # Assuming you have a `UniformPrior` class as the prior
        prior = NormalPrior(sizex)
        self.data_gen = DataGenRademacher(prior, num_sigma_samples, num_data_samples)

    def test_gen_data(self):
        m = 5
        data = self.data_gen.gen_data(m)
        self.assertIsInstance(data, dict)
        self.assertIn("X", data)
        self.assertIn("sigmas", data)

        # Test shape of X
        X = data["X"]
        self.assertEqual(X.size()[0], self.data_gen.num_data_samples)
        self.assertEqual(X.size()[1], m)
        self.assertEqual(X.size()[2], self.data_gen.prior.sizex)

        # Test shape of sigmas
        sigmas = data["sigmas"]
        self.assertEqual(sigmas.size()[0], self.data_gen.num_sigma_samples)
        self.assertEqual(self.data_gen.seed, None)
        self.assertEqual(self.data_gen.prior.seed, None)

        # Test values of sigmas
        self.assertTrue(torch.all((sigmas == 1) | (sigmas == -1)))


if __name__ == "__main__":
    unittest.main()
