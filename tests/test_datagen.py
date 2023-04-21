import unittest
import torch
import numpy as np
from qml_mor.datagen import DataGenCapacity, DataGenFat


class TestDataGenCapacity(unittest.TestCase):
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


class TestDataGenFat(unittest.TestCase):
    def test_gen_dataset(self):
        d = 3
        Sb = 5
        Sr = 2
        sizex = 2
        gamma = 0.5
        seed = 0

        datagen = DataGenFat(sizex=sizex, Sb=Sb, Sr=Sr, gamma=gamma, seed=seed)

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


if __name__ == "__main__":
    unittest.main()
