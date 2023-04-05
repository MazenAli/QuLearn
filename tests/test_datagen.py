import unittest
import torch
from qml_mor.datagen import DataGenCapacity


class TestDataGenCapacity(unittest.TestCase):
    def test_gen_dataset(self):
        N = 3
        sizex = 2
        num_samples = 10
        seed = 0

        datagen = DataGenCapacity(sizex=sizex, seed=seed)

        data = datagen.gen_data(N, num_samples)
        x = data["X"]
        y = data["Y"]

        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape, (N, sizex))
        self.assertEqual(y.shape, (num_samples, N))


if __name__ == "__main__":
    unittest.main()
