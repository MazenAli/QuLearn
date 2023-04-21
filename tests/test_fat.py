import torch
import unittest

from qml_mor.fat import fat_shattering_dim, check_shattering, normalize_const
from qml_mor.datagen import DataGenFat
from qml_mor.optimize import AdamTorch


class TestFunctions(unittest.TestCase):
    def test_normalize_const(self):
        weights = torch.tensor([1.0, 2.0, 3.0])
        gamma = 0.5
        C = normalize_const(weights, gamma)
        self.assertIsInstance(C, float)
        self.assertGreater(C, 0.0)

    def test_check_shattering(self):
        sizex = 3
        d = 2
        Sb = 2
        Sr = 3
        gamma = 0.1

        datagen = DataGenFat(sizex, Sb, Sr, 2.0 * gamma)

        class DummyModel:
            def __call__(self, x, params):
                return torch.sum(x * params[0])

        model = DummyModel()

        loss_fn = torch.nn.MSELoss()
        params = [torch.tensor([1.0, 1.0, 1.0], requires_grad=True)]
        opt = AdamTorch(params, loss_fn, opt_steps=20)

        shattered = check_shattering(model, datagen, opt, d, gamma)
        self.assertTrue(shattered)

    def test_fat_shattering_dim(self):
        sizex = 3
        dmin = 1
        dmax = 3
        Sb = 2
        Sr = 3
        gamma = 0.1

        datagen = DataGenFat(sizex, Sb, Sr, 2.0 * gamma)

        class DummyModel:
            def __call__(self, x, params):
                return torch.sum(x * params[0])

        model = DummyModel()

        loss_fn = torch.nn.MSELoss()
        params = [torch.tensor([1.0, 1.0, 1.0], requires_grad=True)]
        opt = AdamTorch(params, loss_fn, opt_steps=20)

        params = [torch.tensor([1.0, 2.0, 3.0])]
        fat_shattering_dimension = fat_shattering_dim(
            model, datagen, opt, dmin, dmax, gamma
        )
        self.assertIsInstance(fat_shattering_dimension, int)
        self.assertGreater(fat_shattering_dimension, 0)


if __name__ == "__main__":
    unittest.main()
