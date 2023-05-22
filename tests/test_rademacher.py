import unittest
import torch
from torch.nn import Linear
from torch.optim import Adam
import math

from qml_mor.rademacher import rademacher
from qml_mor.trainer import RegressionTrainer
from qml_mor.loss import RademacherLoss
from qml_mor.datagen import NormalPrior, DataGenRademacher


class TestRademacher(unittest.TestCase):
    def test_rademacher(self):
        # Setup
        sizex = 3
        num_data_samples = 2
        num_sigma_samples = 3
        m = 4
        seed = 0

        model = Linear(sizex, 1, dtype=torch.float64)
        prior = NormalPrior(sizex, seed=seed)
        datagen = DataGenRademacher(
            prior, num_sigma_samples, num_data_samples, seed=seed
        )
        data = datagen.gen_data(m)
        loss_fn = RademacherLoss(data["sigmas"][0])
        opt = Adam(model.parameters(), lr=0.1, amsgrad=True)
        trainer = RegressionTrainer(
            opt, loss_fn, num_epochs=500, opt_stop=1e-18, best_loss=False
        )

        # Call function
        radval = rademacher(model, trainer, data["X"], data["sigmas"], datagen)
        d = sizex + 1
        bound = math.sqrt(2 * d * math.log(math.e * m / d) / m)

        self.assertGreater(radval.item(), 0.0)
        self.assertLessEqual(radval.item(), bound)


if __name__ == "__main__":
    unittest.main()
