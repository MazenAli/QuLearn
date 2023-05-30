import torch
from torch.nn import Linear
from torch.optim import Adam
import math

from qulearn.rademacher import rademacher
from qulearn.trainer import RegressionTrainer
from qulearn.loss import RademacherLoss
from qulearn.datagen import NormalPrior, DataGenRademacher


def test_rademacher():
    # Setup
    sizex = 3
    num_data_samples = 2
    num_sigma_samples = 3
    m = 4
    seed = 0

    model = Linear(sizex, 1, dtype=torch.float64)
    prior = NormalPrior(sizex, seed=seed)
    datagen = DataGenRademacher(prior, num_sigma_samples, num_data_samples, seed=seed)
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

    assert radval.item() > 0.0
    assert radval.item() <= bound
