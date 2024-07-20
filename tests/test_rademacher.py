import logging
import math

import torch
from torch.nn import Linear
from torch.optim import Adam

from qulearn.datagen import DataGenRademacher, NormalPrior
from qulearn.loss import RademacherLoss
from qulearn.rademacher import rademacher
from qulearn.trainer import SupervisedTrainer


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
    metrics = {"Loss": loss_fn}
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    trainer = SupervisedTrainer(
        opt, loss_fn=loss_fn, metrics=metrics, num_epochs=100, logger=logger
    )

    # Call function
    radval = rademacher(model, trainer, data["X"], data["sigmas"], datagen)
    X = data["X"]
    X_reshaped = X.view(-1, X.shape[2])
    norms = torch.norm(X_reshaped, dim=1, p=2)
    B = torch.max(norms)
    W = math.sqrt(sum(p.norm(p=2).item() ** 2 for p in model.parameters()))
    bound = B * W / math.sqrt(num_data_samples)

    assert radval.item() > 0.0
    assert radval.item() <= bound.item()
