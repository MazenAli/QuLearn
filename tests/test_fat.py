import os
import tempfile
import torch
from torch.nn import Linear
from torch.optim import Adam

from qulearn.fat import fat_shattering_dim, check_shattering, normalize_const
from qulearn.datagen import DataGenFat, UniformPrior
from qulearn.trainer import RegressionTrainer


def test_normalize_const():
    weights = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    gamma = 0.5
    sizex = 3
    C = normalize_const(weights, gamma, sizex)
    assert isinstance(C, float)
    assert C > 0.0

def test_check_shattering():
    sizex = 3
    d = 2
    Sb = 2
    Sr = 3
    gamma = 0.1
    prior = UniformPrior(sizex)
    datagen = DataGenFat(prior, Sb, Sr, 2.0 * gamma)

    model = Linear(sizex, 1, dtype=torch.float64)

    loss_fn = torch.nn.MSELoss()
    opt = Adam(model.parameters(), lr=0.1)
    trainer = RegressionTrainer(opt, loss_fn=loss_fn, num_epochs=100, best_loss=False)

    shattered = check_shattering(model, datagen, trainer, d, gamma)
    assert shattered

def test_fat_shattering_dim():
    sizex = 3
    dmin = 1
    dmax = 3
    Sb = 2
    Sr = 3
    gamma = 0.1
    prior = UniformPrior(sizex)
    datagen = DataGenFat(prior, Sb, Sr, 2.0 * gamma)
    model = Linear(sizex, 1, dtype=torch.float64)

    loss_fn = torch.nn.MSELoss()
    opt = Adam(model.parameters(), lr=0.1)
    trainer = RegressionTrainer(opt, loss_fn, num_epochs=100, best_loss=False)

    fat_shattering_dimension = fat_shattering_dim(
        model, datagen, trainer, dmin, dmax, gamma
    )
    assert isinstance(fat_shattering_dimension, int)
    assert fat_shattering_dimension > 0

def test_linear_model():
    sizex = 3
    dmin = 2
    dmax = 6
    Sb = 20
    Sr = 5
    gamma = 0.1
    seed = 0
    prior = UniformPrior(sizex)
    datagen = DataGenFat(prior, Sb, Sr, 2.0 * gamma, seed=seed)

    model = Linear(sizex, 1, dtype=torch.float64)

    loss_fn = torch.nn.MSELoss()
    opt = Adam(model.parameters(), lr=0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model")
        trainer = RegressionTrainer(opt, loss_fn, num_epochs=500, file_name=path)
        fat_shattering_dimension = fat_shattering_dim(
            model, datagen, trainer, dmin, dmax, gamma
        )

        assert isinstance(fat_shattering_dimension, int)
        assert fat_shattering_dimension >= sizex
