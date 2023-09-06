import os
import pytest
import tempfile
import torch
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from qulearn.trainer import SupervisedTrainer


def test_trainer():
    # Create a sample input dataset X and corresponding labels Y
    N = 104
    X = torch.randn(N, 10, dtype=torch.float64)
    A = torch.randn(10, 1, dtype=torch.float64)
    eps = torch.randn(N, dtype=torch.float64) * 0.01
    b = torch.randn(1, dtype=torch.float64)
    Y = torch.matmul(X, A) + b + eps

    model = torch.nn.Linear(10, 1, bias=True, dtype=torch.float64)

    batch_size = 4
    shuffle = True

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    opt = Adam(model.parameters(), lr=0.1, amsgrad=True)
    loss_fn = torch.nn.MSELoss()
    metrics = {"Val_loss": loss_fn}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model")
        writer = SummaryWriter(path)
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        num_epochs = 20
        trainer = SupervisedTrainer(opt, loss_fn, num_epochs, metrics, writer, logger)
        trainer.train(model, loader, loader)

    # Calculate the final loss value
    with torch.no_grad():
        pred = model(X)
        final_loss = loss_fn(pred, Y).item()
        assert 0.0 == pytest.approx(final_loss, abs=1e-3)
