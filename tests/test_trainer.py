import os
import io
import pytest
import tempfile
import torch
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from qulearn.trainer import SupervisedTrainer, RidgeRegression
from qulearn.qkernel import QKernel
from qulearn.qlayer import HadamardLayer, ParallelEntangledIQPEncoding
from torch.nn import MSELoss


def test_trainer():
    # Create a sample input dataset X and corresponding labels Y
    N = 104
    d = 10
    X = torch.randn(N, d, dtype=torch.float64)
    A = torch.randn(d, 1, dtype=torch.float64)
    eps = torch.randn(N, 1, dtype=torch.float64) * 0.01
    b = torch.randn(1, dtype=torch.float64)*torch.ones(N, 1, dtype=torch.float64)
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


@pytest.fixture
def setup_ridge_regression():
    lambda_reg = 0.1
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    loss_fn = torch.nn.MSELoss()
    metrics = {"Val_loss": loss_fn}
    trainer = RidgeRegression(lambda_reg, metrics, logger)

    embed = HadamardLayer(wires=2)
    X_train = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    labels = torch.ones(3)
    train_data = TensorDataset(X_train, labels)
    valid_data = TensorDataset(X_train, labels)
    train_data = DataLoader(train_data, batch_size=3)
    valid_data = DataLoader(valid_data, batch_size=3)
    qkernel = QKernel(embed, X_train)

    return trainer, qkernel, X_train, labels, train_data, valid_data, logger, metrics


def test_train(setup_ridge_regression):
    (
        trainer,
        qkernel,
        X_train,
        labels,
        train_data,
        valid_data,
        _,
        metrics,
    ) = setup_ridge_regression

    predicted = qkernel(X_train)
    init_loss = metrics["Val_loss"](predicted, labels)
    # Train the model
    trainer.train(qkernel, train_data, valid_data)

    predicted = qkernel(X_train)
    final_loss = metrics["Val_loss"](predicted, labels)

    assert final_loss < init_loss


def test_kernel_ridge_regression(setup_ridge_regression):
    trainer, qkernel, _, labels, _, _, _, _ = setup_ridge_regression

    inputs = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    labels = torch.ones(3, 1)
    alpha = trainer.kernel_ridge_regression(qkernel, inputs, labels)

    assert alpha.shape == labels.shape


def test_input_dimension_check(setup_ridge_regression):
    trainer, qkernel, _, _, _, _, _, _ = setup_ridge_regression

    # Ensure ValueError is raised for incorrect input dimensions
    inputs = torch.Tensor([1.0, 2.0, 3.0])
    labels = torch.ones(3)
    with pytest.raises(ValueError):
        trainer.kernel_ridge_regression(qkernel, inputs, labels)


def run_training():
    wires = 1 * 3
    embed = ParallelEntangledIQPEncoding(wires, num_features=1)
    num_features = 1
    num_samples = 10
    X_train = torch.randn((num_samples, num_features))
    labels = torch.randn((num_samples))
    model = QKernel(embed, X_train)

    predicted = model(X_train)
    fn = MSELoss()
    loss_before = fn(predicted, labels)

    lambda_reg = 1.0
    metrics = {"mse_loss": fn}
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    trainer = RidgeRegression(lambda_reg, metrics, logger)
    dataset = TensorDataset(X_train, labels)
    train_loader = DataLoader(dataset, batch_size=num_samples)
    valid_loader = DataLoader(dataset, batch_size=num_samples)
    trainer.train(model, train_loader, valid_loader)

    predicted = model(X_train)
    loss_after = fn(predicted, labels)

    log_contents = log_capture_string.getvalue()
    logger.removeHandler(ch)

    return loss_before, loss_after, log_contents


def test_training_behavior():
    loss_before, loss_after, logs = run_training()

    assert (
        loss_after < loss_before
    ), f"Loss did not decrease after training. Before: {loss_before}, After: {loss_after}"

    assert "Train - Metrics: mse_loss:" in logs, "Train logging missing or incorrect"
    assert (
        "Validate - Metrics: mse_loss:" in logs
    ), "Validation logging missing or incorrect"
