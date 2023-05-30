import os
import tempfile
import torch
from torch.optim import Adam
from qulearn.qlayer import IQPEmbeddingLayer, RYCZLayer, HamiltonianLayer
from qulearn.observable import parities_all_observables
from qulearn.trainer import RegressionTrainer


def test_trainer():
    num_qubits = 2
    num_layers = 1
    num_reups = 1
    # Create a sample input dataset X and corresponding labels Y
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    Y = torch.tensor([[3.0], [7.0]], dtype=torch.float64)

    # QNN model
    embed = IQPEmbeddingLayer(num_qubits, n_repeat=num_reups)
    var = RYCZLayer(num_qubits, n_layers=num_layers)
    observables = parities_all_observables(num_qubits)
    model = HamiltonianLayer(embed, var, observables=observables)

    opt = Adam(model.parameters(), lr=0.1, amsgrad=True)
    loss_fn = torch.nn.MSELoss()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model")
        trainer = RegressionTrainer(opt, loss_fn, num_epochs=20, file_name=path)
        batch_size = 1
        shuffle = True
        data = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=shuffle
        )
        trainer.train(model, loader, loader)
        state = torch.load(f"{path}_bestmre")
        model.load_state_dict(state)

    # Calculate the final loss value
    mse_loss = torch.nn.MSELoss()
    pred = model(X)
    final_loss = mse_loss(pred, Y).item()

    # Check if the final loss value has reduced after training
    for p in model.parameters():
        torch.nn.init.normal_(p)

    pred = model(X)
    initial_loss = mse_loss(pred, Y).item()
    assert final_loss < initial_loss
