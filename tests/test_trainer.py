import unittest
import os
import tempfile
import torch
from torch.nn import Parameter
from torch.optim import Adam
import pennylane as qml
from qml_mor.models import IQPEReuploadSU2Parity
from qml_mor.trainer import RegressionTrainer

# Set up global constants for testing
NUM_QUBITS = 2
THETA_SHAPE = (2, 2, NUM_QUBITS - 1, 2)
W_SHAPE = 2**NUM_QUBITS


class TestTrainer(unittest.TestCase):
    def test_trainer(self):
        # Create a sample input dataset X and corresponding labels Y
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        Y = torch.tensor([[3.0], [7.0]], dtype=torch.float64)

        # Initialize model parameters
        INIT_THETA = torch.zeros(
            (2, NUM_QUBITS), requires_grad=True, dtype=torch.float64
        )
        THETA = torch.zeros(THETA_SHAPE, requires_grad=True, dtype=torch.float64)
        W = torch.ones(W_SHAPE, requires_grad=True, dtype=torch.float64)
        params = [INIT_THETA, THETA, W]

        # Model
        dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)
        loss_fn = torch.nn.MSELoss()
        model = IQPEReuploadSU2Parity(dev, params)
        opt = Adam(model.parameters(), lr=0.1, amsgrad=True)

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

        # Check if the returned optimized parameters
        # have the same shape as the input parameters
        self.assertEqual(model.init_theta.shape, params[0].shape)
        self.assertTrue(torch.allclose(params[0], model.init_theta))

        INIT_THETA = torch.zeros(
            (2, NUM_QUBITS), requires_grad=True, dtype=torch.float64
        )
        THETA = torch.zeros(THETA_SHAPE, requires_grad=True, dtype=torch.float64)
        W = torch.ones(W_SHAPE, requires_grad=True, dtype=torch.float64)
        params = [INIT_THETA, THETA, W]
        self.assertTrue(not torch.allclose(params[0], model.init_theta))

        # Calculate the final loss value
        mse_loss = torch.nn.MSELoss()
        pred = model(X)
        final_loss = mse_loss(pred, Y).item()

        # Check if the final loss value has reduced after training
        model.init_theta = Parameter(INIT_THETA)
        model.theta = Parameter(THETA)
        model.W = Parameter(W)
        pred = model(X)
        initial_loss = mse_loss(pred, Y).item()
        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
