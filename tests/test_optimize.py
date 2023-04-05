import unittest
import torch
import pennylane as qml
from qml_mor.models import IQPEReuploadSU2Parity
from qml_mor.optimize import AdamTorch

# Set up global constants for testing
NUM_QUBITS = 2
THETA_SHAPE = (2, 2, NUM_QUBITS - 1, 2)
W_SHAPE = 2**NUM_QUBITS


class TestTrainAdam(unittest.TestCase):
    def test_AdamTorch(self):
        # Create a sample input dataset X and corresponding labels Y
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        Y = torch.tensor([3.0, 7.0], dtype=torch.float64)

        # Initialize model parameters
        INIT_THETA = torch.zeros(
            (2, NUM_QUBITS), requires_grad=True, dtype=torch.float64
        )
        THETA = torch.zeros(THETA_SHAPE, requires_grad=True, dtype=torch.float64)
        W = torch.ones(W_SHAPE, requires_grad=True, dtype=torch.float64)
        params = [INIT_THETA, THETA, W]

        # Model
        model = IQPEReuploadSU2Parity(params)
        dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)

        @qml.qnode(dev, interface="torch")
        def qnn(x, params):
            return model.qfunction(x, params)

        # Train the model using the train_Adam function
        loss_fn = torch.nn.MSELoss()
        opt = AdamTorch(params=params, loss_fn=loss_fn, opt_steps=20)
        data = {"X": X, "Y": Y}
        optimized_params = opt.optimize(qnn, data)

        # Check if the returned optimized parameters
        # have the same shape as the input parameters
        self.assertEqual(optimized_params[0].shape, params[0].shape)
        self.assertTrue(torch.allclose(params[0], optimized_params[0]))

        INIT_THETA = torch.zeros(
            (2, NUM_QUBITS), requires_grad=True, dtype=torch.float64
        )
        THETA = torch.zeros(THETA_SHAPE, requires_grad=True, dtype=torch.float64)
        W = torch.ones(W_SHAPE, requires_grad=True, dtype=torch.float64)
        params = [INIT_THETA, THETA, W]
        self.assertTrue(not torch.allclose(params[0], optimized_params[0]))

        # Calculate the final loss value
        mse_loss = torch.nn.MSELoss()
        N = len(X)
        pred = torch.stack([qnn(X[k], optimized_params) for k in range(N)])
        final_loss = mse_loss(pred, Y).item()

        # Check if the final loss value has reduced after training
        pred = torch.stack([qnn(X[k], params) for k in range(N)])
        initial_loss = mse_loss(pred, Y).item()
        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
