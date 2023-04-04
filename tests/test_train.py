import unittest
import torch
from qml_mor.train import train_torch


class TestTrainAdam(unittest.TestCase):
    def test_train_torch(self):
        # Define a simple  model
        def qnn_model(x, params):
            return torch.matmul(x, params[0])

        # Create a sample input dataset X and corresponding labels Y
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        Y = torch.tensor([[3.0], [7.0]])

        # Initialize model parameters
        params = [torch.tensor([[0.0], [0.0]], requires_grad=True)]

        # Train the model using the train_Adam function
        opt = torch.optim.Adam(params, lr=0.1, amsgrad=True)
        optimized_params = train_torch(opt, qnn_model, params, X, Y)

        # Check if the returned optimized parameters
        # have the same shape as the input parameters
        self.assertEqual(optimized_params[0].shape, params[0].shape)

        # Calculate the final loss value
        mse_loss = torch.nn.MSELoss()
        N = len(X)
        pred = torch.stack([qnn_model(X[k], optimized_params) for k in range(N)])
        final_loss = mse_loss(pred, Y).item()

        # Check if the final loss value has reduced after training
        params = [torch.tensor([[0.0], [0.0]], requires_grad=True)]
        initial_loss = mse_loss(torch.matmul(X, params[0]), Y).item()
        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
