import unittest
from typing import List
import os
import tempfile
import torch
from torch.nn import Linear
from torch.optim import Adam
import pennylane as qml
from qml_mor.models import IQPEReuploadSU2Parity
from qml_mor.capacity import capacity, fit_rand_labels
from qml_mor.datagen import DataGenCapacity
from qml_mor.trainer import RegressionTrainer


class TestCapacity(unittest.TestCase):
    def test_capacity_qnn(self):
        Nmin = 1
        Nmax = 3
        num_samples = 2

        num_qubits = 3
        num_reups = 1
        num_layers = 1
        sizex = num_qubits

        datagen = DataGenCapacity(sizex=sizex, num_samples=num_samples)

        omega = 1.0
        init_theta = torch.randn(num_reups, num_qubits, requires_grad=True)
        theta = torch.randn(
            num_reups, num_layers, num_qubits - 1, 2, requires_grad=True
        )
        W = torch.randn(2**num_qubits, requires_grad=True)

        params = [init_theta, theta, W]

        dev = qml.device("default.qubit", wires=num_qubits, shots=None)
        model = IQPEReuploadSU2Parity(dev, params, omega)
        loss_fn = torch.nn.MSELoss()
        opt = Adam(model.parameters(), lr=0.1, amsgrad=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            trainer = RegressionTrainer(opt, loss_fn, num_epochs=20, file_name=path)
            C = capacity(model, datagen, trainer, Nmin, Nmax)
            self.assertIsInstance(C, List)
            self.assertLessEqual(C[0][3], 100)
            self.assertGreaterEqual(C[0][3], 0)

    def test_capacity_linear(self):
        num_samples = 10

        sizex = 3

        Nmin = sizex - 2
        Nmax = sizex + 2
        seed = 0

        datagen = DataGenCapacity(sizex=sizex, num_samples=num_samples, seed=seed)
        loss_fn = torch.nn.MSELoss()
        model = Linear(sizex, 1, dtype=torch.float64)
        opt = Adam(model.parameters(), lr=0.1, amsgrad=True)
        trainer = RegressionTrainer(opt, loss_fn, num_epochs=500, best_loss=False)
        C = capacity(model, datagen, trainer, Nmin, Nmax, stop_count=2)

        self.assertIsInstance(C, List)
        self.assertGreaterEqual(C[0][3], 0)
        self.assertGreaterEqual(len(C), 4)
        self.assertGreaterEqual(C[3][3], C[4][3])


class TestFitRandLabels(unittest.TestCase):
    def test_fit_labels(self):
        N = 3
        sizex = 2
        num_samples = 10
        num_qubits = sizex
        num_reups = 1
        num_layers = 1

        datagen = DataGenCapacity(sizex=sizex, num_samples=num_samples)

        omega = 1.0
        init_theta = torch.randn(num_reups, num_qubits, requires_grad=True)
        theta = torch.randn(
            num_reups, num_layers, num_qubits - 1, 2, requires_grad=True
        )
        W = torch.randn(2**num_qubits, requires_grad=True)

        params = [init_theta, theta, W]

        dev = qml.device("default.qubit", wires=num_qubits, shots=None)
        model = IQPEReuploadSU2Parity(dev, params, omega)

        loss_fn = torch.nn.MSELoss()
        opt = Adam(params, lr=0.1, amsgrad=True)
        trainer = RegressionTrainer(opt, loss_fn, num_epochs=10, best_loss=False)
        mre = fit_rand_labels(model, datagen, trainer, N)

        self.assertIsInstance(mre, float)


if __name__ == "__main__":
    unittest.main()
