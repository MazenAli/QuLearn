import unittest
from typing import List
import torch
import pennylane as qml
from qml_mor.models import IQPEReuploadSU2Parity
from qml_mor.capacity import capacity, fit_rand_labels
from qml_mor.datagen import DataGenCapacity
from qml_mor.optimize import AdamTorch


class TestCapacity(unittest.TestCase):
    def test_capacity_qnn(self):
        Nmin = 1
        Nmax = 3
        num_samples = 2

        num_qubits = 3
        num_reups = 1
        num_layers = 1
        sizex = num_qubits

        datagen = DataGenCapacity(sizex=sizex)

        omega = 1.0
        init_theta = torch.randn(num_reups, num_qubits, requires_grad=True)
        theta = torch.randn(
            num_reups, num_layers, num_qubits - 1, 2, requires_grad=True
        )
        W = torch.randn(2**num_qubits, requires_grad=True)

        params = [init_theta, theta, W]

        model = IQPEReuploadSU2Parity(params, omega)
        dev = qml.device("default.qubit", wires=num_qubits, shots=None)

        @qml.qnode(dev, interface="torch")
        def qnn_model(x, params):
            return model.qfunction(x, params)

        loss_fn = torch.nn.MSELoss()
        opt = AdamTorch(params, loss_fn, opt_steps=20)

        C = capacity(Nmin, Nmax, num_samples, datagen, opt, qnn_model)
        Cmin = min(C)
        Cmax = max(C)

        self.assertIsInstance(C, List)
        self.assertLessEqual(Cmax, 100)
        self.assertGreaterEqual(Cmin, 0)


class TestFitRandLabels(unittest.TestCase):
    def test_fit_labels(self):
        N = 3
        sizex = 2
        num_samples = 10
        num_qubits = sizex
        num_reups = 1
        num_layers = 1

        datagen = DataGenCapacity(sizex=sizex)

        omega = 1.0
        init_theta = torch.randn(num_reups, num_qubits, requires_grad=True)
        theta = torch.randn(
            num_reups, num_layers, num_qubits - 1, 2, requires_grad=True
        )
        W = torch.randn(2**num_qubits, requires_grad=True)

        params = [init_theta, theta, W]

        model = IQPEReuploadSU2Parity(params, omega)
        dev = qml.device("default.qubit", wires=num_qubits, shots=None)

        @qml.qnode(dev, interface="torch")
        def qnn_model(x, params):
            return model.qfunction(x, params)

        loss_fn = torch.nn.MSELoss()
        opt = AdamTorch(params, loss_fn, opt_steps=10)
        mre = fit_rand_labels(N, num_samples, datagen, opt, qnn_model)

        self.assertIsInstance(mre, float)


if __name__ == "__main__":
    unittest.main()
