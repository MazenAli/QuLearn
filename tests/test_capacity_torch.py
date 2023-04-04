import unittest
from typing import List
import torch
import pennylane as qml
from qml_mor.models import IQPEReuploadSU2Parity
from qml_mor.capacity_torch import capacity, fit_labels, gen_dataset


class TestCapacity(unittest.TestCase):
    def test_capacity(self):
        Nmin = 1
        Nmax = 3
        sizex = 2
        num_samples = 2
        params = [
            torch.randn(1, sizex, dtype=torch.float64, requires_grad=True),
            torch.randn(1, dtype=torch.float64, requires_grad=True),
        ]

        def qnn_model(x, params):
            res = torch.nn.functional.linear(x, params[0], params[1]).pow(2)[0]
            return res

        opt = torch.optim.Adam(params, lr=0.1, amsgrad=True)
        C = capacity(Nmin, Nmax, sizex, num_samples, opt, qnn_model, params)
        self.assertIsInstance(C, List)

    def test_capacity_qnn(self):
        Nmin = 1
        Nmax = 3
        num_samples = 2

        num_qubits = 3
        num_reups = 1
        num_layers = 1
        sizex = num_qubits

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

        opt = torch.optim.Adam(params, lr=0.1, amsgrad=True)
        C = capacity(Nmin, Nmax, sizex, num_samples, opt, qnn_model, params)

        self.assertIsInstance(C, List)

    def test_capacity_qnn_trivial(self):
        Nmin = 1
        Nmax = 5
        num_samples = 5

        num_qubits = 3
        num_reups = 1
        num_layers = 0
        sizex = num_qubits

        omega = 0.0
        init_theta = torch.randn(num_reups, num_qubits, requires_grad=True)
        theta = torch.randn(
            num_reups, num_layers, num_qubits - 1, 2, requires_grad=False
        )
        W = torch.randn(2**num_qubits, requires_grad=False)

        params = [init_theta, theta, W]

        model = IQPEReuploadSU2Parity(params, omega)
        dev = qml.device("default.qubit", wires=num_qubits, shots=None)

        @qml.qnode(dev, interface="torch")
        def qnn_model(x, params):
            return model.qfunction(x, params)

        opt = torch.optim.Adam(params, lr=0.1, amsgrad=True)
        C = capacity(Nmin, Nmax, sizex, num_samples, opt, qnn_model, params)
        Cmin = min(C)
        Cmax = max(C)

        self.assertLessEqual(Cmax, 100)
        self.assertGreaterEqual(Cmin, 0)


class TestFitLabels(unittest.TestCase):
    def test_fit_labels(self):
        N = 3
        sizex = 2
        num_samples = 10
        params = [
            torch.randn(1, sizex, dtype=torch.float64, requires_grad=True),
            torch.randn(1, dtype=torch.float64, requires_grad=True),
        ]
        opt_steps = 10
        opt_stop = 1e-5

        def qnn_model(x, params):
            return torch.nn.functional.linear(x, params[0], params[1]).pow(2)[0]

        opt = torch.optim.Adam(params, lr=0.1, amsgrad=True)
        mre = fit_labels(
            N,
            sizex,
            num_samples,
            opt,
            qnn_model,
            params,
            opt_steps=opt_steps,
            opt_stop=opt_stop,
        )

        self.assertIsInstance(mre, float)


class TestGenDataset(unittest.TestCase):
    def test_gen_dataset(self):
        N = 3
        sizex = 2
        num_samples = 10
        seed = 0

        x, y = gen_dataset(N, sizex, num_samples, seed)

        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape, (N, sizex))
        self.assertEqual(y.shape, (num_samples, N))


if __name__ == "__main__":
    unittest.main()
