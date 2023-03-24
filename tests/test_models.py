import unittest
import torch
import pennylane as qml
from qml_mor.models import (
    IQPEReuploadSU2Parity,
    iqpe_reupload_su2_parity,
    sequence_generator,
    parities,
)

# Set up global constants for testing
NUM_QUBITS = 3
THETA_SHAPE = (2, 2, NUM_QUBITS - 1, 2)
W_SHAPE = 2**NUM_QUBITS
X = torch.tensor([0.0, 0.0, 0.0], requires_grad=False)
INIT_THETA = torch.zeros((2, NUM_QUBITS), requires_grad=True)
THETA = torch.zeros(THETA_SHAPE, requires_grad=True)
W = torch.ones(W_SHAPE, requires_grad=True)


class TestIQPEReuploadSU2Parity(unittest.TestCase):
    def test_params(self):
        # Test that the getter and setter for params work correctly
        model = IQPEReuploadSU2Parity([INIT_THETA, THETA, W])
        new_params = [
            torch.ones((2, NUM_QUBITS)),
            torch.ones(THETA_SHAPE),
            torch.ones(W_SHAPE),
        ]
        model.params = new_params
        self.assertEqual(model.params, new_params)

    def test_omega(self):
        # Test that the getter and setter for omega work correctly
        model = IQPEReuploadSU2Parity([INIT_THETA, THETA, W])
        model.omega = 0.5
        self.assertEqual(model.omega, 0.5)

    def test_qfunction(self):
        # Test that the qfunction returns a PennyLane Expectation object
        model = IQPEReuploadSU2Parity([INIT_THETA, THETA, W])
        output = model.qfunction(X, [INIT_THETA, THETA, W])
        self.assertIsInstance(output, qml.measurements.ExpectationMP)

    def test_qfunction_output(self):
        # Test that the qfunction is equal to 8 on a trivial zero input
        model = IQPEReuploadSU2Parity([INIT_THETA, THETA, W])
        dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)

        @qml.qnode(dev, interface="torch")
        def circuit(x, params):
            return model.qfunction(x, params)

        output = circuit(X, [INIT_THETA, THETA, W]).item()
        self.assertAlmostEqual(output, 8.0)


class TestIqpeReuploadSu2Parity(unittest.TestCase):
    def test_output_shape(self):
        # Test that the output of iqpe_reupload_su2_parity has the correct shape
        output = iqpe_reupload_su2_parity(X, INIT_THETA, THETA, W)
        self.assertEqual(output.shape(), (1,))

    def test_raises_value_error(self):
        # Test that iqpe_reupload_su2_parity raises a ValueError for incorrect
        # input shapes
        with self.assertRaises(ValueError):
            iqpe_reupload_su2_parity(X, INIT_THETA, torch.ones((2, NUM_QUBITS, 4)), W)

    def test_output_type(self):
        # Test that the output of iqpe_reupload_su2_parity is a measurement process
        output = iqpe_reupload_su2_parity(X, INIT_THETA, THETA, W)
        self.assertIsInstance(output, qml.measurements.ExpectationMP)


class TestSequenceGenerator(unittest.TestCase):
    def test_sequence_length(self):
        # Test that sequence_generator returns the correct number of sequences
        sequences = sequence_generator(NUM_QUBITS)
        self.assertEqual(len(sequences), 2**NUM_QUBITS)

    def test_sequence_type(self):
        # Test that the elements of the sequences returned by sequence_generator
        # are lists
        sequences = sequence_generator(NUM_QUBITS)
        for seq in sequences:
            self.assertIsInstance(seq, list)


class TestParities(unittest.TestCase):
    def test_parity_length(self):
        # Test that the number of observables returned by parities is correct
        observables = parities(NUM_QUBITS)
        self.assertEqual(len(observables), 2**NUM_QUBITS)

    def test_parity_type(self):
        # Test that the elements of the observables returned by parities are
        # PennyLane observables
        observables = parities(NUM_QUBITS)
        for obs in observables:
            self.assertIsInstance(obs, qml.operation.Observable)
