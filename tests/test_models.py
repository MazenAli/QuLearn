import unittest
import torch
import pennylane as qml
from qml_mor.models import (
    LinearModel,
    IQPEReuploadSU2Parity,
    parity_hamiltonian,
    iqpe_reupload_su2_circuit,
    iqpe_reupload_su2_expectation,
    iqpe_reupload_su2_probs,
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
    def test_omega(self):
        # Test that the getter and setter for omega work correctly
        model = IQPEReuploadSU2Parity()
        model.omega = 0.5
        self.assertEqual(model.omega, 0.5)

    def test_qfunction(self):
        # Test that the qfunction returns a PennyLane Expectation object
        model = IQPEReuploadSU2Parity()
        output = model.qfunction(X, [INIT_THETA, THETA, W])
        self.assertIsInstance(output, qml.measurements.ExpectationMP)

    def test_probs(self):
        # Test that the qfunction returns a PennyLane Probability object
        model = IQPEReuploadSU2Parity()
        output = model.probabilities(X, [INIT_THETA, THETA, W])
        self.assertIsInstance(output, qml.measurements.ProbabilityMP)

    def test_qfunction_output(self):
        # Test that the qfunction is equal to 8 on a trivial zero input
        model = IQPEReuploadSU2Parity()
        dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)

        @qml.qnode(dev, interface="torch")
        def circuit(x, params):
            return model.qfunction(x, params)

        output = circuit(X, [INIT_THETA, THETA, W]).item()
        self.assertAlmostEqual(output, 8.0)

    def test_probs_output(self):
        # Test that the qfunction is equal to 8 on a trivial zero input
        model = IQPEReuploadSU2Parity()
        dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)

        @qml.qnode(dev, interface="torch")
        def circuit(x, params):
            return model.probabilities(x, params)

        output = circuit(X, [INIT_THETA, THETA, W])
        self.assertAlmostEqual(output[0].item(), 1.0)


class TestIqpeReuploadSu2Parity(unittest.TestCase):
    def test_error(self):
        # Test that the output of iqpe_reupload_su2 has the correct shape
        with self.assertRaises(ValueError):
            parity_hamiltonian(len(X), torch.ones((5)))

    def test_output_shape(self):
        # Test that the output of iqpe_reupload_su2 has the correct shape
        H = parity_hamiltonian(len(X), W)
        output = iqpe_reupload_su2_expectation(X, INIT_THETA, THETA, H)
        self.assertEqual(output.shape(), (1,))

    def test_raises_value_error(self):
        # Test that iqpe_reupload_su2 raises a ValueError for incorrect
        # input shapes
        H = parity_hamiltonian(len(X), W)
        with self.assertRaises(ValueError):
            iqpe_reupload_su2_circuit(X, INIT_THETA, torch.ones((2, NUM_QUBITS, 4)), H)

    def test_output_type(self):
        H = parity_hamiltonian(len(X), W)
        # Test that the output of iqpe_reupload_su2_parity is a measurement process
        output = iqpe_reupload_su2_expectation(X, INIT_THETA, THETA, H)
        self.assertIsInstance(output, qml.measurements.ExpectationMP)

    def test_output_probs_type(self):
        # Test that the output of iqpe_reupload_su2_parity is a probability process
        output = iqpe_reupload_su2_probs(X, INIT_THETA, THETA)
        self.assertIsInstance(output, qml.measurements.ProbabilityMP)


class TestLinearModel(unittest.TestCase):
    def test_output_shape(self):
        # Test that the output of linear model has the correct shape and type
        model = LinearModel()
        sizex = 3
        X = torch.zeros(sizex)
        P = [torch.zeros(sizex + 1, requires_grad=True)]
        output = model(X, P)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, torch.Size([]))

    def test_raises_value_error(self):
        # Test that iqpe_reupload_su2_parity raises a ValueError for incorrect
        # input shapes
        model = LinearModel()
        sizex = 3
        X = torch.zeros(sizex)
        P = [torch.zeros(sizex, requires_grad=True)]
        with self.assertRaises(ValueError):
            model(X, P)


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


if __name__ == "__main__":
    unittest.main()
