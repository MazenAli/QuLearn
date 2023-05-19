import unittest
import torch
import pennylane as qml
from qml_mor.models import (
    ModelType,
    IQPEReuploadSU2Parity,
    parity_hamiltonian,
    iqpe_reupload_su2_circuit,
    iqpe_reupload_su2_expectation,
    iqpe_reupload_su2_probs,
    sequence_generator,
    parities,
    parities_outcome,
    parities_outcome_probs,
)
from qml_mor.utils import probabilities_to_dictionary, samples_to_dictionary


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
        params = [INIT_THETA, THETA, W]
        dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)
        model = IQPEReuploadSU2Parity(dev, params, omega=0.5)
        self.assertEqual(model.omega, 0.5)

    def test_qfunction(self):
        # Test that the qfunction returns a PennyLane Expectation object
        params = [INIT_THETA, THETA, W]
        dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)
        model = IQPEReuploadSU2Parity(dev, params)
        output = model.expectation(X)
        self.assertIsInstance(output, qml.measurements.ExpectationMP)

    def test_probs(self):
        # Test that the qfunction returns a PennyLane Probability object
        dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)
        params = [INIT_THETA, THETA, W]
        model = IQPEReuploadSU2Parity(dev, params, model_type=ModelType.Probabilities)
        output = model.probabilities(X)
        self.assertIsInstance(output, qml.measurements.ProbabilityMP)

    def test_qfunction_output(self):
        # Test that the qfunction is equal to 8 on a trivial zero input
        dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)
        params = [INIT_THETA, THETA, W]
        model = IQPEReuploadSU2Parity(dev, params)
        output = model(X).item()
        self.assertAlmostEqual(output, 8.0)

    def test_probs_output(self):
        # Test that the qfunction is equal to 8 on a trivial zero input
        dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)
        params = [INIT_THETA, THETA, W]
        model = IQPEReuploadSU2Parity(dev, params, model_type=ModelType.Probabilities)
        output = model(X)
        self.assertAlmostEqual(output[0].item(), 1.0)

    def test_probs_outcome(self):
        # Test that the qfunction is equal to 8 on a trivial zero input
        dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)
        params = [INIT_THETA, THETA, W]
        model = IQPEReuploadSU2Parity(dev, params, model_type=ModelType.Probabilities)
        probs = model(X)
        prob_dic = probabilities_to_dictionary(probs)

        outcome = model.outcome_probs(prob_dic, params)
        E = 0.0
        for val, p in outcome.items():
            E += val * p

        @qml.qnode(dev, interface="torch")
        def circuit(x, params):
            return model.expectation(x, params)

        model = IQPEReuploadSU2Parity(dev, params)
        expec = model(X)

        self.assertAlmostEqual(expec.item(), E.item())

    def test_samples_outcome(self):
        # Test that the qfunction is equal to 8 on a trivial zero input
        shots = 10
        dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=shots)
        params = [INIT_THETA, THETA, W]
        model = IQPEReuploadSU2Parity(dev, params, model_type=ModelType.Samples)
        probs = model(X)
        prob_dic = samples_to_dictionary(probs)

        outcome = model.outcome_probs(prob_dic, params)
        E = 0.0
        for val, p in outcome.items():
            E += val * p
        model = IQPEReuploadSU2Parity(dev, params, model_type=ModelType.Expectation)
        expec = model(X)
        self.assertAlmostEqual(expec.item(), E.item())


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


class TestParitiesOutcome(unittest.TestCase):
    def test_parities_outcome(self):
        bitstring = "0101"
        H = [qml.PauliZ(0) @ qml.PauliZ(2), qml.Identity(1), qml.Identity(3)]
        H = qml.Hamiltonian([1.0, 0.0, 0.0], H)
        outcome = parities_outcome(bitstring, H)
        self.assertEqual(outcome, 1.0)

    def test_parities_outcome_invalid_operator(self):
        bitstring = "0101"
        H = qml.PauliZ(0) @ qml.PauliX(2)
        with self.assertRaises(ValueError):
            parities_outcome(bitstring, H)

    def test_parities_outcome_probs(self):
        probs = {"0100": 0.5, "1010": 0.5}
        H = [qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)]
        H = qml.Hamiltonian([0.8], H)
        outcomes = parities_outcome_probs(probs, H)
        self.assertEqual(outcomes, {0.8: 0.5, -0.8: 0.5})

    def test_parities_outcome_probs_invalid_operator(self):
        probs = {"0101": 0.5, "1010": 0.5}
        H = qml.PauliZ(0) @ qml.PauliX(2)
        H.return_type = None
        with self.assertRaises(ValueError):
            parities_outcome_probs(probs, H)


if __name__ == "__main__":
    unittest.main()
