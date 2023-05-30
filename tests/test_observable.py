import pytest
import torch
import pennylane as qml
from qulearn.observable import parity_all_hamiltonian, parities_all_observables


def test_parity_all_hamiltonian():
    # Test case 1
    num_qubits = 2
    weights = torch.tensor([1, 2, 3, 4])
    H = parity_all_hamiltonian(num_qubits, weights)
    assert isinstance(H, qml.Hamiltonian)
    assert len(H.ops) == 4
    assert len(H.coeffs) == 4

    # Test case 2
    num_qubits = 3
    weights = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    H = parity_all_hamiltonian(num_qubits, weights)
    assert isinstance(H, qml.Hamiltonian)
    assert len(H.ops) == 8
    assert len(H.coeffs) == 8


def test_parity_all_hamiltonian_invalid_weights():
    # Test case with invalid weights
    num_qubits = 2
    weights = torch.tensor([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        parity_all_hamiltonian(num_qubits, weights)


def test_parity_all_hamiltonian_invalid_num_qubits():
    # Test case with mismatched num_qubits and weights shape
    num_qubits = 2
    weights = torch.tensor([1, 2, 3])
    with pytest.raises(ValueError):
        parity_all_hamiltonian(num_qubits, weights)


def test_parities_all_observables():
    # Test case 1
    n = 2
    observables = parities_all_observables(n)
    assert len(observables) == 4
    assert observables[0].name[0] == "PauliZ"
    assert observables[3].name == "Identity"

    # Test case 2
    n = 3
    observables = parities_all_observables(n)
    assert len(observables) == 8
    assert observables[0].name[0] == "PauliZ"
    assert observables[7].name == "Identity"
