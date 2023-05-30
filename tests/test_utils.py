import pennylane as qml
import torch
from collections import Counter
from qulearn.utils import (
    probabilities_to_dictionary,
    samples_to_dictionary,
    all_bin_sequences,
    parities_outcome,
    parities_outcome_probs,
)


# probabilities_to_dictionary
def test_probabilities_to_dictionary():
    probs = torch.tensor([0.1, 0.9])
    assert probabilities_to_dictionary(probs) == {"0": 0.1, "1": 0.9}


# samples_to_dictionary
def test_samples_to_dictionary():
    samples = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    assert samples_to_dictionary(samples) == {"01": 0.5, "10": 0.5}


# all_bin_sequences
def test_all_bin_sequences():
    assert Counter(map(tuple, all_bin_sequences(2))) == Counter(
        map(tuple, [[0, 1], [1], [0], []])
    )


# parities_outcome
def test_parities_outcome():
    bitstring = "01"
    coeffs = [1, -1]
    obs = [qml.Identity(0), qml.PauliZ(1)]
    H = qml.Hamiltonian(coeffs, obs)
    assert parities_outcome(bitstring, H) == 0.0


# parities_outcome_probs
def test_parities_outcome_probs():
    probs = {"01": 0.5, "10": 0.5}
    coeffs = [1, -1]
    obs = [qml.Identity(0), qml.PauliZ(1)]
    H = qml.Hamiltonian(coeffs, obs)
    assert parities_outcome_probs(probs, H) == {0.0: 0.5, 2.0: 0.5}
