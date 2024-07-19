"""Frequently used functions."""

from typing import Dict, List, Tuple

# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import math
from itertools import chain, combinations

import pennylane as qml
import torch

Tensor: TypeAlias = torch.Tensor
Observable: TypeAlias = qml.Hamiltonian


def probabilities_to_dictionary(probs: Tensor) -> Dict[str, Tensor]:
    """
    Convert 1D tensor of probabilities to dictionary of probabilities of bitstrings.

    :param probs: Probabilities.
    :type probs: Tensor
    :return: Probabilities of bitstrings.
    :rtype: Dict
    :raises ValueError: If length of probs is not a power of 2.
    """

    n = int(math.log2(len(probs)))
    if 2**n != len(probs):
        raise ValueError(f"Length of probs ({len(probs)}) is not a power of 2.")

    result = {}
    for i, p in enumerate(probs):
        bistring = bin(i)[2:].zfill(n)
        result[bistring] = p

    return result


def samples_to_dictionary(samples: Tensor) -> Dict[str, float]:
    """Convert 2D tensor of samples to dictionary of probabilities of bitstrings.

    :param samples: Samples.
    :type samples: Tensor
    :return: Probabilities of bitstrings.
    :rtype: Dict
    :raises ValueError: If samples not a Tensor of integers.
    """

    if samples.is_floating_point() or samples.is_complex():
        raise ValueError("Samples must be tensors of integers.")

    bitstrings = ["".join(str(b.item()) for b in sample) for sample in samples]
    bitstring_counts = {bs: bitstrings.count(bs) for bs in set(bitstrings)}
    total = sum(bitstring_counts.values())
    bitstring_probs = {bs: count / total for bs, count in bitstring_counts.items()}

    return bitstring_probs


def all_bin_sequences(n: int) -> List[Tuple[int, ...]]:
    """
    Generates all possible binary sequences of length n.

    :param n: The length of the binary sequences.
    :type n: int
    :return: A list of all binary sequences of length n,
        represented as a list of integers.
    :rtype: List[List[int]]
    """

    elements = list(range(n))
    return list(chain.from_iterable(combinations(elements, r) for r in range(n + 1)))


def parities_outcome(bitstring: str, H: Observable) -> float:
    """
    Compute the measurement outcome for a given bit string and Hamiltonian.
    Only works for Hamiltonians with identity and Pauli Z.

    :param bitstring: Input bit string.
    :type bitstring: string: str
    :param H: Hamiltonian.
    :type H: Observable
    :return: Real-valued outcome.
    :rtype: float
    :raises ValueError: If number of qubits does not match or
        operators other than I or Z are detected.
    """

    num_qubits = len(bitstring)
    num_wires = len(H.wires)

    if num_qubits != num_wires:
        raise ValueError(
            f"Number of qubits ({num_qubits}) " f"does not match number of wires ({num_wires})"
        )

    sum = 0.0
    for idx, O in enumerate(H.ops):
        if not isinstance(O.name, list):
            if O.name == "Identity":
                sum += H.coeffs[idx]
            elif O.name == "PauliZ":
                i = O.wires[0]
                sign = (-1) ** (int(bitstring[-1 - i]))
                sum += sign * H.coeffs[idx]
            else:
                raise ValueError("All operators must be PauliZ or Identity.")

        else:
            if not all(name == "PauliZ" for name in O.name):
                raise ValueError("All operators must be PauliZ or Identity.")

            sign = 1
            for w in O.wires:
                sign *= (-1) ** (int(bitstring[-1 - w]))

            sum += sign * H.coeffs[idx]

    return sum


def parities_outcome_probs(probs: Dict[str, float], H: Observable) -> Dict[float, float]:
    """
    Compute (real-valued) outputs with corresponding probabilities.

    :param probs: Dictionary of probabilities of bitstrings.
    :type probs: Dict
    :param H: Hamiltonian determening the outcome.
    :type H: Observable
    :return: Values with corresponding probabilities.
    :rtype: Dict
    """

    result = {}
    for b, p in probs.items():
        val = parities_outcome(b, H)
        result[val] = p

    return result
