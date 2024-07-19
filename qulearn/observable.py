import math
from typing import List

import pennylane as qml

from .types import Hamiltonian, Observable, ParitySequence, Tensor
from .utils import all_bin_sequences


def parity_all_hamiltonian(num_qubits: int, weights: Tensor) -> Hamiltonian:
    """
    Hamiltonian corresponding to the parity of all combinations of Pauli Z operators.

    :param num_qubits: Number of qubits.
    :type num_qubits: int
    :param weights: Observable weights of shape (2^num_qubits,)
    :type weights: Tensor
    :return: Observable corresponding to the parity of Pauli Z operators
    :rtype: Observable
    :raises ValueError: If shape of weights is not as specified above.
    """

    shapeW = weights.shape
    if len(shapeW) != 1:
        raise ValueError(f"W (shape={shapeW}) must be a 1-dim tensor")

    W_qubits = math.log2(len(weights))
    if not W_qubits.is_integer():
        raise ValueError(f"Length of W ({W_qubits}) not a power of 2")

    if int(W_qubits) != num_qubits:
        raise ValueError(f"num_qubits and W (shape={shapeW}) do not match")

    obs = parities_all_observables(num_qubits)
    H = qml.Hamiltonian(weights, obs)

    return H


def parities_all_observables(n: int) -> List[Observable]:
    """
    Generates a list of observables corresponding to the parity of all
    possible binary combinations of n qubits.

    :param n: The number of qubits.
    :type n: int
    :return: A list of observables corresponding to the parity of all
        possible binary combinations of n qubits plus Idenity.
    :rtype: List[Observable]
    """

    seq = all_bin_sequences(n)
    return sequence2parity_observable(seq)


def sequence2parity_observable(parity_sequence: ParitySequence) -> List[Observable]:
    """
    Generates a list of observables corresponding to the parity of
    the indeces defined by the sequence.

    :param parity_sequence: The sequence of Z parities.
    :type parity_sequence: ParitySequence
    :return: A list of observables corresponding to the parity of all
        possible binary combinations of n qubits plus Idenity.
    :rtype: List[Observable]
    """

    ops = []
    for par in parity_sequence:
        if par:
            tmp = qml.PauliZ(par[0])
            if len(par) > 1:
                for i in par[1:]:
                    tmp = tmp @ qml.PauliZ(i)

            ops.append(tmp)

    ops.append(qml.Identity(0))

    return ops
