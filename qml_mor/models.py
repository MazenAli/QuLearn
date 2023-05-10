from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic
import math

# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import torch
import pennylane as qml

Tensor: TypeAlias = torch.Tensor
QFuncOutput: TypeAlias = qml.measurements.ExpectationMP
Observable: TypeAlias = qml.operation.Observable
X = TypeVar("X")
P = TypeVar("P")
Y = TypeVar("Y")


class QNNModel(ABC, Generic[X, P]):
    """Abstract base class for a quantum neural network model."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def qfunction(self, x: X, params: P) -> QFuncOutput:
        """Abstract method for the quantum function."""
        pass

    def circuit(self, x: X, params: P) -> QFuncOutput:
        """Constructs a circuit for a given input and parameters."""
        return self.qfunction(x, params)


class IQPEReuploadSU2Parity(QNNModel[Tensor, List[Tensor]]):
    """
    An IQP embedding circuit with additional SU(2) gates and parity measurements.

    Args:
        omega (float, optional): The exponential feature scaling factor.
            Defaults to 0.0.
    """

    def __init__(self, omega: float = 0.0) -> None:
        self.omega = omega

    def qfunction(self, x: Tensor, params: List[Tensor]) -> QFuncOutput:
        """
        Returns the expectation value circuit of the Hamiltonian of the circuit given
        the input features and the parameters.

        Args:
            x (Tensor): The input features for the circuit of dimension (sizex,).
            params (List[Tensor]): The parameters for the circuit. Must be a list of
                three tensors: the initial thetas, the main thetas, and the weights W.

        Returns:
            QFuncOutput: The expectation value of the Hamiltonian of the circuit.

        Raises:
            ValueError: If the length of params is not 3.
        """

        if len(params) != 3:
            raise ValueError("Parameters must be a list of 3 tensors")

        init_theta = params[0]
        theta = params[1]
        H = self.Hamiltonian(params)

        return iqpe_reupload_su2(x, init_theta, theta, H, self.omega)

    def Hamiltonian(self, params: List[Tensor]) -> Observable:
        """
        Hamiltonian corresponding to the parity of Pauli Z operators.

        Args:
            params (List[Tensor]): The parameters for the circuit. Must be a list of
                three tensors: the initial thetas, the main thetas, and the weights W.

        Returns:
            Observable: Observable corresponding to the parity of Pauli Z operators

        Raises:
            ValueError: If shape of params or W incorrect.
        """

        if len(params) != 3:
            raise ValueError("Parameters must be a list of 3 tensors")

        W = params[2]
        shapeW = W.shape
        if len(shapeW) != 1:
            raise ValueError(f"W (shape={shapeW}) must be a 1-dim tensor")

        num_qubits = math.log2(len(W))
        if not num_qubits.is_integer():
            raise ValueError(f"Length of W ({num_qubits}) not a power of 2")
        num_qubits = int(num_qubits)

        H = parity_hamiltonian(num_qubits, W)

        return H


class Model(ABC, Generic[X, P, Y]):
    """Abstract base class for regression models f(x, params)."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, x: X, params: P) -> Y:
        """Abstract method for call."""
        pass


class LinearModel(Model[Tensor, List[Tensor], Tensor]):
    """
    Linear regression model
    f(x, params) = x^T*params[0][0:len(x)] + params[0][len(x)]
    """

    def __call__(self, x: Tensor, params: List[Tensor]) -> Tensor:
        """
        Evaluate linear model for given parameters.

        Args:
            x (Tensor): feature tensor x.
            params (List[Tensor]): List of parameter tensors. Should have lenght 1.

        Returns:
            Tensor: Value of model evaluated at x and params.
                Computational graph attached.

        Raises:
            ValueError: If len(x)+1!=len(params[0]).
        """

        sizex = len(x)
        sizep = len(params[0])

        if sizex + 1 != sizep:
            raise ValueError(
                f"Tensors x with size {sizex} and params[0] with size {sizep} "
                "should satisfy sizex+1=sizep!"
            )

        res = torch.sum(x * params[0][0:sizex]) + params[0][sizex]
        return res


def parity_hamiltonian(num_qubits: int, W: Tensor) -> Observable:
    """
    Hamiltonian corresponding to the parity of Pauli Z operators.

    Args:
        num_qubits (int): Number of qubits.
        W (Tensor): Observable weights of shape (2^num_qubits,)

    Returns:
        Observable: Observable corresponding to the parity of Pauli Z operators

    Raises:
        ValueError: If shape of W is not as specified above.
    """

    shapeW = W.shape
    if len(shapeW) != 1:
        raise ValueError(f"W (shape={shapeW}) must be a 1-dim tensor")

    W_qubits = math.log2(len(W))
    if not W_qubits.is_integer():
        raise ValueError(f"Length of W ({W_qubits}) not a power of 2")

    if int(W_qubits) != num_qubits:
        raise ValueError(f"num_qubits and W (shape={shapeW}) do not match")

    obs = parities(num_qubits)
    H = qml.Hamiltonian(W, obs)

    return H


def iqpe_reupload_su2(
    x: Tensor, init_theta: Tensor, theta: Tensor, H: Observable, omega: float = 0.0
) -> QFuncOutput:
    """
    Quantum function that calculates the expectation value
    of a Hamiltonian H in the state defined by x and theta.

    Args:
        x (Tensor): Input tensor of shape (num_qubits,)
        init_theta (Tensor): Initial rotation angles for each qubit,
            of shape (reps, num_qubits)
        theta (Tensor): Rotation angles for each layer and each qubit,
            of shape (reps, num_layers, num_qubits-1, 2)
        W (Tensor): Observable weights of shape (2^num_qubits,)
        omega (float, optional): Exponential feature scaling factor. Defaults to 0.0.

    Returns:
        QFuncOutput: Expectation value of the parity of Pauli Z operators

    Raises:
        ValueError: If any of the tensors does not have shape as specified above.
    """

    shape_x = x.shape
    shape_init = init_theta.shape
    shape = theta.shape
    if len(shape_x) != 1:
        raise ValueError(f"x (shape={shape_x}) must be a 1-dim tensor")
    if len(shape_init) != 2:
        raise ValueError(f"init_theta (shape={shape_init}) must be a 2-dim tensor")
    if len(shape) != 4:
        raise ValueError(f"theta (shape={shape}) must be a 4-dim tensor")
    if shape[3] != 2:
        raise ValueError(f"Last dimension of theta {shape[3]} should be 2")

    num_qubits = len(x)
    init_qubits = shape_init[1]
    theta_qubits = shape[2]
    if num_qubits != init_qubits:
        raise ValueError(
            f"num_qubits in x (shape={shape_x}) and "
            f"init_theta (shape={shape_init}) do not match"
        )
    if num_qubits != theta_qubits + 1:
        raise ValueError(
            f"num_qubits in x (shape={x.shape}) and "
            f"theta (shape={shape}) do not match"
        )

    reps = shape_init[0]
    reps_theta = shape[0]
    if reps != reps_theta:
        raise ValueError(
            f"reps in init_theta ({reps}) does not match reps in theta ({reps_theta})"
        )

    wires = range(num_qubits)
    for layer in range(reps):
        features = 2 ** (omega * layer) * x
        initial_layer_weights = init_theta[layer]
        weights = theta[layer]

        qml.IQPEmbedding(features=features, wires=wires)
        qml.SimplifiedTwoDesign(
            initial_layer_weights=initial_layer_weights,
            weights=weights,
            wires=wires,
        )

    return qml.expval(H)


def sequence_generator(n: int) -> List[List[int]]:
    """
     Generates all possible binary sequences of length n.

    Args:
        n (int): The length of the binary sequences.

    Returns:
        List[List[int]]: A list of all binary sequences of length n,
            represented as a list of integers.

    """

    if n == 0:
        return [[]]
    else:
        sequences = []
        for sequence in sequence_generator(n - 1):
            sequences.append(sequence + [n - 1])
            sequences.append(sequence)
        return sequences


def parities(n: int) -> List[Observable]:
    """
    Generates a list of observables corresponding to the parity of all
    possible binary combinations of n qubits.

    Args:
        n (int): The number of qubits.

    Returns:
        List[Observable]: A list of observables corresponding to the parity of all
            possible binary combinations of n qubits.

    """

    seq = sequence_generator(n)
    ops = []
    for par in seq:
        if par:
            tmp = qml.PauliZ(par[0])
            if len(par) > 1:
                for i in par[1:]:
                    tmp = tmp @ qml.PauliZ(i)

            ops.append(tmp)

    ops.append(qml.Identity(0))

    return ops
