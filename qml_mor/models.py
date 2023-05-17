from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Dict
from enum import Enum
import math

# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import torch
from torch.nn import Parameter
from torch.nn import Module
import pennylane as qml

QDevice: TypeAlias = qml.Device
Tensor: TypeAlias = torch.Tensor
Expectation: TypeAlias = qml.measurements.ExpectationMP
Observable: TypeAlias = qml.Hamiltonian
Probability: TypeAlias = qml.measurements.ProbabilityMP
Sample: TypeAlias = qml.measurements.SampleMP
X = TypeVar("X")
P = TypeVar("P")
T = TypeVar("T")


class ModelType(Enum):
    Expectation = "expectation"
    Probabilities = "probabilities"
    Samples = "samples"


class QNNModel(Module, Generic[X, T]):
    """Base class for a quantum neural network model."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: X) -> T:
        """Forward model evaluation."""
        pass


class IQPEReuploadSU2Parity(QNNModel[Tensor, Tensor]):
    """
    An IQP embedding circuit with additional SU(2) gates and parity measurements.

    Args:
        qdevice (QDevice): Quantum device.
        params (List[Tensor]): List of initiial parameters, 3 tensors.
        omega (float, optional): The exponential feature scaling factor.
            Defaults to 0.0.
        model_type (ModelType, optional): Specify type of model.
            Defaults to Expectation.

    Raises:
        ValueError: If params or model_type of invalid form.
    """

    def __init__(
        self,
        qdevice: QDevice,
        params: List[Tensor],
        omega: float = 0.0,
        model_type: ModelType = ModelType.Expectation,
    ) -> None:
        super().__init__()
        self._check_model_type(model_type)
        self._check_params(params)

        self.qdevice = qdevice
        self.init_theta = Parameter(params[0])
        self.theta = Parameter(params[1])
        self.W = Parameter(params[2])
        self.omega = omega
        self.model_type = model_type

    def forward(self, X: Tensor) -> Tensor:
        """
        Returns expectation values by default,
        or probabilities if stat_model set to True.

        Args:
            X (Tensor): Input tensor of dimension (dimX,) or (num_samples, dimX).
        """

        if self.model_type == ModelType.Expectation:
            qnode = qml.QNode(self.expectation, self.qdevice, interface="torch")
        elif self.model_type == ModelType.Probabilities:
            qnode = qml.QNode(self.probabilities, self.qdevice, interface="torch")
        elif self.model_type == ModelType.Samples:
            qnode = qml.QNode(self.sample, self.qdevice, interface="torch")

        if len(X.shape) == 1:
            out = qnode(X, self.parameters())
        else:
            Nx = X.size(0)
            out = torch.stack([qnode(X[k]) for k in range(Nx)])

        return out

    def expectation(self, x: Tensor) -> Expectation:
        """
        Returns the expectation value circuit of the Hamiltonian of the circuit given
        the input features and the parameters.

        Args:
            x (Tensor): The input features for the circuit of dimension (num_qubits,).

        Returns:
            Expectation: The expectation value of the Hamiltonian of the circuit.
        """
        H = self.Hamiltonian()

        return iqpe_reupload_su2_expectation(
            x, self.init_theta, self.theta, H, self.omega
        )

    def probabilities(self, x: Tensor) -> Probability:
        """
        Returns the probabilities of measurements for the circuit given
        the input features and the parameters.

        Args:
            x (Tensor): The input features for the circuit of dimension (num_qubits,).

        Returns:
            Probability: The measurement probabilities of the circuit.
        """

        return iqpe_reupload_su2_probs(x, self.init_theta, self.theta, self.omega)

    def sample(self, x: Tensor) -> Sample:
        """
        Returns the samples of measurements for the circuit given
        the input features and the parameters.

        Args:
            x (Tensor): The input features for the circuit of dimension (num_qubits,).

        Returns:
            Sample: The measurement samples of the circuit.
        """

        return iqpe_reupload_su2_sample(x, self.init_theta, self.theta, self.omega)

    def Hamiltonian(self) -> Observable:
        """
        Hamiltonian corresponding to the parity of Pauli Z operators.

        Returns:
            Observable: Observable corresponding to the parity of Pauli Z operators
        """

        num_qubits = math.log2(len(self.W))
        if not num_qubits.is_integer():
            raise ValueError(f"Length of W ({num_qubits}) not a power of 2")
        num_qubits = int(num_qubits)

        H = parity_hamiltonian(num_qubits, self.W)

        return H

    def outcome_probs(
        self, probs: Dict[str, float], params: List[Tensor]
    ) -> Dict[float, float]:
        """
        Compute (real-valued) outputs with corresponding probabilities.

        Args:
            probs (Dict): Dictionary of probabilities of bitstrings.
            params (List[Tensor]): QNN input and parameters.

        Returns:
            Dict: Values with corresponding probabilities.
        """

        H = self.Hamiltonian()
        outcomes = parities_outcome_probs(probs, H)

        return outcomes

    def _check_model_type(self, model_type: ModelType) -> None:
        if not isinstance(model_type, ModelType):
            raise ValueError(
                f"Invalid model_type. "
                f"Expected instance of ModelType, got {type(model_type)}."
            )

    def _check_params(self, params: List[Tensor]) -> None:
        length = len(params)
        if length != 3:
            raise ValueError(f"params must be a 3-dim tensor (not {length})")

        init_theta = params[0]
        theta = params[1]
        W = params[2]

        shape_init = init_theta.shape
        shape_theta = theta.shape
        shape_W = W.shape
        if len(shape_init) != 2:
            raise ValueError(f"init_theta (shape={shape_init}) must be a 2-dim tensor")
        if len(shape_theta) != 4:
            raise ValueError(f"theta (shape={shape_theta}) must be a 4-dim tensor")
        if shape_theta[3] != 2:
            raise ValueError(f"Last dimension of theta {shape_theta[3]} should be 2")
        if len(shape_W) != 1:
            raise ValueError(f"W (shape={shape_W}) must be a 1-dim tensor")


class Model(ABC, Generic[X, P, T]):
    """Abstract base class for regression models f(x, params)."""

    @abstractmethod
    def __call__(self, x: X, params: P) -> T:
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


def iqpe_reupload_su2_circuit(
    x: Tensor,
    init_theta: Tensor,
    theta: Tensor,
    omega: float = 0.0,
) -> None:
    """
    Quantum function that prepares QNN circuit defined by x and theta.

    Args:
        x (Tensor): Input tensor of shape (num_qubits,).
        init_theta (Tensor): Initial rotation angles for each qubit,
            of shape (reps, num_qubits).
        theta (Tensor): Rotation angles for each layer and each qubit,
            of shape (reps, num_layers, num_qubits-1, 2).
        omega (float, optional): Exponential feature scaling factor. Defaults to 0.0.

    Returns:
        None: Expectation value of the parity of Pauli Z operators

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


def iqpe_reupload_su2_expectation(
    x: Tensor,
    init_theta: Tensor,
    theta: Tensor,
    H: Observable,
    omega: float = 0.0,
) -> Expectation:
    """
    Quantum function that calculates the expectation value
    of a Hamiltonian H in the state defined by x and theta.

    Args:
        x (Tensor): Input tensor of shape (num_qubits,).
        init_theta (Tensor): Initial rotation angles for each qubit,
            of shape (reps, num_qubits).
        theta (Tensor): Rotation angles for each layer and each qubit,
            of shape (reps, num_layers, num_qubits-1, 2).
        H (Observable): Hamiltonian.
        omega (float, optional): Exponential feature scaling factor. Defaults to 0.0.

    Returns:
        Expectation: Expectation value of the parity of Pauli Z operators
    """

    iqpe_reupload_su2_circuit(x, init_theta, theta, omega)
    return qml.expval(H)


def iqpe_reupload_su2_probs(
    x: Tensor,
    init_theta: Tensor,
    theta: Tensor,
    omega: float = 0.0,
) -> Probability:
    """
    Quantum function that calculates the measurement probabilities
    in the state defined by x and theta.

    Args:
        x (Tensor): Input tensor of shape (num_qubits,).
        init_theta (Tensor): Initial rotation angles for each qubit,
            of shape (reps, num_qubits).
        theta (Tensor): Rotation angles for each layer and each qubit,
            of shape (reps, num_layers, num_qubits-1, 2).
        omega (float, optional): Exponential feature scaling factor. Defaults to 0.0.

    Returns:
        Probability: Measurement probabilities.
    """

    iqpe_reupload_su2_circuit(x, init_theta, theta, omega)
    return qml.probs()


def iqpe_reupload_su2_sample(
    x: Tensor,
    init_theta: Tensor,
    theta: Tensor,
    omega: float = 0.0,
) -> Probability:
    """
    Quantum function that calculates the measurement probabilities
    in the state defined by x and theta.

    Args:
        x (Tensor): Input tensor of shape (num_qubits,).
        init_theta (Tensor): Initial rotation angles for each qubit,
            of shape (reps, num_qubits).
        theta (Tensor): Rotation angles for each layer and each qubit,
            of shape (reps, num_layers, num_qubits-1, 2).
        omega (float, optional): Exponential feature scaling factor. Defaults to 0.0.

    Returns:
        Sample: Measurement samples.
    """

    iqpe_reupload_su2_circuit(x, init_theta, theta, omega)
    return qml.sample()


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


def parities_outcome(bitstring: str, H: Observable) -> float:
    """
    Compute the measurement outcome for a given bit string and Hamiltonian.
    Only works for Hamiltonians with identity and Pauli Z.

    Args:
        bitstring (str): Input bit string.
        H (Observable): Hamiltonian.

    Returns:
        float: Real-valued outcome.

    Raises:
        ValueError: If number of qubits does not match
            or operators other than I or Z are detected.
    """

    num_qubits = len(bitstring)
    num_wires = len(H.wires)

    if num_qubits != num_wires:
        raise ValueError(
            f"Number of qubits ({num_qubits}) "
            f"does not match number of wires ({num_wires})"
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


def parities_outcome_probs(
    probs: Dict[str, float], H: Observable
) -> Dict[float, float]:
    """
    Compute (real-valued) outputs with corresponding probabilities.

    Args:
        probs (Dict): Dictionary of probabilities of bitstrings.
        H (Observable): Hamiltonian determening the outcome.

    Returns:
        Dict: Values with corresponding probabilities.
    """

    result = {}
    for b, p in probs.items():
        val = parities_outcome(b, H)
        result[val] = p

    return result
