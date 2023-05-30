# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

from typing import Iterable, Any, Optional, Union, List

from enum import Enum
import math
import torch
from torch import nn
import pennylane as qml

DEFAULT_QDEV_CFG = {"name": "default.qubit", "shots": None}

QDevice: TypeAlias = qml.Device
QNode: TypeAlias = qml.QNode
Tensor: TypeAlias = torch.Tensor
Wires: TypeAlias = Union[int, Iterable[Any]]
Expectation: TypeAlias = qml.measurements.ExpectationMP
Observable: TypeAlias = qml.operation.Observable
Probability: TypeAlias = qml.measurements.ProbabilityMP
Sample: TypeAlias = qml.measurements.SampleMP


class MeasurementType(Enum):
    """Measurement type for a measurement layer."""

    """Expectation: return expected value of observable."""
    Expectation = "expectation"
    """Probabilities: return vector of probabilities."""
    Probabilities = "probabilities"
    """Samples: return measurement samples."""
    Samples = "samples"


class CircuitLayer(nn.Module):
    """
    Base class for a quantum circuit layer.

    A circuit layer transforms a quantum state but does not perform any measurements.
    Thus, a circuit layer produces no classical output.
    It can be combined with a measurement layer that returns classical output.
    By default, the base class does nothing (zero state).
    Derived classes need to override :meth:`circuit`.

    :param wires: Number or list of circuits.
    :type wires: Wires
    """

    def __init__(self, wires: Wires) -> None:
        super().__init__()
        self.set_wires(wires)

    def forward(self, x: Tensor) -> None:
        """Forward pass. See :meth:`circuit`"""
        self.circuit(x)

    def circuit(self, _: Tensor) -> None:
        """Applies input-dependent unitary transformations to a circuit.

        :param x: Input data samples.
        :type x: Tensor.
        :return: None for the base class.
        :rtype: None.
        """
        return None

    def set_wires(self, wires: Wires) -> None:
        """Set circuit (qubit) wires."""
        if isinstance(wires, int):
            self.num_wires = wires
            self.wires = list(range(wires))
        else:
            self.num_wires = len(list(wires))
            self.wires = list(wires)


class MeasurementLayer(nn.Module):
    """
    Base class for measurment layers.

    Measurment layers are appended to quantum circuits and return classical output.

    :param circuits: Quantum circuits before the measurement.
    :type circuits: tuple
    :param qdevice: Quantum device. If None, the default device will be used.
    :type qdevice: Optional[QDevice], defaults to None
    :param measurement_type: Type of quantum measurement.
    :type measurement_type: MeasurementType, defaults to MeasurementType.Probabilities
    :param observable: Observable to measure.
        None only works with Probabilities and Samples.
    :type observable: Optional[Observable], defaults to None
    :param kwargs: Additional keyword arguments for qml.QNode.
    """

    def __init__(
        self,
        *circuits,
        qdevice: Optional[QDevice] = None,
        measurement_type: MeasurementType = MeasurementType.Probabilities,
        observable: Optional[Observable] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.circuits = nn.ModuleList(circuits)

        if qdevice is None:
            self.qdevice = qml.device(wires=self.circuits[0].wires, **DEFAULT_QDEV_CFG)
        else:
            self.qdevice = qdevice

        self.wires = self.circuits[0].wires
        self.measurement_type = measurement_type
        self.observable = observable
        self.interface = kwargs.pop("interface", "torch")
        self.kwargs = kwargs
        self.qnode = None

        self.check_measurement_type()

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass, depending on measurement type.
        See :meth:`expectation`, :meth:`probabilities` and :meth:`samples`.

        :param x: Input data samples.
        :type circuits: Tensor.
        :return: Forward evaluation of model on data.
        :rtype: Tensor
        """
        qnode = self.set_qnode()

        if x is not None:
            if len(x.shape) == 1:
                out = qnode(x)
                if self.measurement_type == MeasurementType.Expectation:
                    out = torch.unsqueeze(out, 0)
            else:
                Nx = x.size(0)
                outs = []
                for k in range(Nx):
                    out_k = qnode(x[k])
                    if self.measurement_type == MeasurementType.Expectation:
                        out_k = torch.unsqueeze(out_k, 0)
                    outs.append(out_k)

                out = torch.stack(outs)
        elif self.measurement_type == MeasurementType.Expectation:
            out = qnode(x)
            out = torch.unsqueeze(out, 0)
        else:
            out = qnode(x)

        return out

    def expectation(self, x: Optional[Tensor] = None) -> Expectation:
        """
        Calculate the expectation value of the observable for given circuits.

        :param x: Input tensor that is passed to the quantum circuits, defaults to None.
        :type x: Optional[Tensor]
        :return: Expectation value object for the observable.
        :rtype: Expectation
        """
        for circuit in self.circuits:
            circuit(x)
        expec = qml.expval(self.observable)
        return expec

    def probabilities(self, x: Optional[Tensor] = None) -> Probability:
        """
        Calculate the outcome probabilities for given circuits.

        :param x: Input tensor that is passed to the quantum circuits, defaults to None.
        :type x: Optional[Tensor]
        :return: Probabilities of the outcomes of the circuits.
        :rtype: Probability
        """
        for circuit in self.circuits:
            circuit(x)
        probs = qml.probs(wires=self.wires)
        return probs

    def samples(self, x: Optional[Tensor] = None) -> Sample:
        """
        Sample the outcomes of the given circuits.

        :param x: Input tensor that is passed to the quantum circuits, defaults to None.
        :type x: Optional[Tensor]
        :return: Samples of the outcomes of the circuits.
        :rtype: Sample
        """
        for circuit in self.circuits:
            circuit(x)
        sample = qml.sample(wires=self.wires)
        return sample

    def set_qnode(self) -> QNode:
        """
        Set the quantum node for the layer and measurement type.

        :return: The set QNode.
        :rtype: QNode
        """

        if self.measurement_type == MeasurementType.Expectation:
            circuit = self.expectation
        elif self.measurement_type == MeasurementType.Probabilities:
            circuit = self.probabilities
        elif self.measurement_type == MeasurementType.Samples:
            circuit = self.samples

        qnode = qml.QNode(
            circuit, self.qdevice, interface=self.interface, **self.kwargs
        )
        self.qnode = qnode
        return self.qnode

    def check_measurement_type(self) -> None:
        """
        Check if the measurement type is valid.
        Raises errors for invalid measurement types.

        :raises NotImplementedError: If the measurement type is not recognized.
        :raises ValueError: If the expectation measurement type doesn't have an
            observable, or if the sample measurement type doesn't have an
            integer number of shots.
        """

        if not isinstance(self.measurement_type, MeasurementType):
            raise NotImplementedError(
                f"Measurement type ({self.measurement_type}) not recognized"
            )
        if self.measurement_type == MeasurementType.Expectation:
            if self.observable is None:
                raise ValueError(
                    f"Measurement type ({self.measurement_type}) "
                    "requires an observable"
                )
        if (
            self.measurement_type == MeasurementType.Samples
            and self.qdevice.shots is None
        ):
            raise ValueError(
                f"Measurement type ({self.measurement_type}) "
                "requires integer number of shots"
            )


class IQPEmbeddingLayer(CircuitLayer):
    """
    Layer for IQP (Instantaneous Quantum Polynomial) embedding.

    :param wires: The wires to be used by the layer
    :type wires: Wires
    :param n_repeat: The number of times to repeat the IQP embedding, defaults to 1
    :type n_repeat: int, optional
    :param kwargs: Extra arguments passed to the IQP embedding
    """

    def __init__(self, wires: Wires, n_repeat: int = 1, **kwargs) -> None:
        super().__init__(wires)
        self.n_repeat = n_repeat
        self.kwargs = kwargs
        self.qfunc = qml.IQPEmbedding

    def circuit(self, x: Tensor) -> None:
        """
        Define the quantum circuit for this layer.

        :param x: Input tensor that is passed to the quantum circuit.
        :type x: Tensor
        """
        self.qfunc(x, self.wires, self.n_repeat, **self.kwargs)


class RYCZLayer(CircuitLayer):
    """
    Layer for the RYCZ (Rotation around Y and Controlled-Z) gates.

    :param wires: The wires to be used by the layer.
    :type wires: Wires
    :param n_layers: The number of layers for the simplified two-design architecture,
        defaults to 1.
    :type n_layers: int, optional
    :param cdevice: Classical device to store the initial layer weights
        and internal layer weights.
    :param dtype: Data type of the weights
    :param kwargs: Extra arguments passed to the SimplifiedTwoDesign.
    """

    def __init__(
        self, wires: Wires, n_layers: int = 1, cdevice=None, dtype=None, **kwargs
    ) -> None:
        super().__init__(wires)

        self.n_layers = n_layers
        self.qfunc = qml.SimplifiedTwoDesign
        self.cdevice = cdevice
        self.dtype = dtype
        self.kwargs = kwargs

        # weight parameters
        self.initial_layer_weights = torch.nn.Parameter(
            torch.empty(self.num_wires, device=self.cdevice, dtype=self.dtype)
        )
        self.weights = torch.nn.Parameter(
            torch.empty(
                (self.n_layers, self.num_wires - 1, 2),
                device=self.cdevice,
                dtype=self.dtype,
            )
        )
        nn.init.uniform_(self.initial_layer_weights, a=0.0, b=2 * math.pi)
        nn.init.uniform_(self.weights, a=0.0, b=2 * math.pi)

    def circuit(self, _: Optional[Tensor] = None) -> None:
        """
        Define the quantum circuit for this layer.

        :param _: Input tensor that is passed to the quantum circuit (ignored).
        :type x: Optional[Tensor]
        """
        self.qfunc(self.initial_layer_weights, self.weights, self.wires, **self.kwargs)


class EmbedVarLayer(CircuitLayer):
    """
    Layer combining an embedding layer and a variation layer.

    :param embed: The embedding circuit layer.
    :type embed: CircuitLayer
    :param var: The variation circuit layer.
    :type var: CircuitLayer
    :param n_repeat: The number of times to repeat the combined layers, defaults to 1.
    :type n_repeat: int, optional
    :param omega: The exponent for the base of the power by which the inputs is scaled
        on each repetition number is raised, defaults to 0.0
    :type omega: float, optional
    :raises ValueError: If the wires of the embedding
        and variation layers are not the same.
    """

    def __init__(
        self,
        embed: CircuitLayer,
        var: CircuitLayer,
        n_repeat: int = 1,
        omega: float = 0.0,
    ) -> None:
        super().__init__(embed.wires)
        self.embed = embed
        self.var = var
        self.n_repeat = n_repeat
        self.omega = omega
        self.check_wires()

    def circuit(self, x: Tensor) -> None:
        """
        Define the quantum circuit for this layer.

        :param x: Input tensor that is passed to the quantum circuit.
        :type x: Tensor
        """
        for i in range(self.n_repeat):
            fac = 2 ** (self.omega * i)
            self.embed.circuit(fac * x)
            self.var.circuit(fac * x)

    def check_wires(self):
        """
        Check that the wires of the embedding and variation layers are the same.

        :raises ValueError: If the wires of the embedding and
            variation layers are not the same.
        """
        if self.embed.wires != self.var.wires:
            raise ValueError(
                f"Embed ({self.embed.wires}) and var ({self.var.wires}) "
                "blocks must have the same wires"
            )


class HamiltonianLayer(MeasurementLayer):
    """
    A layer that computes the expectation of a Hamiltonian.

    The Hamiltonian is defined by a list of observables and their associated weights.
    The weights are trainable parameters.

    :param circuits: Quantum circuits that make up the circuit layer before measurement.
    :type circuits: tuple
    :param observables: Observables defining the Hamiltonian.
    :type observables: list of Observable
    :param qdevice: Quantum device. If None specified, the default device is used.
    :type qdevice: Optional[QDevice]
    :param cdevice: Classical device to store the observable weights.
        If None specified, the default device is used.
    :param dtype: Data type of the observable weights.
        If None specified, the default data type is used.
    :param kwargs: Additional keyword arguments passed to the superclass.
    """

    def __init__(
        self,
        *circuits,
        observables: List[Observable],
        qdevice: Optional[QDevice] = None,
        cdevice=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(
            *circuits,
            qdevice=qdevice,
            measurement_type=MeasurementType.Expectation,
            observable=observables[0],
            **kwargs,
        )
        self.observables = observables
        self.cdevice = cdevice
        self.dtype = dtype

        # set observable weights
        self.num_weights = len(list(observables))
        self.observable_weights = torch.nn.Parameter(
            torch.empty(self.num_weights, device=self.cdevice, dtype=self.dtype)
        )
        nn.init.normal_(self.observable_weights)
        self.observable = qml.Hamiltonian(self.observable_weights, observables)

    def expectation(self, x: Optional[Tensor] = None) -> Expectation:
        """
        Compute the expectation of the Hamiltonian.

        :param x: Input tensor, defaults to None.
        :type x: Optional[Tensor]
        :return: Expectation value of the Hamiltonian.
        :rtype: Expectation
        """
        for circuit in self.circuits:
            circuit(x)
        self.observable = qml.Hamiltonian(self.observable_weights, self.observables)
        expec = qml.expval(self.observable)
        return expec
