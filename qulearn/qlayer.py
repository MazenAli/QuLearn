# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

from typing import Iterable, Any, Optional, Union, Dict

from enum import Enum
import math
import torch
from torch import nn
import pennylane as qml

from .hat_basis import HatBasis
from .mps import HatBasisMPS, MPSQGates

DEFAULT_QDEV_CFG = {"name": "default.qubit", "shots": None}

QDevice: TypeAlias = qml.Device
CDevice: TypeAlias = torch.device
DType: TypeAlias = torch.dtype
QNode: TypeAlias = qml.QNode
Tensor: TypeAlias = torch.Tensor
Wires: TypeAlias = Union[int, Iterable[Any]]
Expectation: TypeAlias = qml.measurements.ExpectationMP
Observable: TypeAlias = qml.operation.Observable
Observables: TypeAlias = Union[
    qml.operation.Observable, Iterable[qml.operation.Observable]
]
Probability: TypeAlias = qml.measurements.ProbabilityMP
Sample: TypeAlias = qml.measurements.SampleMP
Entropy: TypeAlias = qml.measurements.VnEntropyMP


class MeasurementType(Enum):
    """Measurement type for a measurement layer."""

    """Expectation: return expected value of observable."""
    Expectation = "expectation"
    """Probabilities: return vector of probabilities."""
    Probabilities = "probabilities"
    """Samples: return measurement samples."""
    Samples = "samples"
    """Entropy: calculate the von Neumann entropy of a subsystem."""
    Entropy = "entropy"


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
        for _ in range(self.n_repeat):
            self.qfunc(x, wires=self.wires, **self.kwargs)


class HatBasisQFE(CircuitLayer):
    """
    Layer for the 1D hat basis quantum feature embedding.

    :param basis: The hat basis class.
    :type basis: HatBasis
    :param wires: The wires to be used by the layer
    :type wires: Wires
    :param sqrt: Set flag to take square roots before applying hat basis.
    :type sqrt: bool
    :param normalize: Set flag to normalize basis vector before embedding.
    :type normalize: bool
    """

    def __init__(
        self,
        wires: Wires,
        basis: HatBasis,
        sqrt: bool = False,
        normalize: bool = False,
    ) -> None:
        super().__init__(wires)
        self.basis = basis
        self.sqrt = sqrt
        self.normalize = normalize
        self.norm = 1.0
        self.hbmps = HatBasisMPS(basis)

    def circuit(self, x: Tensor) -> None:
        """
        Define the quantum circuit for this layer.

        :param x: Input tensor that is passed to the quantum circuit.
        :type x: Tensor
        """

        position = int(self.basis.position(x))
        a, b = self.basis.nonz_vals(x)

        if self.sqrt:
            # sometimes the values are close to 0 and negative
            a = torch.sqrt(torch.abs(a))
            b = torch.sqrt(torch.abs(b))

        if position == -1:
            self.norm = b.item()
            for q in range(self.num_wires):
                qml.Identity(wires=self.wires[q])
            return None

        elif position == -2:
            self.norm = a.item()
            for q in range(self.num_wires):
                qml.PauliX(wires=self.wires[q])
            return None

        self.norm = torch.sqrt(a**2 + b**2).item()
        if self.normalize:
            a /= self.norm
            b /= self.norm

        # for compatibility (TODO: remove)
        first = a.item()
        second = b.item()

        mps = self.hbmps.mps_hatbasis(first, second, position)
        mpsgates = MPSQGates(mps)

        s = mpsgates.max_rank_power
        Us = mpsgates.qgates()
        N = len(Us)
        count = 0
        for k in range(N - 1, -1, -1):
            wires_idx = list(
                range(self.num_wires - count - s - 1, self.num_wires - count)
            )
            subwires = [self.wires[idx] for idx in wires_idx]
            qml.QubitUnitary(Us[k], wires=subwires, unitary_check=False)

            count += 1

    def compute_norm(self, x: Tensor) -> float:
        """
        Compute the norm of the basis vector for the given input x.

        :param x: Input tensor that is passed to basis vector.
        :type x: Tensor
        :returns: The norm.
        :rtype: float
        """
        position = int(self.basis.position(x))
        a, b = self.basis.nonz_vals(x)

        if self.sqrt:
            a = torch.sqrt(a)
            b = torch.sqrt(b)

        if position == -1:
            self.norm = b.item()
            return self.norm

        elif position == -2:
            self.norm = a.item()
            return self.norm

        self.norm = torch.sqrt(a**2 + b**2).item()
        return self.norm


class Linear2DBasisQFE(CircuitLayer):
    """
    Layer for the 2D hat basis quantum feature embedding.

    :param basis: The hat basis class.
    :type basis: HatBasis
    :param wires: The wires to be used by the layer
    :type wires: Wires
    :param sqrt: Set flag to take square roots before applying hat basis.
    :type sqrt: bool
    :param normalize: Set flag to normalize basis vector before embedding.
    :type normalize: bool
    """

    def __init__(
            self,
            wires: Wires,
            basis: HatBasis,
            sqrt: bool = False,
            normalize: bool = False,
            zorder: bool = False,
    ) -> None:
        super().__init__(wires)
        self.basis = basis
        self.sqrt = sqrt
        self.normalize = normalize
        self.norm = 1.0
        self.hbmps = HatBasisMPS(basis)
        self.zorder = zorder
        self.mps = None
        self.mps1 = None
        self.mps2 = None

    def circuit(self, x: Tensor) -> None:
        """
        Define the quantum circuit for this layer.

        :param x: Input tensor that is passed to the quantum circuit.
        :type x: Tensor
        """
        self._check_input(x)

        x1 = x[0]
        x2 = x[1]
        position1 = int(self.basis.position(x1))
        position2 = int(self.basis.position(x2))
        a1, b1 = self.basis.nonz_vals(x1)
        a2, b2 = self.basis.nonz_vals(x2)

        if self.sqrt:
            # sometimes the values are close to 0 and negative
            a1 = torch.sqrt(torch.abs(a1))
            b1 = torch.sqrt(torch.abs(b1))
            a2 = torch.sqrt(torch.abs(a2))
            b2 = torch.sqrt(torch.abs(b2))

        # TODO: cover the case where x or y are outside of bounds

        val1 = a1 * a2
        val2 = a1 * b2
        val3 = a2 * b1
        val4 = a2 * b2
        self.norm = torch.sqrt(val1 ** 2 + val2 ** 2 + val3 ** 2 + val4 ** 2)

        if self.normalize:
            a1 /= torch.sqrt(self.norm)
            b1 /= torch.sqrt(self.norm)
            a2 /= torch.sqrt(self.norm)
            b2 /= torch.sqrt(self.norm)

        self.norm = self.norm.item()

        # for compatibility (TODO: remove)
        first1 = a1.item()
        second1 = b1.item()
        first2 = a2.item()
        second2 = b2.item()

        mps1 = self.hbmps.mps_hatbasis(first1, second1, position1)
        mps2 = self.hbmps.mps_hatbasis(first2, second2, position2)

        if self.zorder:
            mps = zkron2(mps2, mps1)
        else:
            mps = kron(mps2, mps1)

        self.mps1 = mps1
        self.mps2 = mps2
        self.mps = mps
        mpsgates = MPSQGates(mps)

        s = mpsgates.max_rank_power
        Us = mpsgates.qgates()
        N = len(Us)
        count = 0
        for k in range(N - 1, -1, -1):
            wires_idx = list(
                range(self.num_wires - count - s - 1, self.num_wires - count)
            )
            subwires = [self.wires[idx] for idx in wires_idx]
            qml.QubitUnitary(Us[k], wires=subwires, unitary_check=False)

            count += 1

    def compute_norm(self, x: Tensor) -> float:
        """
        Compute the norm of the basis vector for the given input x.

        :param x: Input tensor that is passed to basis vector.
        :type x: Tensor
        :returns: The norm.
        :rtype: float
        """
        self._check_input(x)

        x1 = x[0]
        x2 = x[1]
        a1, b1 = self.basis.nonz_vals(x1)
        a2, b2 = self.basis.nonz_vals(x2)

        if self.sqrt:
            # sometimes the values are close to 0 and negative
            a1 = torch.sqrt(torch.abs(a1))
            b1 = torch.sqrt(torch.abs(b1))
            a2 = torch.sqrt(torch.abs(a2))
            b2 = torch.sqrt(torch.abs(b2))

        # TODO: cover the case where x or y are outside of bounds

        val1 = a1 * a2
        val2 = a1 * b2
        val3 = a2 * b1
        val4 = a2 * b2
        self.norm = torch.sqrt(val1 ** 2 + val2 ** 2 + val3 ** 2 + val4 ** 2).item()

        return self.norm

    def _check_input(self, x: Tensor):
        if x.dim() > 2:
            raise ValueError("Input tensor must have 2 dimensions")

        if torch.any(torch.abs(x) >= 1):
            raise ValueError("Out of bounds case is not implemented")


class RYCZLayer(CircuitLayer):
    """
    Layer for the RYCZ (Rotation around Y and Controlled-Z) gates.

    :param wires: The wires to be used by the layer.
    :type wires: Wires
    :param n_layers: The number of layers for the simplified two-design architecture,
        defaults to 1.
    :type n_layers: int, optional
    :param cdevice: Classical device to store the initial layer weights
        and internal layer weights, defaults to None.
    :type cdevice: CDevice, optional
    :param dtype: Data type of the weights, defaults to None.
    :type dtype: DType, optional
    :param kwargs: Extra arguments passed to the SimplifiedTwoDesign.
    """

    def __init__(
        self,
        wires: Wires,
        n_layers: int = 1,
        cdevice: Optional[CDevice] = None,
        dtype: Optional[DType] = None,
        **kwargs,
    ) -> None:
        super().__init__(wires)

        self.n_layers = n_layers
        self.qfunc = qml.SimplifiedTwoDesign
        self.cdevice = cdevice
        self.dtype = dtype
        self.kwargs = kwargs

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


class AltRotCXLayer(CircuitLayer):
    """
    Layer for alternating CNOT gates with universal 1-qubit rotation.

    :param wires: The wires to be used by the layer.
    :type wires: Wires
    :param n_layers: The number of layers for the simplified two-design architecture,
        defaults to 1.
    :type n_layers: int, optional
    :param cdevice: Classical device to store the initial layer weights
        and internal layer weights, defaults to None.
    :type cdevice: CDevice, optional
    :param dtype: Data type of the weights, defaults to None.
    :type dtype: DType, optional
    :param kwargs: Extra arguments passed to the SimplifiedTwoDesign.
    """

    def __init__(
        self,
        wires: Wires,
        n_layers: int = 1,
        cdevice: Optional[CDevice] = None,
        dtype: Optional[DType] = None,
        **kwargs,
    ) -> None:
        super().__init__(wires)

        self.n_layers = n_layers
        self.cdevice = cdevice
        self.dtype = dtype
        self.kwargs = kwargs

        # weight parameters
        self.initial_layer_weights = torch.nn.Parameter(
            torch.empty((self.num_wires, 3), device=self.cdevice, dtype=self.dtype)
        )
        self.weights = torch.nn.Parameter(
            torch.empty(
                (self.n_layers, 2 * (self.num_wires - 1), 3),
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
        :type _: Optional[Tensor]
        """
        for index, q in enumerate(self.wires):
            qml.Rot(
                self.initial_layer_weights[index, 0],
                self.initial_layer_weights[index, 1],
                self.initial_layer_weights[index, 2],
                q,
            )

        for layer in range(self.n_layers):
            for i in range(0, len(self.wires) - 1, 2):
                qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
                qml.Rot(
                    self.weights[layer, i, 0],
                    self.weights[layer, i, 1],
                    self.weights[layer, i, 2],
                    self.wires[i],
                )
                qml.Rot(
                    self.weights[layer, i + 1, 0],
                    self.weights[layer, i + 1, 1],
                    self.weights[layer, i + 1, 2],
                    self.wires[i + 1],
                )

            offset = int(self.num_wires / 2) * 2
            for i in range(1, len(self.wires) - 1, 2):
                qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
                qml.Rot(
                    self.weights[layer, offset + i - 1, 0],
                    self.weights[layer, offset + i - 1, 1],
                    self.weights[layer, offset + i - 1, 2],
                    self.wires[i],
                )
                qml.Rot(
                    self.weights[layer, offset + i, 0],
                    self.weights[layer, offset + i, 1],
                    self.weights[layer, offset + i, 2],
                    self.wires[i + 1],
                )


class AltRXCXLayer(CircuitLayer):
    """
    Layer for alternating CNOT gates with RX rotations.

    :param wires: The wires to be used by the layer.
    :type wires: Wires
    :param n_layers: The number of layers for the simplified two-design architecture,
        defaults to 1.
    :type n_layers: int, optional
    :param cdevice: Classical device to store the initial layer weights
        and internal layer weights, defaults to None.
    :type cdevice: CDevice, optional
    :param dtype: Data type of the weights, defaults to None.
    :type dtype: DType, optional
    :param kwargs: Extra arguments passed to the SimplifiedTwoDesign.
    """

    def __init__(
        self,
        wires: Wires,
        n_layers: int = 1,
        cdevice: Optional[CDevice] = None,
        dtype: Optional[DType] = None,
        **kwargs,
    ) -> None:
        super().__init__(wires)

        self.n_layers = n_layers
        self.cdevice = cdevice
        self.dtype = dtype
        self.kwargs = kwargs

        # weight parameters
        self.initial_layer_weights = torch.nn.Parameter(
            torch.empty((self.num_wires), device=self.cdevice, dtype=self.dtype)
        )
        self.weights = torch.nn.Parameter(
            torch.empty(
                (self.n_layers, 2 * (self.num_wires - 1)),
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
        :type _: Optional[Tensor]
        """
        for index, q in enumerate(self.wires):
            qml.RX(
                self.initial_layer_weights[index],
                q,
            )

        for layer in range(self.n_layers):
            for i in range(0, len(self.wires) - 1, 2):
                qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
                qml.RX(
                    self.weights[layer, i],
                    self.wires[i],
                )
                qml.RX(
                    self.weights[layer, i + 1],
                    self.wires[i + 1],
                )

            offset = int(self.num_wires / 2) * 2
            for i in range(1, len(self.wires) - 1, 2):
                qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
                qml.RX(
                    self.weights[layer, offset + i - 1],
                    self.wires[i],
                )
                qml.RX(
                    self.weights[layer, offset + i],
                    self.wires[i + 1],
                )


class IQPERYCZLayer(CircuitLayer):
    """
    Layer combining an IQP embedding layer and an RY-CZ variational layer.

    :param wires: The wires to be used by the layer.
    :type wires: Wires
    :param num_uploads: The number of times to repeat data uploading, defaults to 1.
    :type num_uploads: int, optional
    :param num_varlayers: The number of times to repeat the variational layer, defaults to 1.
    :type num_varlayers: int, optional
    :param num_repeat: The number of times to repeat the combined layers, defaults to 1.
    :type num_repeat: int, optional
    :param base: The base of the exponent by which the inputs are scaled
        on each repetition, defaults to 1.0
    :type base: float, optional
    :param omega: The exponent for the base of the power by which the inputs are scaled
        on each repetition, defaults to 0.0
    :type omega: float, optional
    :param cdevice: Classical device to store the observable weights.
        If None specified, the default device is used.
    :type cdevice: CDevice, optional
    :param dtype: Data type of the variational weights.
    :type dtype: DType, optional
    :param iqpe_opts: Options for the IQPE class. Defaults to empty.
    :type iqpe_opts: Dict, optional
    :param rycz_opts: Options for the RYCZLayer class. Defaults to empty.
    :type rycz_opts: Dict, optional
    """

    def __init__(
        self,
        wires: Wires,
        num_uploads: int = 1,
        num_varlayers: int = 1,
        num_repeat: int = 1,
        base: Tensor = torch.tensor(1.0),
        omega: Tensor = torch.tensor(0.0),
        cdevice: Optional[CDevice] = None,
        dtype: Optional[DType] = None,
        iqpe_opts: Dict = {},
        rycz_opts: Dict = {},
    ) -> None:
        super().__init__(wires)
        self.num_uploads = num_uploads
        self.num_varlayers = num_varlayers
        self.num_repeat = num_repeat
        self.base = base
        self.omega = omega
        self.cdevice = cdevice
        self.dtype = dtype
        self.iqpe_opts = iqpe_opts
        self.rycz_opts = rycz_opts

        self.blocks = nn.ModuleList()

        num_var_repeats = 1
        if len(self.wires) == 1:
            num_var_repeats = self.num_varlayers

        for _ in range(self.num_repeat):
            embed_layer = IQPEmbeddingLayer(
                self.wires, self.num_uploads, **self.iqpe_opts
            )

            var_layers = []
            for _ in range(num_var_repeats):
                var_layer = RYCZLayer(
                    self.wires,
                    self.num_varlayers,
                    self.cdevice,
                    self.dtype,
                    **self.rycz_opts,
                )
                var_layers.append(var_layer)
            all_layers = [embed_layer] + var_layers
            block = nn.Sequential(*all_layers)
            self.blocks.append(block)

    def circuit(self, x: Tensor) -> None:
        """
        Define the quantum circuit for this layer.

        :param x: Input tensor that is passed to the quantum circuit.
        :type x: Tensor
        """
        for i, block in enumerate(self.blocks):
            fac = self.base ** (self.omega * i)
            block(fac * x)


class IQPEAltRotCXLayer(CircuitLayer):
    """
    Layer combining an IQP embedding layer and an alternating U3-CX variational layer.

    :param wires: The wires to be used by the layer.
    :type wires: Wires
    :param num_uploads: The number of times to repeat data uploading, defaults to 1.
    :type num_uploads: int, optional
    :param num_varlayers: The number of times to repeat the variational layer, defaults to 1.
    :type num_varlayers: int, optional
    :param num_repeat: The number of times to repeat the combined layers, defaults to 1.
    :type num_repeat: int, optional
    :param base: The base of the exponent by which the inputs are scaled
        on each repetition, defaults to 1.0
    :type base: float, optional
    :param omega: The exponent for the base of the power by which the inputs are scaled
        on each repetition, defaults to 0.0
    :type omega: float, optional
    :param cdevice: Classical device to store the observable weights.
        If None specified, the default device is used.
    :type cdevice: CDevice, optional
    :param dtype: Data type of the variational weights.
    :type dtype: DType, optional
    :param iqpe_opts: Options for the IQPE class. Defaults to empty.
    :type iqpe_opts: Dict, optional
    :param altrotcx_opts: Options for the AltRotCXLayer class. Defaults to empty.
    :type altrotcx_opts: Dict, optional
    """

    def __init__(
        self,
        wires: Wires,
        num_uploads: int = 1,
        num_varlayers: int = 1,
        num_repeat: int = 1,
        base: Tensor = torch.tensor(1.0),
        omega: Tensor = torch.tensor(0.0),
        cdevice: Optional[CDevice] = None,
        dtype: Optional[DType] = None,
        iqpe_opts: Dict = {},
        altrotcx_opts: Dict = {},
    ) -> None:
        super().__init__(wires)
        self.num_uploads = num_uploads
        self.num_varlayers = num_varlayers
        self.num_repeat = num_repeat
        self.base = base
        self.omega = omega
        self.cdevice = cdevice
        self.dtype = dtype
        self.iqpe_opts = iqpe_opts
        self.altrotcx_opts = altrotcx_opts

        self.blocks = nn.ModuleList()

        for _ in range(self.num_repeat):
            embed_layer = IQPEmbeddingLayer(
                self.wires, self.num_uploads, **self.iqpe_opts
            )
            var_layer = AltRotCXLayer(
                self.wires,
                self.num_varlayers,
                self.cdevice,
                self.dtype,
                **self.altrotcx_opts,
            )
            block = nn.Sequential(embed_layer, var_layer)
            self.blocks.append(block)

    def circuit(self, x: Tensor) -> None:
        """
        Define the quantum circuit for this layer.

        :param x: Input tensor that is passed to the quantum circuit.
        :type x: Tensor
        """
        for i, block in enumerate(self.blocks):
            fac = self.base ** (self.omega * i)
            block(fac * x)


class HadamardLayer(CircuitLayer):
    """
    A layer that adds Hadamard gates to each wire.

    :param wires: The wires to be used by the layer.
    :type wires: Wires
    """

    def __init__(self, wires: Wires):
        super().__init__(wires)
        self.qfunc = qml.Hadamard

    def circuit(self, _: Optional[Tensor] = None) -> None:
        """
        Apply the quantum circuit for this layer.

        :param _: Input tensor that is passed to the quantum circuit (ignored).
        :type _: Optional[Tensor]
        """
        for wire in self.wires:
            self.qfunc(wire)


class ParallelIQPEncoding(CircuitLayer):
    """
    A class that applies the IQPEmbedding to different parts of the input data.

    :param wires: The wires on which the circuit will be applied
    :type wires: Wires
    :param num_features: The number of features in the input
    :type num_features: int
    :param n_repeat: The number of times the IQPEmbedding will be repeated, defaults to 1
    :type n_repeat: int, optional
    :param base: The base of the exponent by which the inputs are scaled
        on each repetition, defaults to 1.0
    :type base: float, optional
    :param omega: The exponent for the base of the power by which the inputs are scaled
        on each repetition, defaults to 0.0
    :type omega: float, optional

    :raises ValueError: If the number of wires is less than the number of features
    :raises ValueError: If the number of wires is not a multiple of the number of features
    """

    def __init__(
        self,
        wires: Wires,
        num_features: int,
        n_repeat: int = 1,
        base: Tensor = torch.tensor(1.0),
        omega: Tensor = torch.tensor(0.0),
        **kwargs,
    ) -> None:
        super().__init__(wires)
        self.num_features = num_features
        self.n_repeat = n_repeat
        self.base = base
        self.omega = omega
        self.kwargs = kwargs
        self.qfunc = qml.IQPEmbedding

        if not self.num_wires >= self.num_features:
            raise ValueError(
                f"The number of wires ({self.num_wires}) must be greater than or equal to the number of features ({self.num_features})."
            )
        if not self.num_wires % self.num_features == 0:
            raise ValueError(
                f"The number of wires ({self.num_wires}) must be a multiple of the number of features ({self.num_features})."
            )

    def circuit(self, x: Tensor) -> None:
        """
        Apply the quantum circuit for this layer.

        :param x: Input tensor that is passed to the quantum circuit.
        :type x: Tensor
        """
        num_features = x.shape[-1]
        if num_features != self.num_features:
            raise ValueError(
                f"Input tensor last dimension ({num_features}) must be equal to the number of features ({self.num_features})."
            )

        freq = 0
        for i in range(0, len(self.wires), num_features):
            x_ = self.base ** (freq * self.omega) * x
            self.qfunc(
                x_,
                self.wires[i : i + num_features],
                self.n_repeat,
                **self.kwargs,
            )
            freq += 1


class ParallelEntangledIQPEncoding(CircuitLayer):
    """
    A class that applies the IQPEmbedding on the entire constructed large feature vector.

    :param wires: The wires on which the circuit will be applied
    :type wires: Wires
    :param num_features: The number of features in the input
    :type num_features: int
    :param n_repeat: The number of times the IQPEmbedding will be repeated, defaults to 1
    :type n_repeat: int, optional
    :param base: The base of the exponent by which the inputs are scaled
        on each repetition, defaults to 1.0
    :type base: float, optional
    :param omega: The exponent for the base of the power by which the inputs are scaled
        on each repetition, defaults to 0.0
    :type omega: float, optional

    :raises ValueError: If the number of wires is less than the number of features
    :raises ValueError: If the number of wires is not a multiple of the number of features
    """

    def __init__(
        self,
        wires: Wires,
        num_features: int,
        n_repeat: int = 1,
        base: Tensor = torch.tensor(1.0),
        omega: Tensor = torch.tensor(0.0),
        **kwargs,
    ) -> None:
        super().__init__(wires)
        self.num_features = num_features
        self.n_repeat = n_repeat
        self.base = base
        self.omega = omega
        self.kwargs = kwargs
        self.qfunc = qml.IQPEmbedding

        if not self.num_wires >= self.num_features:
            raise ValueError(
                f"The number of wires ({self.num_wires}) must be greater than or equal to the number of features ({self.num_features})."
            )
        if not self.num_wires % self.num_features == 0:
            raise ValueError(
                f"The number of wires ({self.num_wires}) must be a multiple of the number of features ({self.num_features})."
            )

    def circuit(self, x: Tensor) -> None:
        """
        Apply the quantum circuit for this layer.

        :param x: Input tensor that is passed to the quantum circuit.
        :type x: Tensor
        """
        num_features = x.shape[-1]
        if num_features != self.num_features:
            raise ValueError(
                f"Input tensor last dimension ({num_features}) must be equal to the number of features ({self.num_features})."
            )

        num_repeats = int(self.num_wires / num_features)

        x_large = []
        for j in range(0, num_repeats):
            x_ = self.base ** (j * self.omega) * x
            x_large.append(x_)

        x_final = torch.cat(x_large)
        self.qfunc(x_final, self.wires, self.n_repeat, **self.kwargs)


class TwoQubitRotCXMPSLayer(CircuitLayer):
    """
    Layer with 2-qubit MPS sctructure.

    :param wires: The wires to be used by the layer.
    :type wires: Wires
    :param n_layers_mps: The number of layers to repeat the MPS,
        defaults to 1.
    :type n_layers: int, optional
    :param n_layers_block: The number of layers for each block of the MPS,
        defaults to 1.
    :type n_layers_block: int, optional
    :param reverse: Flag to reverse the MPS sequence to bottom to top,
        defaults to False.
    :type reverse: bool, optional
    :param cdevice: Classical device to store the initial layer weights
        and internal layer weights, defaults to None.
    :type cdevice: CDevice, optional
    :param dtype: Data type of the weights, defaults to None.
    :type dtype: DType, optional
    """

    def __init__(
        self,
        wires: Wires,
        n_layers_mps: int = 1,
        n_layers_block: int = 1,
        reverse: bool = False,
        cdevice: Optional[CDevice] = None,
        dtype: Optional[DType] = None,
    ) -> None:
        super().__init__(wires)

        self.n_layers_mps = n_layers_mps
        self.n_layers_block = n_layers_block
        self.reverse = reverse
        self.cdevice = cdevice
        self.dtype = dtype
        self.n_blocks = self.num_wires - 1

        self.weights = torch.nn.Parameter(
            torch.empty(
                (self.n_layers_mps, self.n_blocks, 2, self.n_layers_block, 3),
                device=self.cdevice,
                dtype=self.dtype,
            )
        )
        self.weights_post = torch.nn.Parameter(
            torch.empty((self.num_wires, 3), device=self.cdevice, dtype=self.dtype)
        )

        nn.init.uniform_(self.weights, a=0.0, b=2 * math.pi)
        nn.init.uniform_(self.weights_post, a=0.0, b=2 * math.pi)

    def circuit(self, _: Optional[Tensor] = None) -> None:
        """
        Define the quantum circuit for this layer.

        :param _: Input tensor that is passed to the quantum circuit (ignored).
        :type x: Optional[Tensor]
        """

        for mps_layer_idx in range(self.n_layers_mps):
            for block_idx in (
                range(self.n_blocks - 1, -1, -1)
                if self.reverse
                else range(self.n_blocks)
            ):
                self._block(mps_layer_idx, block_idx)

        for i, q in enumerate(self.wires):
            qml.Rot(
                self.weights_post[i, 0],
                self.weights_post[i, 1],
                self.weights_post[i, 2],
                q,
            )

    def _block(self, mps_layer_idx, block_idx):
        qprev = self.wires[block_idx]
        qnext = self.wires[block_idx + 1]

        for block_layer in range(self.n_layers_block):
            qml.Rot(
                self.weights[mps_layer_idx, block_idx, 0, block_layer, 0],
                self.weights[mps_layer_idx, block_idx, 0, block_layer, 1],
                self.weights[mps_layer_idx, block_idx, 0, block_layer, 2],
                qprev,
            )
            qml.Rot(
                self.weights[mps_layer_idx, block_idx, 1, block_layer, 0],
                self.weights[mps_layer_idx, block_idx, 1, block_layer, 1],
                self.weights[mps_layer_idx, block_idx, 1, block_layer, 2],
                qnext,
            )
            qml.CNOT(wires=(qprev, qnext))


class EmbedU(CircuitLayer):
    """
    Layer that embeds an arbitrary unitary.

    :param wires: The wires to be used by the layer.
    :type wires: Wires
    :param U: The unitary to embed.
    :type U: Tensor.
    """

    def __init__(self, wires: Wires, U: Tensor) -> None:
        super().__init__(wires)
        self.U = U

    def circuit(self, _: Optional[Tensor] = None) -> None:
        """
        Define the quantum circuit for this layer.

        :param _: Input tensor that is passed to the quantum circuit (ignored).
        :type x: Optional[Tensor]
        """
        qml.QubitUnitary(self.U, wires=self.wires, unitary_check=False)


class MeasurementLayer(nn.Module):
    """
    Base class for measurment layers.

    Measurement layers are appended to quantum circuits and return classical output.

    :param circuits: Quantum circuits before the measurement.
    :type circuits: tuple
    :param qdevice: Quantum device. If None, the default device will be used.
    :type qdevice: Optional[QDevice], defaults to None
    :param measurement_type: Type of quantum measurement.
    :type measurement_type: MeasurementType, defaults to MeasurementType.Probabilities
    :param observables: Observables to measure.
        None only works with Probabilities and Samples.
    :type observables: Optional[Observables], defaults to None
    :param kwargs: Additional keyword arguments for qml.QNode.
    """

    def __init__(
        self,
        *circuits,
        qdevice: Optional[QDevice] = None,
        measurement_type: MeasurementType = MeasurementType.Probabilities,
        observables: Optional[Observables] = None,
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

        self.observables = observables
        if observables is not None and not isinstance(observables, Iterable):
            self.observables = [observables]

        self.interface = kwargs.pop("interface", "torch")
        self.diff_method = kwargs.pop("diff_method", "backprop")
        self.kwargs = kwargs
        self.check_measurement_type()
        self.qnode = self.set_qnode()

        self.subwires = [0]

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass, depending on measurement type.
        See :meth:`expectation`, :meth:`probabilities` and :meth:`samples`.

        :param x: Input data samples.
        :type circuits: Tensor.
        :return: Forward evaluation of model on data.
        :rtype: Tensor
        """
        self.qnode = self.set_qnode()

        if x is not None:
            if len(x.shape) == 1:
                out = self.qnode(x)
            else:
                outs = [self.qnode(xk) for xk in torch.unbind(x)]
                out = torch.stack(outs)

            if (len(x.shape) == 1 and len(out.shape) == 0) or (
                len(x.shape) > 1 and len(out.shape) == 1
            ):
                out = out.unsqueeze(-1)

        else:
            out = self.qnode(x)

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

        expec = None
        if self.observables is not None:
            expec = [qml.expval(obs) for obs in self.observables]

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

    def entropy(self, x: Optional[Tensor] = None) -> Entropy:
        """
        Calculate the Von Neumann Entropy of a subsystem.

        :param x: Input tensor that is passed to the quantum circuits, defaults to None.
        :type x: Optional[Tensor]
        :return: Entropy value object for a subsystem.
        :rtype: Entropy
        """
        for circuit in self.circuits:
            circuit(x)
        entropy = qml.vn_entropy(self.subwires)
        return entropy

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
        elif self.measurement_type == MeasurementType.Entropy:
            circuit = self.entropy

        qnode = qml.QNode(
            circuit,
            self.qdevice,
            interface=self.interface,
            diff_method=self.diff_method,
            **self.kwargs,
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
            if self.observables is None:
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
    :type cdevice: CDevice, optional
    :param dtype: Data type of the observable weights.
    :type dtype: DType, optional
    :param kwargs: Additional keyword arguments passed to the superclass.
    """

    def __init__(
        self,
        *circuits,
        observables: Iterable[Observable],
        qdevice: Optional[QDevice] = None,
        cdevice=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(
            *circuits,
            qdevice=qdevice,
            measurement_type=MeasurementType.Expectation,
            observables=observables,
            **kwargs,
        )
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
