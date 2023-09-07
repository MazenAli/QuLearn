# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

from typing import Optional

import pennylane as qml
import torch
from torch import nn

from .qlayer import CircuitLayer

DEFAULT_QDEV_CFG = {"name": "default.qubit", "shots": None}

FeatureEmbed: TypeAlias = CircuitLayer
Tensor: TypeAlias = torch.Tensor
QDevice: TypeAlias = qml.Device
QNode: TypeAlias = qml.QNode
Expectation: TypeAlias = qml.measurements.ExpectationMP


class QKernel(nn.Module):
    """
    A quantum kernel module that represents a quantum embedding kernel.

    :param embed: The quantum feature embedding mechanism.
    :type embed: FeatureEmbed
    :param X_train: Training data tensor.
    :type X_train: Tensor
    :param qdevice: Quantum device. If None, the default device will be used.
    :type qdevice: Optional[QDevice], defaults to None
    :param kwargs: Additional keyword arguments for qml.QNode.
    """

    def __init__(
        self,
        embed: FeatureEmbed,
        X_train: Tensor,
        qdevice: Optional[QDevice] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embed = embed
        self.X_train = X_train
        self.num_samples = X_train.shape[0]
        self.alpha = nn.Parameter(
            torch.empty(
                self.num_samples,
                device=self.X_train.device,
                dtype=self.X_train.dtype,
            )
        )
        nn.init.normal_(self.alpha)

        if qdevice is None:
            self.qdevice = qml.device(wires=self.embed.wires, **DEFAULT_QDEV_CFG)
        else:
            self.qdevice = qdevice

        self.interface = kwargs.pop("interface", "torch")
        self.diff_method = kwargs.pop("diff_method", "backprop")
        self.qnode = self.set_qnode()

    def kernel_circuit(self, x: Tensor, x_: Tensor) -> Expectation:
        """
        Define the quantum circuit that calculates the kernel between input vectors.

        :param x: Input tensor representing the first data point.
        :type x: Tensor
        :param x_: Input tensor representing the second data point.
        :type x_: Tensor

        :return: Expectation value of the projected state.
        :rtype: Expectation
        """

        self.embed(x)
        qml.adjoint(self.embed)(x_)
        state = torch.zeros(self.embed.num_wires, dtype=torch.int32, device=x.device)
        projector = qml.Projector(state, self.embed.wires)

        return qml.expval(projector)

    def kernel_matrix(self, x: Tensor, x_: Tensor) -> Tensor:
        """
        Compute the kernel matrix between two sets of input vectors.

        :param x: Input tensor representing the first set of data points.
        :type x: Tensor
        :param x_: Input tensor representing the second set of data points.
        :type x_: Tensor

        :return: Kernel matrix of shape (number of data points in x, number of data points in x_).
        :rtype: Tensor

        :raises ValueError: If the input tensors do not have exactly 2 dimensions.
        """

        lx = len(x.shape)
        lx_ = len(x_.shape)
        if lx != 2 or lx_ != 2:
            raise ValueError(f"Inputs must have 2 dimensions each, not {lx} and {lx_}")

        K = torch.empty((x.shape[0], x_.shape[0]), dtype=x.dtype, device=x.device)
        for i, xi in enumerate(x):
            for j, xj in enumerate(x_):
                Kij = self.qnode(xi, xj)
                K[i, j] = Kij

        return K

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward computation of the quantum kernel using the training data and alpha parameters.

        :param x: Input tensor representing the data points.
        :type x: Tensor

        :return: Resultant tensor after the forward pass.
        :rtype: Tensor
        """

        K = self.kernel_matrix(x, self.X_train)
        out = torch.matmul(K, self.alpha)
        return out

    def set_qnode(self) -> QNode:
        """
        Define and set the quantum node for the kernel circuit.

        :return: The defined quantum node.
        :rtype: QNode
        """

        circuit = self.kernel_circuit
        qnode = qml.QNode(
            circuit,
            self.qdevice,
            interface=self.interface,
            diff_method=self.diff_method,
        )
        self.qnode = qnode
        return self.qnode
