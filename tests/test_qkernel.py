import pytest
import torch
import pennylane as qml

from qulearn.qlayer import HadamardLayer
from qulearn.qkernel import QKernel

DEFAULT_QDEV_CFG = {"name": "default.qubit", "wires": 2, "shots": None}


def test_qkernel_initialization():
    embed = HadamardLayer(wires=2)
    X_train = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    kernel = QKernel(embed, X_train)

    assert kernel.embed == embed
    assert torch.equal(kernel.X_train, X_train)
    assert isinstance(kernel.alpha, torch.nn.Parameter)
    assert isinstance(kernel.qdevice, type(qml.device(**DEFAULT_QDEV_CFG)))
    assert isinstance(kernel.qnode, qml.QNode)


def test_qkernel_input_shape_validation():
    embed = HadamardLayer(wires=2)
    X_train = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    kernel = QKernel(embed, X_train)

    x1 = torch.tensor([0.1, 0.2])
    x2 = torch.tensor([[0.1], [0.2]])

    with pytest.raises(ValueError):
        kernel.kernel_matrix(x1, x2)


def test_qkernel_forward():
    embed = HadamardLayer(wires=2)
    X_train = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    kernel = QKernel(embed, X_train)
    x = torch.tensor([[0.5, 0.6]])
    result = kernel(x)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1,)
