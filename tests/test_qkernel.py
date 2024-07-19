import pennylane as qml
import pytest
import torch

from qulearn.qkernel import QKernel
from qulearn.qlayer import HadamardLayer

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


def test_QKernel_X_train_getter_and_setter():
    embed = HadamardLayer(wires=2)  # You would need to define or mock this
    X_train_initial = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    qkernel = QKernel(embed, X_train_initial)

    # Check the getter
    assert torch.equal(qkernel.X_train, X_train_initial)

    # Check the setter
    new_X_train = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    qkernel.X_train = new_X_train
    assert torch.equal(qkernel.X_train, new_X_train)

    # Verify that other attributes like num_samples and alpha were updated
    assert qkernel.num_samples == 2
    assert qkernel.alpha.shape[0] == 2
