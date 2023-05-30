import torch
import pytest
import pennylane as qml
from qulearn.qlayer import (
    MeasurementType,
    CircuitLayer,
    MeasurementLayer,
    IQPEmbeddingLayer,
    RYCZLayer,
    EmbedVarLayer,
    HamiltonianLayer,
)


# Unit tests for CircuitLayer class


def test_circuit_layer_init():
    wires = 3
    layer = CircuitLayer(wires)
    assert layer.num_wires == wires
    assert layer.wires == list(range(wires))


def test_circuit_layer_forward():
    wires = 3
    layer = CircuitLayer(wires)
    x = torch.tensor([0.1, 0.2, 0.3])
    assert layer.forward(x) is None


def test_circuit_layer_set_wires_int():
    wires = 3
    layer = CircuitLayer(wires)
    layer.set_wires(5)
    assert layer.num_wires == 5
    assert layer.wires == list(range(5))


def test_circuit_layer_set_wires_iterable():
    wires = 3
    layer = CircuitLayer(wires)
    layer.set_wires([0, 1, 2, 3])
    assert layer.num_wires == 4
    assert layer.wires == [0, 1, 2, 3]


# Unit tests for MeasurementLayer class


@pytest.fixture(scope="function")
def mock_measurement_layer():
    wires = 2
    circuit1 = CircuitLayer(wires)
    circuit2 = CircuitLayer(wires)
    yield MeasurementLayer(circuit1, circuit2, observable=qml.PauliZ(0))


def test_measurement_layer_init(mock_measurement_layer):
    layer = mock_measurement_layer
    assert len(layer.circuits) == 2
    assert layer.qdevice is not None
    assert layer.measurement_type == MeasurementType.Probabilities
    assert layer.observable is not None
    assert layer.interface == "torch"


def test_measurement_layer_forward(mock_measurement_layer):
    layer = mock_measurement_layer
    x = torch.tensor([0.1, 0.2])
    assert layer.forward(x) is not None


def test_measurement_layer_expectation(mock_measurement_layer):
    layer = mock_measurement_layer
    x = torch.tensor([0.1, 0.2])
    assert layer.expectation(x) is not None


def test_measurement_layer_probabilities(mock_measurement_layer):
    layer = mock_measurement_layer
    x = torch.tensor([0.1, 0.2])
    assert layer.probabilities(x) is not None


def test_measurement_layer_samples(mock_measurement_layer):
    layer = mock_measurement_layer
    x = torch.tensor([0.1, 0.2])
    assert layer.samples(x) is not None


def test_measurement_layer_set_qnode(mock_measurement_layer):
    layer = mock_measurement_layer
    qnode = layer.set_qnode()
    assert qnode is not None


def test_measurement_layer_check_measurement_type(mock_measurement_layer):
    layer = mock_measurement_layer
    layer.measurement_type = "invalid"
    with pytest.raises(NotImplementedError):
        layer.check_measurement_type()


# Unit tests for IQPEmbeddingLayer class


def test_iqp_embedding_layer_init():
    wires = 3
    layer = IQPEmbeddingLayer(wires)
    assert layer.n_repeat == 1
    assert layer.wires == list(range(wires))


def test_iqp_embedding_layer_circuit():
    wires = 3
    layer = IQPEmbeddingLayer(wires)
    x = torch.tensor([0.1, 0.2, 0.3])
    assert layer.circuit(x) is None


# Unit tests for RYCZLayer class


def test_rycz_layer_init():
    wires = 3
    layer = RYCZLayer(wires)
    assert layer.n_layers == 1
    assert layer.wires == list(range(wires))
    assert layer.initial_layer_weights is not None
    assert layer.weights is not None


def test_rycz_layer_circuit():
    wires = 3
    layer = RYCZLayer(wires)
    x = torch.tensor([0.1, 0.2, 0.3])
    assert layer.circuit(x) is None


# Unit tests for EmbedVarLayer class


@pytest.fixture(scope="function")
def mock_embed_var_layer():
    embed = CircuitLayer(2)
    var = CircuitLayer(2)
    yield EmbedVarLayer(embed, var)


def test_embed_var_layer_init(mock_embed_var_layer):
    layer = mock_embed_var_layer
    assert layer.embed is not None
    assert layer.var is not None
    assert layer.n_repeat == 1
    assert layer.wires == list(range(2))


def test_embed_var_layer_circuit(mock_embed_var_layer):
    layer = mock_embed_var_layer
    x = torch.tensor([0.1, 0.2])
    assert layer.circuit(x) is None


def test_embed_var_layer_check_wires():
    embed = CircuitLayer(2)
    var = CircuitLayer(3)
    with pytest.raises(ValueError):
        EmbedVarLayer(embed, var)


# Unit tests for HamiltonianLayer class


def test_hamiltonian_layer_init():
    wires = 2
    observables = [qml.PauliX(0), qml.PauliZ(1)]
    layer = HamiltonianLayer(CircuitLayer(wires), observables=observables)
    assert len(layer.circuits) == 1
    assert layer.observable_weights is not None
    assert layer.observable is not None


def test_hamiltonian_layer_forward():
    wires = 2
    observables = [qml.PauliX(0), qml.PauliZ(1)]
    layer = HamiltonianLayer(CircuitLayer(wires), observables=observables)
    x = torch.tensor([0.1, 0.2])
    assert layer.forward(x) is not None


# Integration tests


def test_integration_circuit_layer_hamiltonian_layer():
    wires = 2
    observables = [qml.PauliX(0), qml.PauliZ(1)]
    circuit_layer = CircuitLayer(wires)
    hamiltonian_layer = HamiltonianLayer(circuit_layer, observables=observables)
    x = torch.tensor([0.1, 0.2])
    assert hamiltonian_layer.forward(x) is not None


def test_integration_iqp_embedding_layer_measurement_layer():
    wires = 2
    embed_layer = IQPEmbeddingLayer(wires)
    measurement_layer = MeasurementLayer(embed_layer)
    x = torch.tensor([0.1, 0.2])
    assert measurement_layer.forward(x) is not None


def test_trivial_output():
    wires = 3
    embed = IQPEmbeddingLayer(wires)
    var = RYCZLayer(wires)
    observables = [qml.Identity(0), qml.PauliZ(0), qml.PauliZ(1), qml.Identity(2)]
    ham_layer = HamiltonianLayer(embed, var, observables=observables)
    for name, param in ham_layer.named_parameters():
        if name == "observable_weights":
            torch.nn.init.ones_(param)
        else:
            torch.nn.init.zeros_(param)

    x = torch.zeros(wires)
    y = ham_layer(x)
    assert 2 == pytest.approx(y.item())
