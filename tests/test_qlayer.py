import torch
import pytest
import pennylane as qml
from qulearn.qlayer import (
    MeasurementType,
    CircuitLayer,
    MeasurementLayer,
    IQPEmbeddingLayer,
    HatBasisQFE,
    TwoQubitRotCXMPSLayer,
    embedU,
    RYCZLayer,
    AltRotCXLayer,
    IQPERYCZLayer,
    IQPEAltRotCXLayer,
    HamiltonianLayer,
    HadamardLayer,
    ParallelIQPEncoding,
    ParallelEntangledIQPEncoding,
)
from qulearn.hat_basis import HatBasis


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
    yield MeasurementLayer(circuit1, circuit2, observables=qml.PauliZ(0))


def test_measurement_layer_init(mock_measurement_layer):
    layer = mock_measurement_layer
    assert len(layer.circuits) == 2
    assert layer.qdevice is not None
    assert layer.measurement_type == MeasurementType.Probabilities
    assert layer.observables is not None
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


def test_measurement_layer_entropy(mock_measurement_layer):
    layer = mock_measurement_layer
    x = torch.tensor([0.1, 0.2])
    assert layer.entropy(x) is not None


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


# Unit tests for AltRotCXLayer class


def test_altrotcx_layer_init():
    wires = 3
    layer = AltRotCXLayer(wires)
    assert layer.n_layers == 1
    assert layer.wires == list(range(wires))
    assert layer.initial_layer_weights is not None
    assert layer.weights is not None


def test_altrotcx_layer_circuit():
    wires = 3
    layer = AltRotCXLayer(wires)
    x = torch.tensor([0.1, 0.2, 0.3])
    assert layer.circuit(x) is None


# Unit tests for EmbedVarLayer class


@pytest.fixture(scope="function")
def mock_embed_ryczvar_layer():
    yield IQPERYCZLayer(wires=2, num_repeat=3)


@pytest.fixture(scope="function")
def mock_embed_altrotvar_layer():
    yield IQPEAltRotCXLayer(wires=2, num_repeat=3)


def test_embed_ryczvar_layer_init(mock_embed_ryczvar_layer):
    layer = mock_embed_ryczvar_layer
    assert layer.num_repeat == 3
    assert layer.wires == list(range(2))


def test_embed_altrotvar_layer_init(mock_embed_altrotvar_layer):
    layer = mock_embed_altrotvar_layer
    assert layer.num_repeat == 3
    assert layer.wires == list(range(2))


def test_embed_ryczvar_layer_circuit(mock_embed_ryczvar_layer):
    layer = mock_embed_ryczvar_layer
    x = torch.tensor([0.1, 0.2])
    assert layer.circuit(x) is None


def test_embed_altrotvar_layer_circuit(mock_embed_altrotvar_layer):
    layer = mock_embed_altrotvar_layer
    x = torch.tensor([0.1, 0.2])
    assert layer.circuit(x) is None


def test_embed_ryczvar_layer_num_parameters(mock_embed_ryczvar_layer):
    layer = mock_embed_ryczvar_layer
    x = torch.tensor([0.1, 0.2])
    num_parameters = sum(p.numel() for p in layer.parameters())
    num_qubits = len(layer.wires)
    expected = layer.num_repeat * (num_qubits + 2 * (num_qubits - 1))
    assert expected == num_parameters


def test_embed_altrotvar_layer_num_parameters(mock_embed_altrotvar_layer):
    layer = mock_embed_altrotvar_layer
    x = torch.tensor([0.1, 0.2])
    num_parameters = sum(p.numel() for p in layer.parameters())
    num_qubits = len(layer.wires)
    expected = layer.num_repeat * 3 * (num_qubits + 2 * (num_qubits - 1))
    assert expected == num_parameters


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
    observables = [
        qml.Identity(0),
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.Identity(2),
    ]
    ham_layer = HamiltonianLayer(embed, var, observables=observables)
    for name, param in ham_layer.named_parameters():
        if name == "observable_weights":
            torch.nn.init.ones_(param)
        else:
            torch.nn.init.zeros_(param)

    x = torch.zeros(wires)
    y = ham_layer(x)
    assert 2 == pytest.approx(y.item())


def test_hadamard_layer():
    wires = qml.wires.Wires(range(3))
    layer = HadamardLayer(wires)

    # Testing that the qfunc is indeed Hadamard
    assert layer.qfunc == qml.Hadamard

    # Testing circuit application
    @qml.qnode(qml.device("default.qubit", wires=wires))
    def circuit():
        layer.circuit()
        return qml.probs(wires=wires)

    probs = circuit()
    assert 0.125 == pytest.approx(
        probs
    )  # Should be equal probabilities for all 8 states


def test_parallel_iqp_encoding():
    wires = qml.wires.Wires(range(4))
    num_features = 2

    # Testing ValueError for wrong number of features
    with pytest.raises(ValueError):
        layer = ParallelIQPEncoding(wires, 5)

    # Testing ValueError for wrong wires number
    with pytest.raises(ValueError):
        layer = ParallelIQPEncoding(qml.wires.Wires(range(3)), 2)

    layer = ParallelIQPEncoding(wires, num_features)
    assert layer.qfunc == qml.IQPEmbedding

    # Testing circuit application
    x = torch.tensor([1, 2], dtype=torch.float32)

    @qml.qnode(qml.device("default.qubit", wires=wires))
    def circuit():
        layer.circuit(x)
        return qml.probs(wires=wires)

    circuit()  # Shouldn't raise any error


def test_parallel_entangled_iqp_encoding():
    wires = qml.wires.Wires(range(4))
    num_features = 2

    # Testing ValueError for wrong number of features
    with pytest.raises(ValueError):
        layer = ParallelEntangledIQPEncoding(wires, 5)

    # Testing ValueError for wrong wires number
    with pytest.raises(ValueError):
        layer = ParallelEntangledIQPEncoding(qml.wires.Wires(range(3)), 2)

    layer = ParallelEntangledIQPEncoding(wires, num_features)
    assert layer.qfunc == qml.IQPEmbedding

    # Testing circuit application
    x = torch.tensor([1, 2], dtype=torch.float32)

    @qml.qnode(qml.device("default.qubit", wires=wires))
    def circuit():
        layer.circuit(x)
        return qml.probs(wires=wires)

    circuit()  # Shouldn't raise any error


def test_forward_1d_tensor_single_observable():
    circuit = IQPEmbeddingLayer(wires=1)
    model = MeasurementLayer(
        circuit,
        observables=qml.PauliZ(0),
        measurement_type=MeasurementType.Expectation,
    )
    x = torch.tensor([0.1])
    output = model(x)
    assert output.shape == torch.Size([1])


def test_forward_1d_tensor_multiple_observables():
    circuit = IQPEmbeddingLayer(wires=3)
    model = MeasurementLayer(
        circuit,
        observables=[qml.PauliZ(i) for i in range(3)],
        measurement_type=MeasurementType.Expectation,
    )
    x = torch.tensor([0.1, 0.2, 0.3])
    output = model(x)
    assert output.shape == torch.Size([3])


def test_forward_2d_tensor_multiple_observables():
    M, N = 5, 3
    x = torch.rand(M, N)
    circuit = IQPEmbeddingLayer(wires=N)
    model = MeasurementLayer(
        circuit,
        observables=[qml.PauliZ(i) for i in range(3)],
        measurement_type=MeasurementType.Expectation,
    )
    output = model(x)
    assert output.shape == torch.Size([M, 3])


def test_forward_none_input_multiple_observables():
    circuit = AltRotCXLayer(wires=3)
    model = MeasurementLayer(
        circuit,
        observables=[qml.PauliZ(i) for i in range(3)],
        measurement_type=MeasurementType.Expectation,
    )
    output = model()
    assert output.shape == torch.Size([3])


def test_forward_1d_tensor_hamiltonian():
    circuit = ParallelIQPEncoding(wires=2, num_features=1)
    observables = [qml.Identity(0), qml.PauliZ(0)]
    model = HamiltonianLayer(circuit, observables=observables)
    x = torch.randn((5, 1))
    output = model(x)
    assert output.shape == torch.Size([5, 1])


def test_forward_shape_3D_singleobs_hamiltonian():
    num_features = 3
    circuit = IQPEmbeddingLayer(wires=num_features)
    model = HamiltonianLayer(circuit, observables=[qml.PauliZ(0)])
    x = torch.randn(num_features)
    output = model(x)
    assert output.shape == torch.Size([1])


def test_forward_shape_1D_obs_hamiltonian():
    num_features = 1
    num_samples = 10
    circuit = IQPEmbeddingLayer(wires=num_features)
    model = HamiltonianLayer(circuit, observables=[qml.PauliZ(0)])
    x = torch.randn((num_samples, num_features))
    output = model(x)
    assert output.shape == torch.Size([num_samples, 1])


@pytest.fixture
def sample_hat_basis():
    return HatBasis(a=0, b=1, num_nodes=4)


def test_hat_basis_qfe_initialization(sample_hat_basis):
    hat_basis_qfe = HatBasisQFE(
        wires=2, basis=sample_hat_basis, sqrt=True, normalize=True
    )
    assert hat_basis_qfe.sqrt is True
    assert hat_basis_qfe.normalize is True


def test_hat_basis_qfe_circuit(sample_hat_basis):
    x = torch.tensor([0.0])
    hat_basis_qfe = HatBasisQFE(wires=2, basis=sample_hat_basis)
    hat_basis_qfe.circuit(x)

    assert hat_basis_qfe.norm == 1.0


def test_hat_basis_qfe_compute_norm(sample_hat_basis):
    x = torch.tensor([1.0])
    hat_basis_qfe = HatBasisQFE(wires=2, basis=sample_hat_basis)
    norm = hat_basis_qfe.compute_norm(x)

    assert isinstance(norm, float)
    assert 1.0 == pytest.approx(norm, abs=1e-4)


def test_TwoQubitRotCXMPSLayer_initialization():
    wires = 4
    layer = TwoQubitRotCXMPSLayer(wires=wires)

    assert layer.n_layers_mps == 1
    assert layer.n_layers_block == 1
    assert not layer.reverse
    assert layer.n_blocks == wires - 1
    assert layer.weights.shape == (1, 3, 2, 1, 3)
    assert layer.weights_post.shape == (wires, 3)


def test_embedU_initialization():
    wires = 2
    U = torch.eye(4)
    layer = embedU(wires=wires, U=U)

    assert torch.equal(layer.U, U)
