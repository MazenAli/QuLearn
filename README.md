# QuLearn

Welcome to QuLearn, a Python package designed to simplify the development and application of quantum and classical machine learning models. It includes a collection of QML applications from Fraunhofer ITWM.

## About

QuLearn is built on top of [PyTorch](https://pytorch.org/) and [PennyLane](https://pennylane.ai/), two well-established libraries in the realms of machine learning and quantum computing. Our goal is to streamline the process of setting up, training, and testing machine learning models, be they classical, quantum, or a mix of both.

QuLearn is suitable for various research applications and aims to democratize access to the exciting field of quantum machine learning. It serves as a platform for researchers, developers, and enthusiasts to implement, experiment, and contribute to this rapidly evolving field.

QuLearn also houses QML applications from Fraunhofer ITWM.

## Getting Started

### Installation

(Installation instructions will be added soon.)

### Basic Usage

#### Creating Models

The idea is to create classical, quantum or hybrid models with a simple syntax similar to PyTorch. In PyTorch, models are created by defining layers and feeding the input through the layers (`forward`). Similarly to classical models, we can build a quantum model using layers. Unlike classical models, not every quantum layer will return classical output. Thus, we distinguish between two types of quantum layers: circuit layers and measurement layers. The former can be applied to input, but returns no output. It only transforms the quantum state. The latter can be applied to a circuit layer (or empty circuuit = zero state) and produces classical output, much like a classical model.

Create a data embedding layer, a (trainable) variational layer and add a measurement layer on top.

```python
import pennylane as qml
from qulearn.layer import IQPEmbeddingLayer, RYCZLayer, MeasurementLayer, MeasurementType

# parameters
num_wires = 3
num_reup = 2
num_layers = 3
observable = qml.PauliZ(0)

# model
upload_layer = IQPEmbeddingLayer(num_wires, num_reup)
var_layer = RYCZLayer(num_wires, num_layers)
model = MeasurementLayer(upload_layer, var_layer, measurement_type=MeasurementType.Expectation, observable=observable)
```

`model` is a subclass of a PyTorch model and behaves the same. We can use it for predictions:

```python
y = model(X)
```

or, we can train this model

```python
from torch.optim import Adam
from qulearn.trainer import RegressionTrainer

opt = Adam(model.parameters(), lr=0.01, amsgrad=True)
loss_fn = torch.nn.MSELoss()
trainer = RegressionTrainer(opt, loss_fn, num_epochs=100)
trainer.train(model, loader_train, loader_valid)
```

We can add more layers to the model

```python
model = MeasurementLayer(upload_layer, var_layer, var_layer, upload_layer, var_layer, measurement_type=MeasurementType.Expectation, observable=observable)
```

We can add parametrized observables to the (quantum) model:

```python
from qulearn.qlayer import HamiltonianLayer

observables = [qml.PauliZ(0), qml.PauliX(1)]
model = HamiltonianLayer(upload_layer, var_layer, observables=observables)
```

We can create our own circuit layer and a hybrid quantum-classical model:

```python
from qulearn.qlayer import CircuitLayer

class MyQuantumCircuit(CircuitLayer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # set my attributes here


    def circuit(self, x) -> None:
        qml.Hadamard(wires=0)
        qml.RY(0.5, wires=1)

        # define your circuit...


class MyHybridModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # first a classical linear layer
        self.ll_in = nn.Linear(3, 3)

        # followed by a quantum layer
        circuit = MyQuantumCircuit()
        observables = [qml.PauliZ(0), qml.PauliX(1)]
        self.qnn = HamiltonianLayer(circuit, observables=observables)

        # concluding with another classical linear layer
        self.ll_out = nn.Linear(1, 1)

    def forward(self, x):

        y = self.ll_in(x)
        y = self.qnn(y)
        y = self.ll_out(y)

        return y
```

## Contributing

We greatly appreciate contributions to the QuLearn project! If you're a newcomer, please take a look at our [Contribution Guide](CONTRIBUTING.md), which provides a detailed guide to get you started. You can contribute in many ways, including but not limited to, reporting bugs, suggesting new features, improving documentation, or writing code patches.

Please remember to follow our [Code of Conduct](CODE_OF_CONDUCT.md), and ensure all your commits follow the Semantic Versioning format.

For feature additions or bug fixes, please create a new branch and submit a merge request. For main branch protection, at least one other developer is required to review your commit before merging.

## License

This project is licensed under the terms of the [MIT License](LICENSE).

## Contact

<a id="contact"></a>

- Name: Mazen Ali
- Email: mazen.ali@itwm.fraunhofer.de
