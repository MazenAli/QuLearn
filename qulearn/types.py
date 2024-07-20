from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import logging

import numpy as np
import pennylane as qml
import tntorch
import torch
from torch.utils.tensorboard import SummaryWriter

# Type aliases
Tensor: TypeAlias = torch.Tensor
Array: TypeAlias = np.ndarray
Device: TypeAlias = torch.device
DataOut: TypeAlias = Dict[str, Tensor]
DataLoader: TypeAlias = torch.utils.data.DataLoader
TensorDataset: TypeAlias = torch.utils.data.TensorDataset
Model: TypeAlias = torch.nn.Module
QModel: TypeAlias = qml.QNode
ParameterList: TypeAlias = List[List[Tensor]]
CDevice: TypeAlias = torch.device
DType: TypeAlias = torch.dtype
Loss: TypeAlias = torch.nn.Module
Capacity = List[Tuple[int, float, int, int]]
MPS: TypeAlias = tntorch.tensor.Tensor
Observable: TypeAlias = qml.operation.Observable
Observables: TypeAlias = Union[qml.operation.Observable, Iterable[qml.operation.Observable]]
Probability: TypeAlias = qml.measurements.ProbabilityMP
Sample: TypeAlias = qml.measurements.SampleMP
Entropy: TypeAlias = qml.measurements.VnEntropyMP
Hamiltonian: TypeAlias = qml.Hamiltonian
ParitySequence: TypeAlias = Sequence[Tuple[int, ...]]
QDevice: TypeAlias = qml.Device
QNode: TypeAlias = qml.QNode
Expectation: TypeAlias = qml.measurements.ExpectationMP
Wires: TypeAlias = Union[int, Iterable[Any]]
Optimizer: TypeAlias = torch.optim.Optimizer
Metric: TypeAlias = Callable
Writer: TypeAlias = SummaryWriter
Logger: TypeAlias = logging.Logger
Parameter: TypeAlias = torch.nn.Parameter
