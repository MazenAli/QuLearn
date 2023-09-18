"""
This module uses the model configs file to build a model for the given config ID.
Modify the global variables here for different meta settings, e.g., different device types.
Note that lightning qubit with adjoint differentation is faster but does not work for trainable Hamiltonians!
It will not throw an error and simply will not optimize the Hamiltonian parameters!
"""

# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import os

os.environ["OMP_NUM_THREADS"] = "8"

from typing import List, Optional, Dict, Callable
import json
from enum import Enum
from itertools import combinations
import torch
from torch import nn
import pennylane as qml
from qulearn.qlayer import (
    CircuitLayer,
    HadamardLayer,
    IQPEmbeddingLayer,
    ParallelIQPEncoding,
    ParallelEntangledIQPEncoding,
    AltRotCXLayer,
    IQPEAltRotCXLayer,
    HamiltonianLayer,
    MeasurementLayer,
    MeasurementType,
)
from qulearn.qkernel import QKernel
from qulearn.observable import sequence2parity_observable, parities_all_observables

Observable: TypeAlias = qml.operation.Observable
Tensor: TypeAlias = torch.Tensor

FILE_PATH = "model_configs.json"
QDEV = {"name": "default.qubit", "shots": None}
CDEV = torch.device("cpu")
DTYPE = torch.float64
INTERFACE = "torch"
DIFFM = "backprop"
MEASTYPE = MeasurementType.Expectation


class HamiltonianType(Enum):
    ParityZ0 = 1
    ParityAllWires = 2
    ParityAllWirePairs = 3
    ParitiesAllWireCombinations = 4


class UploadType(Enum):
    Sequential = 1
    Parallel = 2
    ParallelEntangled = 3


def observableZ0(_: Optional[int] = None) -> List[Observable]:
    return [qml.PauliZ(0)]


def observableAllWires(num_wires: int) -> List[Observable]:
    sequence = [tuple(range(num_wires))]
    return sequence2parity_observable(sequence)


def observableAllWirePairs(num_wires: int) -> List[Observable]:
    sequence = list(combinations(range(num_wires), 2))
    return sequence2parity_observable(sequence)


def observableAllWireCombinations(num_wires: int) -> List[Observable]:
    return parities_all_observables(num_wires)


observable_opts: Dict[HamiltonianType, Callable[[int], List[Observable]]] = {
    HamiltonianType.ParityZ0: observableZ0,
    HamiltonianType.ParityAllWires: observableAllWires,
    HamiltonianType.ParityAllWirePairs: observableAllWirePairs,
    HamiltonianType.ParitiesAllWireCombinations: observableAllWireCombinations,
}


type_conversion_hamiltonian = {
    "Z0": HamiltonianType.ParityZ0,
    "AllWires": HamiltonianType.ParityAllWires,
    "AllWirePairs": HamiltonianType.ParityAllWirePairs,
    "AllWireCombinations": HamiltonianType.ParitiesAllWireCombinations,
}

type_conversion_upload = {
    "Sequential": UploadType.Sequential,
    "Parallel": UploadType.Parallel,
    "ParallelEntangled": UploadType.ParallelEntangled,
}


def observable_list(num_wires: int, type: HamiltonianType) -> List[Observable]:
    return observable_opts[type](num_wires)


class QNNModel(nn.Module):
    def __init__(self, **config) -> None:
        super().__init__()
        self.init_varlayer = config["init_varlayer"]
        self.init_varlayer_nlayers = config["init_varlayer_nlayers"]
        self.upload_type = type_conversion_upload[config["upload_type"]]
        self.num_features = config["num_features"]
        self.num_repeat_parallel = config["num_repeat_parallel"]
        self.num_reup = config["num_reuploads"]
        self.num_layers = config["num_varlayers"]
        self.num_repeats = config["num_repeats"]
        self.omega = torch.tensor(config["omega"], dtype=DTYPE, device=CDEV)
        self.hamiltonian_type = config["hamiltonian_type"]
        self.id = config["id"]

        self.num_wires = self.num_features
        if self.upload_type in (UploadType.Parallel, UploadType.ParallelEntangled):
            self.num_wires = self.num_features * self.num_repeat_parallel

        hadamards = HadamardLayer(self.num_wires)
        init_var = AltRotCXLayer(
            self.num_wires, self.init_varlayer_nlayers, CDEV, DTYPE
        )

        embedvar = CircuitLayer(self.num_wires)
        if self.upload_type == UploadType.Sequential:
            embedvar = IQPEAltRotCXLayer(
                self.num_wires,
                self.num_reup,
                self.num_layers,
                self.num_repeats,
                self.omega,
                CDEV,
                DTYPE,
            )
        else:
            var = AltRotCXLayer(self.num_wires, self.num_layers, CDEV, DTYPE)
            embed = CircuitLayer(self.num_wires)
            if self.upload_type == UploadType.Parallel:
                embed = ParallelIQPEncoding(
                    self.num_wires, self.num_features, self.num_reup, self.omega
                )
            elif self.upload_type == UploadType.ParallelEntangled:
                embed = ParallelEntangledIQPEncoding(
                    self.num_wires, self.num_features, self.num_reup, self.omega
                )

        obs_type = type_conversion_hamiltonian[self.hamiltonian_type]
        observables = observable_list(self.num_wires, obs_type)
        qdevice = qml.device(**QDEV, wires=self.num_wires)

        if self.upload_type == UploadType.Sequential:
            if self.init_varlayer:
                self.qnn = HamiltonianLayer(
                    hadamards,
                    init_var,
                    embedvar,
                    observables=observables,
                    qdevice=qdevice,
                    cdevice=CDEV,
                    dtype=DTYPE,
                    interface=INTERFACE,
                    diff_method=DIFFM,
                )
            else:
                self.qnn = HamiltonianLayer(
                    embedvar,
                    observables=observables,
                    qdevice=qdevice,
                    cdevice=CDEV,
                    dtype=DTYPE,
                    interface=INTERFACE,
                    diff_method=DIFFM,
                )
        else:
            if self.init_varlayer:
                self.qnn = HamiltonianLayer(
                    hadamards,
                    init_var,
                    embed,
                    var,
                    observables=observables,
                    qdevice=qdevice,
                    cdevice=CDEV,
                    dtype=DTYPE,
                    interface=INTERFACE,
                    diff_method=DIFFM,
                )
            else:
                self.qnn = HamiltonianLayer(
                    embed,
                    var,
                    observables=observables,
                    qdevice=qdevice,
                    cdevice=CDEV,
                    dtype=DTYPE,
                    interface=INTERFACE,
                    diff_method=DIFFM,
                )

    def forward(self, x: Tensor) -> Tensor:
        y = self.qnn(x)
        return y


class QKernelModel(nn.Module):
    def __init__(self, **config) -> None:
        super().__init__()
        self.upload_type = type_conversion_upload[config["upload_type"]]
        self.num_features = config["num_features"]
        self.num_repeat_parallel = config["num_repeat_parallel"]
        self.num_reup = config["num_reuploads"]
        self.omega = torch.tensor(config["omega"], dtype=DTYPE, device=CDEV)
        self.id = config["id"]

        self.num_wires = self.num_features
        if self.upload_type in (UploadType.Parallel, UploadType.ParallelEntangled):
            self.num_wires = self.num_features * self.num_repeat_parallel

        embed = CircuitLayer(self.num_wires)
        if self.upload_type == UploadType.Sequential:
            embed = IQPEmbeddingLayer(
                self.num_wires,
                self.num_reup,
            )
        else:
            embed = CircuitLayer(self.num_wires)
            if self.upload_type == UploadType.Parallel:
                embed = ParallelIQPEncoding(
                    self.num_wires, self.num_features, self.num_reup, self.omega
                )
            elif self.upload_type == UploadType.ParallelEntangled:
                embed = ParallelEntangledIQPEncoding(
                    self.num_wires, self.num_features, self.num_reup, self.omega
                )

        qdevice = qml.device(**QDEV, wires=self.num_wires)
        num_samples = 2
        X_train = torch.randn(
            (num_samples, self.num_features), device=CDEV, dtype=DTYPE
        )
        self.qnn = QKernel(
            embed,
            X_train,
            qdevice,
            interface=INTERFACE,
            diff_method=DIFFM,
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.qnn(x)
        return y


class QNNStatModel(nn.Module):
    def __init__(self, **config) -> None:
        super().__init__()
        self.init_varlayer = config["init_varlayer"]
        self.init_varlayer_nlayers = config["init_varlayer_nlayers"]
        self.upload_type = type_conversion_upload[config["upload_type"]]
        self.num_features = config["num_features"]
        self.num_repeat_parallel = config["num_repeat_parallel"]
        self.num_reup = config["num_reuploads"]
        self.num_layers = config["num_varlayers"]
        self.num_repeats = config["num_repeats"]
        self.omega = torch.tensor(config["omega"], dtype=DTYPE, device=CDEV)
        self.hamiltonian_type = config["hamiltonian_type"]
        self.id = config["id"]

        self.num_wires = self.num_features
        if self.upload_type in (UploadType.Parallel, UploadType.ParallelEntangled):
            self.num_wires = self.num_features * self.num_repeat_parallel

        hadamards = HadamardLayer(self.num_wires)
        init_var = AltRotCXLayer(
            self.num_wires, self.init_varlayer_nlayers, CDEV, DTYPE
        )

        embedvar = CircuitLayer(self.num_wires)
        if self.upload_type == UploadType.Sequential:
            embedvar = IQPEAltRotCXLayer(
                self.num_wires,
                self.num_reup,
                self.num_layers,
                self.num_repeats,
                self.omega,
                CDEV,
                DTYPE,
            )
        else:
            var = AltRotCXLayer(self.num_wires, self.num_layers, CDEV, DTYPE)
            embed = CircuitLayer(self.num_wires)
            if self.upload_type == UploadType.Parallel:
                embed = ParallelIQPEncoding(
                    self.num_wires, self.num_features, self.num_reup, self.omega
                )
            elif self.upload_type == UploadType.ParallelEntangled:
                embed = ParallelEntangledIQPEncoding(
                    self.num_wires, self.num_features, self.num_reup, self.omega
                )

        qdevice = qml.device(**QDEV, wires=self.num_wires)

        if self.upload_type == UploadType.Sequential:
            if self.init_varlayer:
                self.qnn = MeasurementLayer(
                    hadamards,
                    init_var,
                    embedvar,
                    measurement_type=MeasurementType.Probabilities,
                    qdevice=qdevice,
                    cdevice=CDEV,
                    dtype=DTYPE,
                    interface=INTERFACE,
                    diff_method=DIFFM,
                )
            else:
                self.qnn = MeasurementLayer(
                    embedvar,
                    measurement_type=MeasurementType.Probabilities,
                    qdevice=qdevice,
                    cdevice=CDEV,
                    dtype=DTYPE,
                    interface=INTERFACE,
                    diff_method=DIFFM,
                )
        else:
            if self.init_varlayer:
                self.qnn = MeasurementLayer(
                    hadamards,
                    init_var,
                    embed,
                    var,
                    measurement_type=MeasurementType.Probabilities,
                    qdevice=qdevice,
                    cdevice=CDEV,
                    dtype=DTYPE,
                    interface=INTERFACE,
                    diff_method=DIFFM,
                )
            else:
                self.qnn = MeasurementLayer(
                    embed,
                    var,
                    measurement_type=MeasurementType.Probabilities,
                    qdevice=qdevice,
                    cdevice=CDEV,
                    dtype=DTYPE,
                    interface=INTERFACE,
                    diff_method=DIFFM,
                )

    def forward(self, x: Tensor) -> Tensor:
        y = self.qnn(x)
        return y


class ModelBuilder:
    def __init__(self, json_file=FILE_PATH, statistical=False):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.statistical = statistical

    def get_model_config(self, model_id):
        for model in self.data:
            if model["id"] == model_id:
                return model
        raise ValueError(f"No model found with id {model_id}")

    def print_model_config(self, model_id):
        config = self.get_model_config(model_id)
        print(json.dumps(config, indent=4))

    def print_all_model_configs(self):
        for model in self.data:
            print(json.dumps(model, indent=4))

    def create_model(self, model_id):
        config = self.get_model_config(model_id)
        # remove 'id' from config as create_model might not expect it
        if self.statistical:
            return QNNStatModel(**config).qnn
        elif config["qkernel"]:
            return QKernelModel(**config).qnn
        else:
            return QNNModel(**config).qnn
