import argparse
import yaml
import os
import uuid
import time
import datetime
import json
import torch
import pennylane as qml
from qml_mor.models import IQPEReuploadSU2Parity
from qml_mor.capacity import capacity
from qml_mor.datagen import DataGenCapacity
from qml_mor.optimize import AdamTorch


def parse_args():
    # Read the configuration file
    with open("cfg_capacity.yaml", "r") as f:
        config = yaml.safe_load(f)
    amsgrad = config.get("amsgrad", False)
    early_stop = config.get("early_stop", False)
    cuda = config.get("cuda", False)

    parser = argparse.ArgumentParser(
        description="Process arguments for QML-MOR capacity experiments"
    )
    parser.add_argument(
        "--Nmin",
        type=int,
        default=config["Nmin"],
        help="Minimum value of N for experiments",
    )
    parser.add_argument(
        "--Nmax",
        type=int,
        default=config["Nmax"],
        help="Maximum value of N for experiments (included)",
    )
    parser.add_argument(
        "--Nstep",
        type=int,
        default=config["Nstep"],
        help="Step size for N in experiments",
    )
    parser.add_argument(
        "--num_qubits",
        type=int,
        default=config["num_qubits"],
        help="Number of qubits in QNN",
    )
    parser.add_argument(
        "--num_reups",
        type=int,
        default=config["num_reups"],
        help="Number of data reuploads in QNN",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=config["num_layers"],
        help="Number of variational layers in QNN",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=config["omega"],
        help="The exponential feature scaling factor",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=config["num_samples"],
        help="Number of samples for random labels",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config["lr"],
        help="Learning rate",
    )
    parser.add_argument(
        "--amsgrad",
        action="store_true",
        default=amsgrad,
        help="Use amsgrad",
    )
    parser.add_argument(
        "--opt_steps",
        type=int,
        default=config["opt_steps"],
        help="Number of optimization steps",
    )
    parser.add_argument(
        "--opt_stop",
        type=float,
        default=config["opt_stop"],
        help="Convergence threshold for optimization",
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        default=early_stop,
        help="Stops iterations early if the previous capacity is at least as large",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config["seed"],
        help="Random seed for generating dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=config["save_dir"],
        help="Directory for saving results",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=cuda,
        help="Run on a GPU",
    )
    args = parser.parse_args()
    return args


def main(args):
    # Define QNN model
    num_qubits = args.num_qubits
    num_layers = args.num_layers
    num_reups = args.num_reups
    omega = args.omega
    seed = args.seed
    cuda = args.cuda

    if seed is not None:
        torch.manual_seed(seed)

    cdevice = torch.device("cpu")
    if cuda:
        cdevice = torch.device("cuda")

    init_theta = torch.randn(num_reups, num_qubits, requires_grad=True, device=cdevice)
    theta = torch.randn(
        num_reups, num_layers, num_qubits - 1, 2, requires_grad=True
    , device=cdevice)
    W = torch.randn(2**num_qubits, requires_grad=True, device=cdevice)
    params = [init_theta, theta, W]

    qnn_model = IQPEReuploadSU2Parity(params, omega)

    # Define qnode
    shots = None
    qdevice = qml.device("lightning.qubit", wires=args.num_qubits, shots=shots)
    qnode = qml.QNode(qnn_model.qfunction, qdevice, interface="torch")

    # Set optimizer
    loss_fn = torch.nn.MSELoss()
    opt = AdamTorch(
        params=params,
        loss_fn=loss_fn,
        lr=args.lr,
        amsgrad=args.amsgrad,
        opt_steps=args.opt_steps,
        opt_stop=args.opt_stop,
    )

    # Set data generating method
    sizex = num_qubits
    num_samples = args.num_samples
    datagen = DataGenCapacity(
        sizex=sizex, num_samples=num_samples, seed=seed, device=cdevice
    )

    # Estimate capacity
    Nmin = args.Nmin
    Nmax = args.Nmax
    Nstep = args.Nstep
    early_stop=args.early_stop

    start_time = time.time()
    capacities = capacity(
        model=qnode,
        datagen=datagen,
        opt=opt,
        Nmin=Nmin,
        Nmax=Nmax,
        Nstep=Nstep,
        early_stop=early_stop
    )
    end_time = time.time()
    time_taken = end_time - start_time

    # Save results
    clock = datetime.datetime.now().strftime("%H:%M:%S")
    date = datetime.date.today().strftime("%d/%m/%Y")
    creation_date = date + ", " + clock

    num_params_gates = torch.numel(init_theta) + torch.numel(theta)
    num_params_obs = torch.numel(W)
    num_params = num_params_gates + num_params_obs

    results = {
        "date": creation_date,
        "cdevice": cdevice.type,
        "time_taken": time_taken,
        "num_qubits": args.num_qubits,
        "num_layers": args.num_layers,
        "num_reups": args.num_reups,
        "omega": omega,
        "Nmin": Nmin,
        "Nmax": Nmax,
        "Nstep": Nstep,
        "num_params_gates": num_params_gates,
        "num_params_obs": num_params_obs,
        "num_params": num_params,
        "num_samples": num_samples,
        "opt_steps": args.opt_steps,
        "opt_stop": args.opt_stop,
        "seed": seed,
        "capacities": capacities,
    }
    exp_id = str(uuid.uuid4())
    direc = args.save_dir
    filename = os.path.join(direc, exp_id + ".json")
    with open(filename, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
