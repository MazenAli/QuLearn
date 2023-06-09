import argparse
import yaml
import os
import uuid
import time
import datetime
import json
import torch
import pennylane as qml
from qulearn.qlayer import IQPEReuploadSU2Parity
from qulearn.loss import RademacherLoss
from qulearn.rademacher import rademacher
from qulearn.datagen import NormalPrior, DataGenRademacher
from qulearn.trainer import AdamTorch


def parse_args():
    # Read the configuration file
    with open("rademacher_iqpe_reup_parity.yaml", "r") as f:
        config = yaml.safe_load(f)
    amsgrad = config.get("amsgrad", False)
    cuda = config.get("cuda", False)

    parser = argparse.ArgumentParser(
        description="Process arguments for QML-MOR rademacher complexity experiments"
    )
    parser.add_argument(
        "--m",
        type=int,
        default=config["m"],
        help="Size of data set",
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
        "--num_data_samples",
        type=int,
        default=config["num_data_samples"],
        help="Number of data samples for estimation",
    )
    parser.add_argument(
        "--num_sigma_samples",
        type=int,
        default=config["num_sigma_samples"],
        help="Numer of sigma samples for estimation",
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
        "--num_epochs",
        type=int,
        default=config["num_epochs"],
        help="Number of optimization steps",
    )
    parser.add_argument(
        "--opt_stop",
        type=float,
        default=config["opt_stop"],
        help="Convergence threshold for optimization",
    )
    parser.add_argument(
        "--stagnation_threshold",
        type=float,
        default=config["stagnation_threshold"],
        help="Stop if relative loss reduction below threshold",
    )
    parser.add_argument(
        "--stagnation_count",
        type=int,
        default=config["stagnation_count"],
        help="Allowed times below threshold",
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
        num_reups, num_layers, num_qubits - 1, 2, requires_grad=True, device=cdevice
    )
    W = torch.randn(2**num_qubits, requires_grad=True, device=cdevice)
    params = [init_theta, theta, W]

    qnn_model = IQPEReuploadSU2Parity(omega)

    # Define qnode
    shots = None
    qdevice = qml.device("lightning.qubit", wires=args.num_qubits, shots=shots)
    qnode = qml.QNode(qnn_model.qfunction, qdevice, interface="torch")

    # Set data generating method
    sizex = num_qubits
    num_sigma_samples = args.num_sigma_samples
    num_data_samples = args.num_data_samples
    prior = NormalPrior(sizex=sizex, seed=seed, device=cdevice)
    datagen = DataGenRademacher(
        prior=prior,
        num_sigma_samples=num_sigma_samples,
        num_data_samples=num_data_samples,
        seed=seed,
        device=cdevice,
    )

    m = args.m
    data = datagen.gen_data(m)
    X = data["X"]
    sigmas = data["sigmas"]

    # Set optimizer
    sigma = sigmas[0]
    loss_fn = RademacherLoss(sigma)
    opt = AdamTorch(
        params=params,
        loss_fn=loss_fn,
        lr=args.lr,
        amsgrad=args.amsgrad,
        num_epochs=args.num_epochs,
        opt_stop=args.opt_stop,
        stagnation_threshold=args.stagnation_threshold,
        stagnation_count=args.stagnation_count,
    )

    # Estimate Rademacher complexity
    start_time = time.time()
    rad_compl = rademacher(model=qnode, opt=opt, X=X, sigmas=sigmas)
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
        "time_taken": time_taken,
        "cdevice": cdevice.type,
        "num_qubits": args.num_qubits,
        "num_layers": args.num_layers,
        "num_reups": args.num_reups,
        "omega": omega,
        "m": m,
        "num_params_gates": num_params_gates,
        "num_params_obs": num_params_obs,
        "num_params": num_params,
        "num_data_samples": num_data_samples,
        "num_sigma_samples": num_sigma_samples,
        "num_epochs": args.num_epochs,
        "opt_stop": args.opt_stop,
        "seed": seed,
        "rademacher": rad_compl.item(),
    }
    exp_id = str(uuid.uuid4())
    direc = args.save_dir
    filename = os.path.join(direc, exp_id + ".json")
    with open(filename, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
