import argparse
import qml_mor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process arguments for QML-MOR capacity experiments"
    )
    parser.add_argument(
        "--Nmin", type=int, default=1, help="Minimum value of N for experiments"
    )
    parser.add_argument(
        "--Nmax", type=int, default=1, help="Maximum value of N for experiments"
    )
    parser.add_argument(
        "--Nstep", type=int, default=1, help="Step size for N in experiments"
    )
    parser.add_argument(
        "--num_qubits", type=int, default=2, help="Number of qubits in QNN"
    )
    parser.add_argument(
        "--num_reups",
        type=int,
        default=1,
        help="Number of data reuploads in QNN",
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of variational layers in QNN"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of samples for random labels",
    )
    parser.add_argument(
        "--opt_steps", type=int, default=300, help="Number of optimization steps"
    )
    parser.add_argument(
        "--opt_stop",
        type=float,
        default=1e-16,
        help="Convergence threshold for optimization",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for generating dataset"
    )
    args = parser.parse_args()
    return args
