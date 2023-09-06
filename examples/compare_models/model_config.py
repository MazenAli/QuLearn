"""
This script is used to update a JSON file with model configurations.
The model configurations are specified by the user through command line arguments.
The models are assumed to be of a certain type.
    - The data embedding uses IQPE circuits. It can be repeated num_reuploads times.
    - The number of wires is either equal to the number of features or twice that. In the latter case, the features are simply repeated x_ = (x, x).
    - The variational layer is a simplified RY-CZ 2-design, repeating num_varlayers times.
    - Embedding and variational layers can be repeated num_repeats times. On each repeat layer j, the feature tensor is multiplied by 2**(omega*j).
    - The Hamiltonian is also variational, i.e., contains trainable parameters. Identity is always included (=bias), plus different Z-measurements.

The following parameters are left to vary:
num_features: Number of features in the model.
num_reuploads: Number of reuploads in the model.
omega: Omega parameter for the model.
num_varlayers: Number of variational layers in the model.
num_repeats: Number of times to repeat the experiment.
hamiltonian_type: The type of Hamiltonian used. Can be one of "Z0", "AllWires", "AllWirePairs", "AllWireCombinations".
double_wires: A boolean flag indicating whether to double the number of wires.
"""

import os
import json
import argparse

FILE_PATH = "model_configs.json"


def update_json_file(params, json_file):
    """
    Updates or creates a JSON file with the specified model parameters.

    :param params: A dictionary containing model parameters.
    :param json_file: The name of the JSON file to update or create.
    """
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
    else:
        data = []

    # Generate a unique id for the new model.
    # If there are no existing models, the id will be 1.
    # Otherwise, the id will be one greater than the id of the last added model.
    model_id = data[-1]["id"] + 1 if data else 1

    # Add the 'id' field to the parameters dictionary.
    params["id"] = model_id

    data.append(params)

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to update model configuration file."
    )

    parser.add_argument(
        "--num_features", type=int, required=True, help="Number of features."
    )
    parser.add_argument(
        "--num_reuploads", type=int, required=True, help="Number of reuploads."
    )
    parser.add_argument(
        "--num_varlayers", type=int, required=True, help="Number of variational layers."
    )
    parser.add_argument(
        "--num_repeats", type=int, required=True, help="Number of repeats."
    )
    parser.add_argument("--omega", type=float, required=True, help="Omega parameter.")
    parser.add_argument(
        "--hamiltonian_type",
        type=str,
        required=True,
        choices=["Z0", "AllWires", "AllWirePairs", "AllWireCombinations"],
        help="Type of Hamiltonian.",
    )
    parser.add_argument(
        "--double_wires", action="store_true", help="Flag for doubling the wires."
    )

    return parser.parse_args()


def main(args):
    params = vars(args)
    update_json_file(params, FILE_PATH)


if __name__ == "__main__":
    args = parse_args()
    main(args)
