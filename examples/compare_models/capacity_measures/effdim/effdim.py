import sys

sys.path.append("../..")

import argparse
import yaml
import os
import uuid
import time
import datetime
import json
import logging
import torch
import numpy as np
from qulearn.datagen import NormalPrior, generate_model_lhs_samples
from qulearn.fim import compute_effdim
from model_builder import ModelBuilder, CDEV, QDEV, DTYPE

CONFIGS_PATH = "../../model_configs.json"


def parse_args():
    with open("effdim_defaults.yaml", "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Process arguments for effective dimension experiments"
    )
    parser.add_argument(
        "--num_data_samples",
        type=int,
        default=config["num_data_samples"],
        help="Number of data samples",
    )
    parser.add_argument(
        "--num_parameter_samples",
        type=int,
        default=config["num_parameter_samples"],
        help="Number of parameter samples",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=config["gamma"],
        help="gamma constant for effective dimension",
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
    args = parser.parse_args()
    return args


def main(args):
    seed = args.seed
    if seed is not None:
        torch.manual_seed(seed)

    model_builder = ModelBuilder(CONFIGS_PATH, statistical=True)
    configs = model_builder.data
    for config in configs:
        model_id = config["id"]
        model = model_builder.create_model(model_id)
        logger = logging.getLogger("effdim")
        logger.setLevel(level=logging.INFO)
        logger.info(f"=============== Model ID: {model_id} | START ===============")

        sizex = model.num_features
        prior = NormalPrior(sizex=sizex, seed=args.seed, device=CDEV)

        start_time = time.time()
        X = prior.gen_data(args.num_data_samples)
        lower_bound = -2.0 * np.pi
        upper_bound = 2.0 * np.pi
        parameter_list = generate_model_lhs_samples(
            model,
            args.num_parameter_samples,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            device=CDEV,
            dtype=DTYPE,
            seed=args.seed,
        )
        logger.info("Parameters generated")
        dimension = sum(p.numel() for p in model.parameters() if p.requires_grad)
        vol = (4.0 * np.pi) ** dimension
        volume = torch.tensor(vol, device=CDEV, dtype=DTYPE)
        weights = torch.tensor(vol, device=CDEV, dtype=DTYPE)
        gamma = torch.tensor(args.gamma, device=CDEV, dtype=DTYPE)
        logger.info("Entering effective dimension estimation")
        effdim = compute_effdim(
            model=model,
            features=X,
            param_list=parameter_list,
            weights=weights,
            volume=volume,
            gamma=gamma,
        )
        end_time = time.time()
        logger.info("Done")
        time_taken = end_time - start_time

        # Save results
        clock = datetime.datetime.now().strftime("%H:%M:%S")
        date = datetime.date.today().strftime("%d/%m/%Y")
        creation_date = date + ", " + clock

        results = {
            "date": creation_date,
            "cdevice": CDEV.type,
            "qdevice": QDEV["name"],
            "shots": QDEV["shots"],
            "model_id": model_id,
            "time_taken": time_taken,
            "num_data_samples": args.num_data_samples,
            "num_parameter_samples": args.num_parameter_samples,
            "gamma": args.gamma,
            "seed": seed,
            "effdim": effdim.item(),
        }
        exp_id = str(uuid.uuid4())
        direc = args.save_dir
        filename = os.path.join(direc, exp_id + ".json")
        with open(filename, "w") as f:
            json.dump(results, f)
        logger.info(f"=============== Model ID: {model_id} | END ===============")


if __name__ == "__main__":
    args = parse_args()
    main(args)
