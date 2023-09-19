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
from torch.optim import Adam
from qulearn.datagen import DataGenCapacity
from qulearn.trainer import SupervisedTrainer, RidgeRegression
from qulearn.memory import memory
from model_builder import ModelBuilder, CDEV, QDEV

CONFIGS_PATH = "../../model_configs.json"


def parse_args():
    # Read the configuration file
    with open("memory_defaults.yaml", "r") as f:
        config = yaml.safe_load(f)
    amsgrad = config.get("amsgrad", False)
    early_stop = config.get("early_stop", False)

    parser = argparse.ArgumentParser(
        description="Process arguments for memory capacity experiments"
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
        "--lambda_reg",
        type=float,
        default=config["lambda_reg"],
        help="Regularization parameter ridge regression",
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
        help="Number of optimization epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["batch_size"],
        help="Batch size for optimization",
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        default=early_stop,
        help="Stops iterations early if the previous capacity is at least as large",
    )
    parser.add_argument(
        "--stop_count",
        type=int,
        default=config["stop_count"],
        help="Allowed number of times capacity not improving",
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
        "--models",
        type=int,
        nargs="+",
        default=config.get("models", []),
        help="List of model IDs",
    )
    args = parser.parse_args()
    return args


def main(args):
    seed = args.seed
    if seed is not None:
        torch.manual_seed(seed)

    model_builder = ModelBuilder(CONFIGS_PATH)
    configs = model_builder.data
    for config in configs:
        model_id = config["id"]
        qkernel = config["qkernel"]
        logger = logging.getLogger("memory")
        logger.setLevel(level=logging.INFO)

        if model_id not in args.models:
            logger.info(f"*** Model ID {model_id} not in list, skipping... ***")
            continue

        model = model_builder.create_model(model_id)
        optimizer = Adam(model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        loss_fn = torch.nn.MSELoss()
        metrics = {"mse_loss": loss_fn}
        logger.info(f"=============== Model ID: {model_id} | START ===============")

        if qkernel:
            trainer = RidgeRegression(
                lambda_reg=args.lambda_reg, metrics=metrics, logger=logger
            )
        else:
            trainer = SupervisedTrainer(
                optimizer=optimizer,
                loss_fn=loss_fn,
                num_epochs=args.num_epochs,
                logger=logger,
            )

        sizex = model.num_features
        datagen = DataGenCapacity(
            sizex=sizex,
            num_samples=args.num_samples,
            seed=seed,
            device=CDEV,
            batch_size=args.batch_size,
        )

        start_time = time.time()
        capacities = memory(
            model,
            datagen,
            trainer,
            args.Nmin,
            args.Nmax,
            args.Nstep,
            args.early_stop,
            args.stop_count,
        )
        end_time = time.time()
        time_taken = end_time - start_time

        best_capacity = max(capacities, key=lambda x: x[-1])

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
            "Nmin": args.Nmin,
            "Nmax": args.Nmax,
            "Nstep": args.Nstep,
            "num_samples": args.num_samples,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "seed": seed,
            "capacity": best_capacity[-1],
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
