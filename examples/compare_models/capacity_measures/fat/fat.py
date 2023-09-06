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
from qulearn.datagen import DataGenFat, UniformPrior
from qulearn.trainer import SupervisedTrainer
from qulearn.fat import fat_shattering_dim
from model_builder import ModelBuilder, CDEV, QDEV

CONFIGS_PATH = "../../model_configs.json"


def parse_args():
    # Read the configuration file
    with open("fat_defaults.yaml", "r") as f:
        config = yaml.safe_load(f)
    amsgrad = config.get("amsgrad", False)

    parser = argparse.ArgumentParser(
        description="Process arguments for fat-shattering dimension experiments"
    )
    parser.add_argument(
        "--dmin",
        type=int,
        default=config["dmin"],
        help="Minimum value of d for experiments",
    )
    parser.add_argument(
        "--dmax",
        type=int,
        default=config["dmax"],
        help="Maximum value of d for experiments (included)",
    )
    parser.add_argument(
        "--dstep",
        type=int,
        default=config["dstep"],
        help="Step size for d in experiments",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=config["gamma"],
        help="Margin value for fat shattering",
    )
    parser.add_argument(
        "--Sb",
        type=int,
        default=config["Sb"],
        help="Number of binary samples to check shattering",
    )
    parser.add_argument(
        "--Sr",
        type=int,
        default=config["Sr"],
        help="Number of level offset samples to check shattering",
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
        help="Number of optimization epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["batch_size"],
        help="Batch size for optimization",
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

    model_builder = ModelBuilder(CONFIGS_PATH)
    configs = model_builder.data
    for config in configs:
        model_id = config["id"]
        model = model_builder.create_model(model_id)
        optimizer = Adam(model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        loss_fn = torch.nn.MSELoss()
        logger = logging.getLogger("fat")
        logger.setLevel(level=logging.INFO)
        logger.info(f"=============== Model ID: {model_id} | START ===============")
        trainer = SupervisedTrainer(
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=args.num_epochs,
            logger=logger,
        )

        sizex = model.num_features
        prior = UniformPrior(sizex=sizex, seed=args.seed, device=CDEV)
        datagen = DataGenFat(
            prior=prior,
            Sb=args.Sb,
            Sr=args.Sr,
            gamma=2.0 * args.gamma,
            seed=args.seed,
            device=CDEV,
            batch_size=args.batch_size,
        )

        start_time = time.time()
        dim = fat_shattering_dim(
            model=model,
            datagen=datagen,
            trainer=trainer,
            dmin=args.dmin,
            dmax=args.dmax,
            gamma=args.gamma,
            dstep=args.dstep,
        )
        end_time = time.time()
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
            "dmin": args.dmin,
            "dmax": args.dmax,
            "dstep": args.dstep,
            "Sb": args.Sb,
            "Sr": args.Sr,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "seed": seed,
            "dimension": dim,
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
