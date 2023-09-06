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
from qulearn.datagen import NormalPrior, DataGenRademacher
from qulearn.trainer import SupervisedTrainer
from qulearn.rademacher import rademacher
from model_builder import ModelBuilder, CDEV, QDEV

CONFIGS_PATH = "../../model_configs.json"


def parse_args():
    with open("rademacher_defaults.yaml", "r") as f:
        config = yaml.safe_load(f)
    amsgrad = config.get("amsgrad", False)

    parser = argparse.ArgumentParser(
        description="Process arguments for Rademacher complexity experiments"
    )
    parser.add_argument(
        "--m",
        type=int,
        default=config["m"],
        help="Size of data set",
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
        help="Number of sigma samples for estimation",
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
        logger = logging.getLogger("rademacher")
        logger.setLevel(level=logging.INFO)
        logger.info(f"=============== Model ID: {model_id} | START ===============")
        trainer = SupervisedTrainer(
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=args.num_epochs,
            logger=logger,
        )

        sizex = model.num_features
        prior = NormalPrior(sizex=sizex, seed=args.seed, device=CDEV)
        datagen = DataGenRademacher(
            prior=prior, num_sigma_samples=args.num_sigma_samples, seed=args.seed
        )

        start_time = time.time()
        data = datagen.gen_data(args.m)
        rad = rademacher(
            model=model,
            trainer=trainer,
            X=data["X"],
            sigmas=data["sigmas"],
            datagen=datagen,
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
            "m": args.m,
            "num_data_samples": args.num_data_samples,
            "num_sigma_sample": args.num_sigma_samples,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "seed": seed,
            "rademacher": rad.item(),
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
