import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision.models.feature_extraction import create_feature_extractor

import wandb
from models import model_attributes
from train import train
from utils import CSVBatchLogger, Logger, log_args, set_seed
from wilds_dataset import get_wilds_data_loaders, log_data

USER = os.environ.get("USER")


def get_data(wandb_enabled=True):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wilds", required=False, default=None
    )  # subgroups = --groups, not (y,g) combo
    parser.add_argument("--groups", required=False, nargs="+")  # Only supports 1 group
    parser.add_argument("--model", default="mobilenet_w1")
    # Confounders
    parser.add_argument("--model_path", "--model_path", default=None)
    parser.add_argument("--epoch_number", type=int, default=0)
    parser.add_argument("--resnet_width", type=int, default=None)

    # Optimization
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--hidden_layer_size", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument(
        "--warmup_steps",
        type=float,
        default=0.1,
        help="<=1 = percentage of total steps. 1+ = number of steps of warmup. only for BERT.",
    )
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.1)
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--save_best", action="store_true", default=False)
    parser.add_argument("--save_last", action="store_true", default=False)
    parser.add_argument("--pretrained", default=True, action="store_true")
    parser.add_argument("--enable_transformations", default=False, action="store_true")

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    args = parser.parse_args()
    args.root_dir = "./data/"

    if wandb_enabled:
        wandb.init(project="waterbirds", entity="vsrivatsa")  # , name="test"
        wandb.config.update(args, allow_val_change=True)
    else:
        wandb.init(mode="disabled")

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, "log.txt"), "w")
    # Record args
    log_args(args, logger)

    set_seed(args.seed)
    args.dataset = "waterbirds"
    (
        train_loader,
        val_loader,
        test_loader,
        train_data,
        val_data,
        test_data,
        dataset_grouper,
    ) = get_wilds_data_loaders(args)
    data = {}
    data["dataset_grouper"] = dataset_grouper
    data["train_loader"] = train_loader
    data["val_loader"] = val_loader
    data["test_loader"] = test_loader
    data["train_data"] = train_data
    data["val_data"] = val_data
    data["test_data"] = test_data
    return data, args, logger, train_data


def load_model(args, train_data):
    if args.model_path:
        return torch.load(args.model_path)

    model = ptcv_get_model(args.model, pretrained=args.pretrained)
    d = list(model.modules())[-1].in_features
    for param in model.parameters():
        param.requires_grad = False

    if args.hidden_layer_size == 0:
        model._modules["output"] = nn.Linear(d, 1)
        return model
    else:
        n_interim = args.hidden_layer_size
        layer1 = nn.Linear(d, n_interim)
        layer_out = nn.Linear(n_interim, 1)
        model._modules["output"] = nn.Sequential(layer1, nn.ReLU(), layer_out)
        return model


def load_criterion(args):
    ## Define the objective
    return torch.nn.BCEWithLogitsLoss(reduction="none")


def train_cli():
    data, args, logger, train_data = get_data()
    if not args.wilds:
        log_data(data, logger)
    model = load_model(args, train_data)

    logger.flush()
    criterion = load_criterion(args)

    train_csv_logger = CSVBatchLogger(
        os.path.join(args.log_dir, "train.csv"),
        train_data.n_groups,
        mode="w",
        wandb=True,
        logger_mode="train",
    )
    val_csv_logger = CSVBatchLogger(
        os.path.join(args.log_dir, "val.csv"),
        train_data.n_groups,
        mode="w",
        wandb=True,
        logger_mode="val",
    )
    test_csv_logger = CSVBatchLogger(
        os.path.join(args.log_dir, "test.csv"),
        train_data.n_groups,
        mode="w",
        wandb=True,
        logger_mode="test",
    )

    dataset_grouper = data["dataset_grouper"]
    train(
        model,
        criterion,
        data,
        logger,
        train_csv_logger,
        val_csv_logger,
        test_csv_logger,
        args,
        epoch_offset=0,
        dataset_grouper=dataset_grouper,
    )

    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()

    for file_name in ["train.csv", "val.csv", "test.csv"]:
        wandb.save(os.path.join(args.log_dir, file_name))
    torch.save(model, os.path.join(wandb.run.dir, "model.pth"))


def final_conv_output():
    data, args, logger, train_data = get_data(wandb_enabled=False)
    if not args.wilds:
        log_data(data, logger)
    model = load_model(args, train_data)
    # print(model.parameters())
    model = create_feature_extractor(model, return_nodes=["features.final_pool"]).to(
        "cuda"
    )
    model.eval()
    dataset_grouper = data["dataset_grouper"]
    all_outputs = np.array([])
    with torch.set_grad_enabled(False):
        for batch_idx, batch in enumerate(data["test_loader"]):
            x = batch[0].to("cuda")
            y = batch[1]
            y = np.expand_dims(y, axis=1)
            outputs = model(x)
            final_layer = outputs["features.final_pool"]
            final_layer_flat = (
                final_layer.reshape((final_layer.shape[0], final_layer.shape[1]))
                .cpu()
                .numpy()
            )
            final_layer_flat_with_outputs = np.hstack((final_layer_flat, y))
            print(final_layer_flat_with_outputs.shape, all_outputs.shape)
            if len(all_outputs) == 0:
                all_outputs = final_layer_flat_with_outputs
            else:
                all_outputs = np.vstack((all_outputs, final_layer_flat_with_outputs))

            print(all_outputs.shape)
    # all_outputs_torch = torch.stack(all_outputs)
    df = pd.DataFrame(all_outputs)
    df.to_csv("waterbirds_mobilenet_final_layer_test.csv", header=False, index=False)
    # print(all_outputs[0])


def main():
    train_cli()
    # final_conv_output()


if __name__ == "__main__":
    main()
