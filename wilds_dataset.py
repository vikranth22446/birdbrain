import os
import time
from collections import Counter

import torch
import torchvision.transforms as transforms
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds.common.grouper import CombinatorialGrouper


def get_wilds_data_loaders(args):
    """
    Loaders a specific wilds dataset based on args.wilds(such as fmow).

    Returns train_loader, val_loader, and test_loader

    """
    target_resolution = (224, 224)

    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset="waterbirds", download=True, root_dir=args.root_dir)
    args.grouper = CombinatorialGrouper(dataset, args.groups)
    randomized_mutations = []
    if args.enable_transformations:
        print("Using random mutations")
        randomized_mutations = [
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33333333333333333333333),
                interpolation=2,
            ),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        print("Using regular mutations")
        randomized_mutations = [
            transforms.Resize(
                target_resolution,
                interpolation=2,
            ),
            # transforms.RandomHorizontalFlip()
        ]
        # randomized_mutations = [
        #     transforms.
        # ]
    dataset_transforms = transforms.Compose(
        randomized_mutations
        + [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # TODO training 780%
    train_data = dataset.get_subset(
        "train", transform=dataset_transforms, frac=args.train_fraction
    )
    # Prepare the standard data loader
    train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size)

    # Get the test set
    test_data = dataset.get_subset("test", transform=dataset_transforms)

    test_loader = get_eval_loader("standard", test_data, batch_size=args.batch_size)
    # Get the val set
    val_data = dataset.get_subset("val", transform=dataset_transforms)

    # Prepare the evaluation data loader
    val_loader = get_eval_loader("standard", val_data, batch_size=args.batch_size)

    for d, n in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        _, g_counts = dataset._eval_grouper.metadata_to_group(
            d.metadata_array, return_counts=True
        )
        # print(len(d), group_counts)
        print("Data loading", n, len(d), g_counts)
        g_counts = [item for item in g_counts]
        d.n_groups = len(g_counts)
        d.group_counts = g_counts
        d.group_str = lambda x: f"region={x}"
    return (
        train_loader,
        val_loader,
        test_loader,
        train_data,
        val_data,
        test_data,
        dataset._eval_grouper,
    )


def log_data(data, logger):
    logger.write("Training Data...\n")
    for group_idx in range(data["train_data"].n_groups):
        logger.write(
            f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts[group_idx]:.0f}\n'
        )
    logger.write("Validation Data...\n")
    for group_idx in range(data["val_data"].n_groups):
        logger.write(
            f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts[group_idx]:.0f}\n'
        )
    if data["test_data"] is not None:
        logger.write("Test Data...\n")
        for group_idx in range(data["test_data"].n_groups):
            logger.write(
                f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts[group_idx]:.0f}\n'
            )
