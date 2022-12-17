import os
# from profilehooks import profile
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import wandb
from loss import LossComputer


# @profile(immediate=True)
def run_epoch(
    epoch,
    model,
    optimizer,
    loader,
    loss_computer,
    logger,
    csv_logger,
    args,
    is_training,
    dataset_grouper,
    show_progress=False,
    log_every=50,
    scheduler=None,
):
    """
    scheduler is only used inside this function if model is bert.
    """
    epoch_start_time = time.time()

    if is_training:
        model.train()
        if "bert" in args.model:
            model.zero_grad()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):

            # batch = tuple(t.to('cuda') for t in batch)
            x = batch[0].to("cuda")
            y = batch[1].to("cuda")
            metadata = batch[2]
            g = dataset_grouper.metadata_to_group(metadata).to("cuda")
            outputs = model(x)
            loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training:
                if "bert" in args.model:
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            if is_training and (batch_idx + 1) % log_every == 0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            if csv_logger:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()


def load_optimizer_and_scheduler(args, model, dataset):
    # BERT uses its own scheduler and optimizer
    if "bert" in args.model:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon
        )
        t_total = len(dataset["train_loader"]) * args.n_epochs
        print(f"\nt_total is {t_total}\n")
        if args.warmup_steps <= 1:
            warmup_steps = args.warmup_steps * t_total
        else:
            warmup_steps = args.warmup_steps
        print(f"\warmup_steps is {warmup_steps}\n")
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=warmup_steps, t_total=t_total
        )
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            # momentum=0.9,
            weight_decay=args.weight_decay,
        )
        if args.step_scheduler:
            scheduler = StepLR(
                optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
            )
        elif args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08,
            )
        else:
            scheduler = None
    return optimizer, scheduler


# 75% class 1
# 25% class 2


def train(
    model,
    criterion,
    dataset,
    logger,
    train_csv_logger,
    val_csv_logger,
    test_csv_logger,
    args,
    epoch_offset,
    dataset_grouper,
):
    model = model.to("cuda")

    train_loss_computer = LossComputer(
        criterion,
        is_robust=False,
        dataset=dataset["train_data"],
        step_size=args.robust_step_size,
    )

    optimizer, scheduler = load_optimizer_and_scheduler(args, model, dataset)
    with torch.set_grad_enabled(False):
        random_image = torch.rand(256, 3, 224, 224)
        test_bias_input = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )(random_image).to("cuda")
        test_bias_output = model(test_bias_input)
        test_bias_preds = torch.argmax(test_bias_output, dim=1).cpu().numpy()
        num_class_0 = np.sum(test_bias_preds == 0)
        num_class_1 = np.sum(test_bias_preds == 1)
        wandb.log(
            {
                f"{test_csv_logger.logger_mode}/model_bias_landbird": num_class_0
                / (num_class_0 + num_class_1)
            }
        )
        wandb.log(
            {
                f"{test_csv_logger.logger_mode}/model_bias_waterbird": num_class_1
                / (num_class_0 + num_class_1)
            }
        )
        print(num_class_0, num_class_1)

    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset + args.n_epochs):
        logger.write("\nEpoch [%d]:\n" % epoch)
        logger.write(f"Training:\n")
        run_epoch(
            epoch,
            model,
            optimizer,
            dataset["train_loader"],
            train_loss_computer,
            logger,
            train_csv_logger,
            args,
            is_training=True,
            dataset_grouper=dataset_grouper,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler,
        )

        # learning rate scheduler
        if args.step_scheduler:
            scheduler.step()  # step scheduler doesn't need loss as an input

        logger.write(f"\nValidation:\n")
        val_loss_computer = LossComputer(
            criterion,
            is_robust=False,
            dataset=dataset["val_data"],
            step_size=args.robust_step_size,
            alpha=args.alpha,
        )
        run_epoch(
            epoch,
            model,
            optimizer,
            dataset["val_loader"],
            val_loss_computer,
            logger,
            val_csv_logger,
            args,
            is_training=False,
            dataset_grouper=dataset_grouper,
        )

        # Test set; don't print to avoid peeking
        if dataset["test_data"] is not None:
            test_loss_computer = LossComputer(
                criterion,
                is_robust=False,
                dataset=dataset["test_data"],
                step_size=args.robust_step_size,
                alpha=args.alpha,
            )
            run_epoch(
                epoch,
                model,
                optimizer,
                dataset["test_loader"],
                test_loss_computer,
                None,
                test_csv_logger,
                args,
                is_training=False,
                dataset_grouper=dataset_grouper,
            )

            with torch.set_grad_enabled(False):
                random_image = torch.rand(256, 3, 224, 224)
                test_bias_input = transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )(random_image).to("cuda")
                test_bias_output = model(test_bias_input)
                test_bias_preds = torch.argmax(test_bias_output, dim=1).cpu().numpy()
                num_class_0 = np.sum(test_bias_preds == 0)
                num_class_1 = np.sum(test_bias_preds == 1)
                wandb.log(
                    {
                        f"{test_csv_logger.logger_mode}/model_bias_landbird": num_class_0
                        / (num_class_0 + num_class_1)
                    }
                )
                wandb.log(
                    {
                        f"{test_csv_logger.logger_mode}/model_bias_waterbird": num_class_1
                        / (num_class_0 + num_class_1)
                    }
                )
                print(num_class_0, num_class_1)

        # Inspect learning rates
        if (epoch + 1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group["lr"]
                logger.write("Current lr: %f\n" % curr_lr)

        if args.scheduler and args.model != "bert":
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(
                    val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss
                )
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss)  # scheduler step to update lr at the end of epoch

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir, "%d_model.pth" % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, "last_model.pth"))

        if args.save_best:
            if args.robust or args.reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.write(f"Current validation accuracy: {curr_val_acc}\n")
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, "best_model.pth"))
                logger.write(f"Best model saved at epoch {epoch}\n")

        if args.automatic_adjustment:
            gen_gap = (
                val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            )
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write("Adjustments updated\n")
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f"  {train_loss_computer.get_group_name(group_idx)}:\t"
                    f"adj = {train_loss_computer.adj[group_idx]:.3f}\n"
                )
        logger.write("\n")
