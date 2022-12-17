import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from utils import set_seed


def gaussian_noise(x, mu, std):
    noise = np.random.normal(mu, std, size=x.shape)
    x_noisy = x + noise
    return x_noisy


def get_x_y(csv_name):
    Xs = []
    ys = []
    gs = []
    with open(csv_name) as f:
        lines = f.readlines()
        for line in lines:
            *x, y, g = line.strip().split(",")
            Xs.append([float(xi) for xi in x])
            ys.append(int(float(y)))
            gs.append(int(float(g)))

    X = np.array(Xs)
    y = np.array(ys)
    g = np.array(gs)
    return X, y, g


def run_experiment(
    random_seed=1, hidden_layer_size=1, num_epochs=40, noise_std=0, snr_used=0
):
    set_seed(random_seed)

    X, y, g = get_x_y("waterbirds_mobilenet_final_layer_train_loader_group.csv")
    if noise_std != 0:
        X = gaussian_noise(X, mu=0, std=noise_std)
        print("Added noise with std", noise_std)

    X_val, y_val, g_val = get_x_y(
        "waterbirds_mobilenet_final_layer_val_loader_group.csv"
    )
    X_test, y_test, g_test = get_x_y(
        "waterbirds_mobilenet_final_layer_test_loader_group.csv"
    )

    def dataloader(X, y, bs=128):
        n = X.shape[0]
        indices = list(range(0, n, bs))
        np.random.shuffle(indices)
        for i in indices:
            Xi = torch.tensor(X[i : i + bs]).float()
            yi = torch.tensor(y[i : i + bs]).long()
            yield Xi, yi

    if hidden_layer_size == 0:
        model = nn.Sequential(
            nn.Linear(1024, 1),
        )
    else:
        model = nn.Sequential(
            nn.Linear(1024, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
        )
    model.train()

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    def compute_acc(model, input_dataset, y, groups):
        outputs = model(torch.tensor(input_dataset).float()).detach().squeeze(1)
        n0 = np.sum(y == 0)
        n1 = np.sum(y == 1)
        p0 = n0 / (n0 + n1)
        p1 = n1 / (n0 + n1)
        logp0 = np.log2(p0)
        logp1 = np.log2(p1)
        entropy = -(p0 * logp0 + p1 * logp1)
        logps = torch.tensor(np.where(y == 0, logp0, logp1)).float()
        accuracy = -((torch.where(outputs > 0., 1, 0) == torch.tensor(y).long()) * logps).sum().item() / (input_dataset.shape[0] * entropy)
        group_acc = []
        for i in range(4):
            indices = (groups == float(i)).nonzero()
            group_dataset = input_dataset[indices]
            group_outputs = outputs[indices]
            group_y = y[indices]
            group_acc.append((torch.where(group_outputs > 0., 1, 0) == torch.tensor(group_y).long()).sum().item() / group_dataset.shape[0])
        return accuracy, group_acc

    mean_x = np.mean(X)
    std_x = np.std(X)
    metrics = {
        "hidden_layer_size": hidden_layer_size,
        "capacity": 1024 * (hidden_layer_size + 1) + hidden_layer_size + 1,
        "num_epochs": num_epochs,
        "random_seed": random_seed,
        "noise_std": noise_std,
        "snr_used": snr_used,
        "train_loss": [],
        "train_acc": [],
        "group0_acc": [],
        "group1_acc": [],
        "group2_acc": [],
        "group3_acc": [],
        "val_acc": [],
        "val_group0_acc": [],
        "val_group1_acc": [],
        "val_group2_acc": [],
        "val_group3_acc": [],
        "val_loss": [],
        "landbird_bias": [],
        "test_acc": 0.0,
        "test_group0_acc": 0.0,
        "test_group1_acc": 0.0,
        "test_group2_acc": 0.0,
        "test_group3_acc": 0.0,
    }

    for epoch in range(num_epochs):
        losses = []
        for Xb, yb in dataloader(X, y):
            optimizer.zero_grad()
            yhat = model(Xb)
            loss = criterion(yhat, yb.unsqueeze(1).float()).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        with torch.set_grad_enabled(False):
            train_acc, train_group_acc = compute_acc(model, X, y, g)
            random_image = torch.rand(1024, 1024)
            test_bias_input = (random_image - mean_x) / std_x
            bias_pred = torch.where(model(test_bias_input) > 0.0, 1, 0).numpy()
            num_class_0, num_class_1 = np.sum(bias_pred == 0), np.sum(bias_pred == 1)
            landbird_bias = num_class_0 / (num_class_0 + num_class_1)
            waterbird_bias = 1 - landbird_bias

            val_acc, val_group_acc = compute_acc(model, X_val, y_val, g_val)
            metrics["train_acc"].append(train_acc)
            metrics["group0_acc"].append(train_group_acc[0])
            metrics["group1_acc"].append(train_group_acc[1])
            metrics["group2_acc"].append(train_group_acc[2])
            metrics["group3_acc"].append(train_group_acc[3])

            metrics["val_acc"].append(val_acc)
            metrics["val_group0_acc"].append(val_group_acc[0])
            metrics["val_group1_acc"].append(val_group_acc[1])
            metrics["val_group2_acc"].append(val_group_acc[2])
            metrics["val_group3_acc"].append(val_group_acc[3])

            metrics["train_loss"].append(np.mean(losses))
            metrics["landbird_bias"].append(landbird_bias)

            # print(f"epoch: {epoch + 1}, accuracy: {train_acc},{train_group_acc} loss: {np.mean(losses)}, val_accuracy: {val_acc}, {val_group_acc}, landbird bias: {landbird_bias}, , landbird bias: {waterbird_bias}")

    test_acc, test_group_acc = compute_acc(model, X_test, y_test, g_test)
    # print("Test acc", test_acc, test_group_acc)
    metrics["test_acc"] = test_acc
    metrics["test_group0_acc"] = test_group_acc[0]
    metrics["test_group1_acc"] = test_group_acc[1]
    metrics["test_group2_acc"] = test_group_acc[2]
    metrics["test_group3_acc"] = test_group_acc[3]
    return metrics


def run_hidden_layer_size_experiment(random_seed=0, num_epochs=40):
    start_time = time.perf_counter()
    metrics_per_hidden_layer_size = {}
    for hidden_layer_size in tqdm([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]):
        metrics_per_hidden_layer_size[hidden_layer_size] = run_experiment(
            random_seed=0, hidden_layer_size=hidden_layer_size, num_epochs=num_epochs
        )
    print("Total time", time.perf_counter() - start_time)
    with open("experiment_hidden_layer_results.json", "w") as f:
        json.dump(metrics_per_hidden_layer_size, f, indent=2)


def run_single_layer_experiment(random_seed=0, num_epochs=40):
    start_time = time.perf_counter()
    metrics = run_experiment(
        random_seed=random_seed, hidden_layer_size=0, num_epochs=num_epochs
    )
    print("Total time", time.perf_counter() - start_time)
    with open("experiment_single_layer_results.json", "w") as f:
        json.dump(metrics, f, indent=2)


def run_snr_experiment(random_seed=0, num_epochs=40):
    X, _, _ = get_x_y("waterbirds_mobilenet_final_layer_train_loader_group.csv")
    # ref: https://stats.stackexchange.com/questions/548619/how-to-add-and-vary-gaussian-noise-to-input-data
    sp = np.mean(X ** 2)
    start_time = time.perf_counter()
    snr_metrics = {}
    for snr in tqdm(
        [
            1e-9,
            1e-8,
            1e-7,
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            0.273,
            1.0,
            1e1,
            1e2,
            1e3,
            1e4,
            1e5,
        ]
    ):
        std_n = (sp / snr) ** 0.5  # Noise std. deviation
        snr_metrics[snr] = run_experiment(
            random_seed=random_seed,
            hidden_layer_size=2,
            num_epochs=num_epochs,
            noise_std=std_n,
            snr_used=snr,
        )
    # print("Total time", time.perf_counter() - start_time)
    with open("experiment_snr_results_2.json", "w") as f:
        json.dump(snr_metrics, f, indent=2)

if __name__ == "__main__":
    # run_hidden_layer_size_experiment(num_epochs=100)
    # run_snr_experiment(num_epochs=100)
    # run_single_layer_experiment(num_epochs=100)
    run_snr_experiment()
  