import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from run_expt import create_feature_extractor, get_data, load_model


def predict_outputs_with_groups(data, model, data_loader_name="train_loader"):
    dataset_grouper = data["dataset_grouper"]
    all_outputs = np.array([])
    all_outputs_with_groups = np.array([])
    print("Computing csv for ", data_loader_name)
    with torch.set_grad_enabled(False):
        for batch in tqdm(data[data_loader_name]):
            x = batch[0].to("cuda")
            y = batch[1]
            metadata = batch[2]
            g = dataset_grouper.metadata_to_group(metadata).unsqueeze(1)
            y = np.expand_dims(y, axis=1).astype(int)

            outputs = model(x)
            final_layer = outputs["features.final_pool"]
            final_layer_flat = (
                final_layer.reshape((final_layer.shape[0], final_layer.shape[1]))
                .cpu()
                .numpy()
            )
            final_layer_flat_with_outputs = np.hstack((final_layer_flat, y))
            if len(all_outputs) == 0:
                all_outputs = final_layer_flat_with_outputs
            else:
                all_outputs = np.vstack((all_outputs, final_layer_flat_with_outputs))
            final_layer_flat_with_outputs = np.hstack((final_layer_flat, y, g))
            if len(all_outputs_with_groups) == 0:
                all_outputs_with_groups = final_layer_flat_with_outputs
            else:
                all_outputs_with_groups = np.vstack(
                    (all_outputs_with_groups, final_layer_flat_with_outputs)
                )
    df = pd.DataFrame(all_outputs)
    df.to_csv(
        f"waterbirds_mobilenet_final_layer_{data_loader_name}.csv",
        header=False,
        index=False,
    )

    df = pd.DataFrame(all_outputs_with_groups)
    df.to_csv(
        f"waterbirds_mobilenet_final_layer_{data_loader_name}_group.csv",
        header=False,
        index=False,
    )


def compute_compressed_csv():
    data, args, logger, train_data = get_data(wandb_enabled=False)
    model = load_model(args, train_data)
    print(model.parameters())
    model = create_feature_extractor(model, return_nodes=["features.final_pool"]).to(
        "cuda"
    )
    model.eval()

    predict_outputs_with_groups(data, model, data_loader_name="train_loader")
    predict_outputs_with_groups(data, model, data_loader_name="val_loader")
    predict_outputs_with_groups(data, model, data_loader_name="test_loader")


if __name__ == "__main__":
    compute_compressed_csv()
