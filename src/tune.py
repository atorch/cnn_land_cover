import glob
import os
import pickle

import numpy as np
import pandas as pd

from constants import SAVED_MODELS_DIR


def get_min_val_loss(config_path):

    history_path = config_path.replace("_model_config.pickle", "_history.pickle")

    with open(history_path, "rb") as f:
        history = pickle.load(f)

    return min(history["val_loss"])


def main():

    config_paths = glob.glob(os.path.join(SAVED_MODELS_DIR, "*_model_config.pickle"))

    val_losses, configs = [], []

    for config_path in config_paths:

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        configs.append(config)

        min_val_loss = get_min_val_loss(config_path)
        val_losses.append(min_val_loss)

    # TODO Also get total number of trainable params for each model
    df = pd.DataFrame(configs)
    df["min_val_loss"] = val_losses
    df["config_path"] = config_paths

    df["n_test_scenes"] = df["test_scenes"].apply(len)

    cols = [
        "base_n_filters",
        "dropout_rate",
        "min_val_loss",
        "config_path",
        "additional_filters_per_block",
        "dilation_rate",
        "kernel_size",
        "test_scenes",
        "n_test_scenes",
    ]
    print(df[cols])

    best_index = np.argmin(val_losses)
    best_config_path = config_paths[best_index]
    print(f"model config with min val_loss: {best_config_path}")
    print(f"min val_loss: {np.min(val_losses)}")

    df[cols].to_csv("tuning_results.csv", index=False)


if __name__ == "__main__":
    main()
