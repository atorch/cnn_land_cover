import glob
import os
import pickle

from constants import SAVED_MODELS_DIR


def main():

    config_paths = glob.glob(os.path.join(SAVED_MODELS_DIR, "*_model_config.pickle"))

    for config_path in config_paths:

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        history_path = config_path.replace("_model_config.pickle", "_history.pickle")

        with open(history_path, "rb") as f:
            history = pickle.load(f)

        max_val_loss, min_val_loss = max(history["val_loss"]), min(history["val_loss"])
        dropout_rate = config["dropout_rate"]
        print(f"{history_path}: config dropout_rate = {dropout_rate}, min val_loss = {min_val_loss}")


if __name__ == "__main__":
    main()
