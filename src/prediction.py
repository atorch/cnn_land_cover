import os

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import numpy as np
import rasterio

from cnn import PIXELS, get_output_names
from constants import CDL_CLASSES_TO_MASK, IMAGE_SHAPE, MODEL_CONFIG
from normalization import get_X_normalized
from utils import get_config, get_label_encoder_and_mapping


def get_colormap(label_encoder):

    building = label_encoder.transform(["building"])[0]
    corn_soy = label_encoder.transform(["corn_soy"])[0]
    developed = label_encoder.transform(["developed"])[0]
    forest = label_encoder.transform(["forest"])[0]
    other = label_encoder.transform(["other"])[0]
    pasture = label_encoder.transform(["pasture"])[0]
    road = label_encoder.transform(["road"])[0]
    water = label_encoder.transform(["water"])[0]
    wetlands = label_encoder.transform(["wetlands"])[0]

    # TODO Put this mapping in yml config?
    return {
        building: {"rgb": (102, 51, 0), "name": "building"},
        corn_soy: {"rgb": (230, 180, 30), "name": "corn_soy"},
        developed: {"rgb": (224, 224, 224), "name": "developed"},
        forest: {"rgb": (0, 102, 0), "name": "forest"},
        other: {"rgb": (255, 255, 255), "name": "other"},
        pasture: {"rgb": (172, 226, 118), "name": "pasture"},
        road: {"rgb": (128, 128, 128), "name": "road"},
        water: {"rgb": (0, 102, 204), "name": "water"},
        wetlands: {"rgb": (0, 153, 153), "name": "wetlands"},
    }


def get_pixel_predictions(model, n_pixel_classes, X_normalized, image_shape):

    pixel_predictions = np.zeros(X_normalized.shape[:2] + (n_pixel_classes,))

    output_names = get_output_names(model)
    pixels_output_index = np.where(np.array(output_names) == PIXELS)[0][0]

    n_predictions = np.zeros_like(pixel_predictions, dtype="uint8")

    # Note: predicting on entire NAIP scene requires too much memory, so we cut the image into
    # four parts (which may slightly overlap) and predict on each of them individually
    prediction_width = image_shape[0] * (
        ((X_normalized.shape[0] // image_shape[0]) // 2) + 1
    )
    prediction_height = image_shape[1] * (
        ((X_normalized.shape[1] // image_shape[1]) // 2) + 1
    )

    # TODO Variables for slice indices, they're used in three places
    X_normalized_top_left = X_normalized[
        np.newaxis, :prediction_width, :prediction_height, :
    ]
    X_normalized_top_right = X_normalized[
        np.newaxis, -prediction_width:, :prediction_height, :
    ]
    X_normalized_bottom_left = X_normalized[
        np.newaxis, :prediction_width, -prediction_height:, :
    ]
    X_normalized_bottom_right = X_normalized[
        np.newaxis, -prediction_width:, -prediction_height:, :
    ]

    model_predictions_top_left = model.predict(X_normalized_top_left)
    model_predictions_top_right = model.predict(X_normalized_top_right)
    model_predictions_bottom_left = model.predict(X_normalized_bottom_left)
    model_predictions_bottom_right = model.predict(X_normalized_bottom_right)

    pixel_predictions[
        :prediction_width, :prediction_height, :
    ] += model_predictions_top_left[pixels_output_index][0]
    pixel_predictions[
        -prediction_width:, :prediction_height, :
    ] += model_predictions_top_right[pixels_output_index][0]
    pixel_predictions[
        :prediction_width, -prediction_height:, :
    ] += model_predictions_bottom_left[pixels_output_index][0]
    pixel_predictions[
        -prediction_width:, -prediction_height:, :
    ] += model_predictions_bottom_right[pixels_output_index][0]

    n_predictions[:prediction_width, :prediction_height, :] += 1
    n_predictions[-prediction_width:, :prediction_height, :] += 1
    n_predictions[:prediction_width, -prediction_height:, :] += 1
    n_predictions[-prediction_width:, -prediction_height:, :] += 1

    has_predictions = np.where(n_predictions > 0)
    pixel_predictions[has_predictions] = (
        pixel_predictions[has_predictions] / n_predictions[has_predictions]
    )

    return pixel_predictions


def predict_pixels_entire_scene(
    model, naip_path, X_mean_train, X_std_train, image_shape, label_encoder, colormap
):

    print(f"Predicting on {naip_path}")
    with rasterio.open(naip_path) as naip:

        X = naip.read()
        profile = naip.profile

    # Swap shape from (band, height, width) to (width, height, band)
    X = np.swapaxes(X, 0, 2)
    X_normalized = get_X_normalized(X, X_mean_train, X_std_train)

    # Predictions have shape (width, height, n_classes)
    n_pixel_classes = len(label_encoder.classes_)

    pixel_predictions = get_pixel_predictions(
        model, n_pixel_classes, X_normalized, image_shape
    )

    profile["dtype"] = str(pixel_predictions.dtype)
    profile["count"] = n_pixel_classes

    naip_file = os.path.split(naip_path)[1]
    outpath = os.path.join("predictions", naip_file.replace(".tif", "_predictions.tif"))

    print(f"Saving {outpath}")

    with rasterio.open(outpath, "w", **profile) as outfile:

        for index in range(n_pixel_classes):

            # Note: rasterio band indices start with 1, not 0
            outfile.write(pixel_predictions[:, :, index].transpose(), index + 1)

    # TODO Put this in a separate function
    pixel_predictions_argmax = np.argmax(pixel_predictions, axis=-1).astype("uint8")

    profile["dtype"] = pixel_predictions_argmax.dtype
    profile["count"] = 1

    outpath = os.path.join(
        "predictions", naip_file.replace(".tif", "_predictions_argmax.tif")
    )

    print(f"Saving {outpath}")

    with rasterio.open(outpath, "w", **profile) as outfile:

        outfile.write(pixel_predictions_argmax.transpose(), 1)

        outfile.write_colormap(1, {k: v["rgb"] for k, v in colormap.items()})


def load_X_mean_and_std_train(model_name):

    infile_mean = model_name.replace(".h5", "_X_mean_train.npy")
    infile_std = model_name.replace(".h5", "_X_std_train.npy")

    return np.load(infile_mean), np.load(infile_std)


def main(model_name="./saved_models/cnn_land_cover_2019_11_10_05.h5"):

    config = get_config(MODEL_CONFIG)
    label_encoder, _ = get_label_encoder_and_mapping()

    model = load_model(model_name)

    X_mean_train, X_std_train = load_X_mean_and_std_train(model_name)

    colormap = get_colormap(label_encoder)

    for test_scene in config["test_scenes"]:

        predict_pixels_entire_scene(
            model,
            test_scene,
            X_mean_train,
            X_std_train,
            IMAGE_SHAPE,
            label_encoder,
            colormap,
        )


if __name__ == "__main__":
    main()
