import os

import numpy as np
import rasterio

from cnn import PIXELS, get_output_names
from constants import CDL_CLASS_BUILDING, CDL_CLASS_ROAD, CDL_CLASS_OTHER
from normalization import get_X_normalized


def get_colormap(label_encoder):

    building = label_encoder.transform([CDL_CLASS_BUILDING])[0]
    corn_soy = label_encoder.transform(["corn_soy"])[0]
    developed = label_encoder.transform(["developed"])[0]
    forest = label_encoder.transform(["forest"])[0]
    other = label_encoder.transform([CDL_CLASS_OTHER])[0]
    pasture = label_encoder.transform(["pasture"])[0]
    road = label_encoder.transform([CDL_CLASS_ROAD])[0]
    water = label_encoder.transform(["water"])[0]
    wetlands = label_encoder.transform(["wetlands"])[0]

    return {
        building: (102, 51, 0),
        corn_soy: (230, 180, 30),
        developed: (224, 224, 224),
        forest: (0, 102, 0),
        other: (255, 255, 255),
        pasture: (172, 226, 118),
        road: (128, 128, 128),
        water: (0, 102, 204),
        wetlands: (0, 153, 153),
    }


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
    pixel_predictions = np.zeros(X.shape[:2] + (n_pixel_classes,))

    # TODO Predict on rest of scene
    prediction_width = (image_shape[0] // 2) * (X_normalized.shape[0] // image_shape[0])
    prediction_height = (image_shape[1] // 2) * (
        X_normalized.shape[1] // image_shape[1]
    )

    X_normalized = X_normalized[
        np.newaxis, :prediction_width, :prediction_height, :
    ].copy()

    model_predictions = model.predict(X_normalized)

    output_names = get_output_names(model)
    pixels_output_index = np.where(np.array(output_names) == PIXELS)[0][0]

    pixel_predictions[:prediction_width, :prediction_height, :] = model_predictions[
        pixels_output_index
    ][0]

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

        outfile.write_colormap(1, colormap)
