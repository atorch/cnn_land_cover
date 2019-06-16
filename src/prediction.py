import os

import numpy as np
import rasterio

from cnn import N_PIXEL_CLASSES, PIXELS, get_output_names
from normalization import get_X_normalized


def predict_pixels_entire_scene(
    model, naip_path, X_mean_train, X_std_train, image_shape
):

    print(f"Predicting on {naip_path}")
    with rasterio.open(naip_path) as naip:

        X = naip.read()
        profile = naip.profile

    # Swap shape from (band, height, width) to (width, height, band)
    X = np.swapaxes(X, 0, 2)
    X_normalized = get_X_normalized(X, X_mean_train, X_std_train)

    # Predictions have shape (width, height, n_classes)
    pixel_predictions = np.zeros(X.shape[:2] + (N_PIXEL_CLASSES,))

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
    profile["count"] = N_PIXEL_CLASSES

    naip_file = os.path.split(naip_path)[1]
    outpath = os.path.join("predictions", naip_file.replace(".tif", "_predictions.tif"))

    print(f"Saving {outpath}")

    with rasterio.open(outpath, "w", **profile) as outfile:

        for index in range(N_PIXEL_CLASSES):

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

    # TODO Colormap  # TODO other, road, forest, water
    colormap = {
        0: (255, 255, 255, 255),
        1: (96, 96, 96, 255),
        2: (0, 102, 0, 255),
        3: (0, 102, 204, 255),
    }

    with rasterio.open(outpath, "w", **profile) as outfile:

        outfile.write(pixel_predictions_argmax.transpose(), 1)

        outfile.write_colormap(1, colormap)
