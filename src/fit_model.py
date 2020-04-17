from collections import Counter
import datetime as dt
import json
import os

from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import Polygon
from sklearn.metrics import classification_report, confusion_matrix

from cnn import (
    get_keras_model,
    get_output_names,
    HAS_BUILDINGS,
    HAS_ROADS,
    IS_MAJORITY_FOREST,
    MODAL_LAND_COVER,
    PIXELS_RESHAPED,
)
from constants import (
    BUILDING_ANNOTATION_DIR,
    CDL_ANNOTATION_DIR,
    CDL_CLASSES_TO_MASK,
    IMAGE_SHAPE,
    MODEL_CONFIG,
    NAIP_DIR,
    ROAD_ANNOTATION_DIR,
    ROAD_ANNOTATION_FOR_MASK_DIR,
    SAVED_MODELS_DIR,
)
from generator import get_generator
from normalization import get_X_mean_and_std, get_X_normalized, normalize_scenes
from prediction import get_colormap, predict_pixels_entire_scene
from utils import get_config, get_label_encoder_and_mapping


def recode_cdl_values(cdl_values, cdl_mapping, label_encoder):

    encoded_other = label_encoder.transform(["other"])[0]

    # Note: preserve CDL dtype (uint8) to save memory
    cdl_recoded = np.full(cdl_values.shape, encoded_other, dtype=cdl_values.dtype)

    for cdl_class_string, cdl_class_ints in cdl_mapping.items():

        if not cdl_class_ints:
            continue

        encoded_cdl_class = label_encoder.transform([cdl_class_string])[0]
        cdl_recoded[np.isin(cdl_values, cdl_class_ints)] = encoded_cdl_class

    return cdl_recoded


def get_y_combined(y_cdl_recoded, y_road, y_road_for_mask, y_building, label_encoder):

    ## This function "burns" roads and buildings into the recoded CDL raster
    ## The road and building datasets appear to be more reliable than the CDL,
    ## and they are higher resolution, so roads and buildings take precedence over CDL codes

    y_combined = y_cdl_recoded.copy()

    road = label_encoder.transform(["road"])[0]
    y_combined[np.where(y_road)] = road

    # Note: this could in theory overwrite some roads
    building = label_encoder.transform(["building"])[0]
    y_combined[np.where(y_building)] = building

    # Note: this masks out pixels that are (1) CDL developed,
    # (2) not roads or buildings, and (3) close to a road (without being covered by a road)
    mask = label_encoder.transform(["mask"])[0]
    developed = label_encoder.transform(["developed"])[0]
    y_combined[
        np.where(np.logical_and(y_road_for_mask, y_combined == developed))
    ] = mask

    return y_combined


def get_annotated_scenes(naip_paths, label_encoder, cdl_mapping):

    annotated_scenes = []

    for naip_path in naip_paths:

        print(f"Reading {naip_path}")
        with rasterio.open(naip_path) as naip:

            X = naip.read()

        naip_file = os.path.split(naip_path)[1]

        cdl_annotation_path = os.path.join(CDL_ANNOTATION_DIR, naip_file)
        with rasterio.open(cdl_annotation_path) as cdl_annotation:

            y_cdl = cdl_annotation.read()

        # Note: shapes are (band, height, width)
        assert X.shape[1:] == y_cdl.shape[1:]

        y_cdl_recoded = recode_cdl_values(y_cdl, cdl_mapping, label_encoder)

        road_annotation_path = os.path.join(ROAD_ANNOTATION_DIR, naip_file)
        with rasterio.open(road_annotation_path) as road_annotation:

            y_road = road_annotation.read()

        road_annotation_for_mask_path = os.path.join(
            ROAD_ANNOTATION_FOR_MASK_DIR, naip_file
        )
        with rasterio.open(road_annotation_for_mask_path) as road_annotation_for_mask:

            y_road_for_mask = road_annotation_for_mask.read()

        building_annotation_path = os.path.join(BUILDING_ANNOTATION_DIR, naip_file)
        with rasterio.open(building_annotation_path) as building_annotation:
            y_building = building_annotation.read()

        y_combined = get_y_combined(
            y_cdl_recoded, y_road, y_road_for_mask, y_building, label_encoder
        )

        # Note: swap NAIP and CDL shape from (band, height, width) to (width, height, band)
        annotated_scenes.append([np.swapaxes(X, 0, 2), np.swapaxes(y_combined, 0, 2)])

    return annotated_scenes


def save_sample_images(sample_batch, X_mean_train, X_std_train, label_encoder):

    # TODO mkdir sample_images

    # Note: pixels are of shape (n_images, width, height, band)
    batch_X = sample_batch[0]

    for image_index in range(batch_X.shape[0]):

        # Note: pixels are normalized; get them back to integers in [0, 255] before saving, and use only RGB bands
        rgb_image = (
            batch_X[image_index, :, :, :3] * X_std_train[:, :, :3]
            + X_mean_train[:, :, :3]
        ).astype(int)

        outpath = f"./sample_images/sample_{str(image_index).rjust(2, '0')}.png"
        print(f"Saving {outpath}")
        plt.imsave(outpath, rgb_image)

        # Note: convert land cover from a one-hot vector back into a string (e.g. "forest" or "water")
        one_hot_vector = sample_batch[1][MODAL_LAND_COVER][image_index]
        land_cover = label_encoder.inverse_transform([one_hot_vector.argmax()])[0]

        labels = {MODAL_LAND_COVER: land_cover}

        for objective in [HAS_BUILDINGS, HAS_ROADS, IS_MAJORITY_FOREST]:
            labels[objective] = int(sample_batch[1][objective][image_index][0])

        outpath = outpath.replace(".png", ".txt")
        with open(outpath, "w") as outfile:
            outfile.write(json.dumps(labels))


def fit_model(config, label_encoder, cdl_mapping):

    training_scenes = get_annotated_scenes(
        config["training_scenes"], label_encoder, cdl_mapping
    )

    unique_training_images = sum(
        [
            (x[0].shape[0] // IMAGE_SHAPE[0]) * (x[0].shape[1] // IMAGE_SHAPE[1])
            for x in training_scenes
        ]
    )
    print(
        f"Done loading {len(training_scenes)} training scenes containing {unique_training_images} unique images of shape {IMAGE_SHAPE}"
    )

    validation_scenes = get_annotated_scenes(
        config["validation_scenes"], label_encoder, cdl_mapping
    )

    X_mean_train, X_std_train = get_X_mean_and_std(training_scenes)

    print(f"X_mean_train = {X_mean_train}")
    print(f"X_std_train = {X_std_train}")

    normalize_scenes(training_scenes, X_mean_train, X_std_train)
    normalize_scenes(validation_scenes, X_mean_train, X_std_train)

    model = get_keras_model(IMAGE_SHAPE, label_encoder)

    # plot_model(model, to_file='model.png')

    training_generator = get_generator(training_scenes, label_encoder, IMAGE_SHAPE)

    sample_batch = next(training_generator)

    for name, values in sample_batch[1].items():
        print(f"Sample batch of {name}: {Counter(values.flatten().tolist())}")

    print(f"Shape of sample batch X: {sample_batch[0][0].shape}")

    save_sample_images(sample_batch, X_mean_train, X_std_train, label_encoder)

    validation_generator = get_generator(validation_scenes, label_encoder, IMAGE_SHAPE)

    # TODO Tensorboard
    history = model.fit(
        x=training_generator,
        steps_per_epoch=50,
        epochs=100,
        verbose=True,
        # TODO EarlyStopping val_loss is not available warning
        callbacks=[
            callbacks.EarlyStopping(
                patience=20, monitor="val_loss", restore_best_weights=True, verbose=True
            )
        ],
        validation_data=validation_generator,
        validation_steps=10,
    )

    return model, X_mean_train, X_std_train


def print_masked_classification_report(
    name, y_pred, y_true, cdl_indices_to_mask, label_encoder
):

    y_true_in_indices_to_mask = np.in1d(y_true, cdl_indices_to_mask)
    y_pred = y_pred[np.logical_not(y_true_in_indices_to_mask)]
    y_true = y_true[np.logical_not(y_true_in_indices_to_mask)]

    print(
        f"Classification report for {name} with masking (i.e. ignoring {CDL_CLASSES_TO_MASK}):"
    )
    print(
        classification_report(
            y_pred=y_pred,
            y_true=y_true,
            target_names=label_encoder.classes_,
            labels=range(len(label_encoder.classes_)),
        )
    )


def print_classification_reports(test_X, test_y, model, label_encoder):

    test_predictions = model.predict(test_X)

    output_names = get_output_names(model)

    cdl_indices_to_mask = label_encoder.transform(CDL_CLASSES_TO_MASK)

    for index, name in enumerate(output_names):

        print(f"Classification report for {name}:")
        if name == MODAL_LAND_COVER:

            y_pred = test_predictions[index].argmax(axis=-1)
            y_true = test_y[name].argmax(axis=-1)

            print(
                classification_report(
                    y_pred=y_pred,
                    y_true=y_true,
                    target_names=label_encoder.classes_,
                    labels=range(len(label_encoder.classes_)),
                )
            )

            print_masked_classification_report(
                name, y_pred, y_true, cdl_indices_to_mask, label_encoder
            )

        elif name == PIXELS_RESHAPED:

            y_pred = test_predictions[index].argmax(axis=-1).flatten()
            y_true = test_y[name].argmax(axis=-1).flatten()

            print(
                classification_report(
                    y_pred=y_pred,
                    y_true=y_true,
                    target_names=label_encoder.classes_,
                    labels=range(len(label_encoder.classes_)),
                )
            )

            print_masked_classification_report(
                name, y_pred, y_true, cdl_indices_to_mask, label_encoder
            )

            # TODO Not very helpful without class names (or normalization)
            print("Confusion matrix:")
            print(
                confusion_matrix(
                    y_pred=label_encoder.classes_[y_pred],
                    y_true=label_encoder.classes_[y_true],
                )
            )

        else:

            # Note: this assumes these are binary classification problems!
            test_predictions[index] = (test_predictions[index] > 0.5).astype(
                test_y[name].dtype
            )

            print(
                classification_report(
                    y_pred=test_predictions[index], y_true=test_y[name]
                )
            )


def get_model_name():

    datetime_now = dt.datetime.now().strftime("%Y_%m_%d_%H")

    return os.path.join(SAVED_MODELS_DIR, f"cnn_land_cover_{datetime_now}.h5")


def save_X_mean_and_std_train(X_mean_train, X_std_train, model_name):

    outfile_mean = model_name.replace(".h5", "_X_mean_train.npy")
    print(f"Saving X_mean_train to {outfile_mean}")
    np.save(outfile_mean, X_mean_train, allow_pickle=False)

    outfile_std = model_name.replace(".h5", "_X_std_train.npy")
    print(f"Saving X_std_train to {outfile_std}")
    np.save(outfile_std, X_std_train, allow_pickle=False)


def main():

    config = get_config(MODEL_CONFIG)
    label_encoder, cdl_mapping = get_label_encoder_and_mapping()

    model_name = get_model_name()
    model, X_mean_train, X_std_train = fit_model(config, label_encoder, cdl_mapping)

    print(f"Saving model to {model_name}")
    model.save(model_name)

    # TODO Also save label encoder?
    save_X_mean_and_std_train(X_mean_train, X_std_train, model_name)

    test_scenes = get_annotated_scenes(
        config["test_scenes"], label_encoder, cdl_mapping
    )
    normalize_scenes(test_scenes, X_mean_train, X_std_train)

    # Note: use a large batch size so that test set stats have a small standard error
    test_generator = get_generator(
        test_scenes, label_encoder, IMAGE_SHAPE, batch_size=600
    )
    test_X, test_y, test_weights = next(test_generator)

    # TODO Show test set loss for each objective
    # Also fit some simple baseline models (null model, regression
    #  tree that only sees average for each band in the image, nearest neighbors...),
    #  compute their test set loss and show on a plot with CNN test loss
    print_classification_reports(test_X, test_y, model, label_encoder)


if __name__ == "__main__":
    main()
