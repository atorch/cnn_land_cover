from collections import Counter
import json
import os
import yaml

import keras
from keras import callbacks
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import Polygon
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from annotate_naip_scenes import CDL_ANNOTATION_DIR, NAIP_DIR, ROAD_ANNOTATION_DIR
from cnn import (
    get_keras_model,
    get_output_names,
    HAS_ROADS,
    IS_MAJORITY_FOREST,
    MODAL_LAND_COVER,
    PIXELS,
    PIXEL_CLASSES,
    N_PIXEL_CLASSES,
)
from generator import get_generator
from normalization import get_X_mean_and_std, get_X_normalized, normalize_scenes
from prediction import predict_pixels_entire_scene


IMAGE_SHAPE = (256, 256, 4)


# Note: any CDL class absent from CDL_MAPPING_FILE is coded as CDL_CLASS_OTHER
CDL_MAPPING_FILE = "./config/cdl_classes.yml"
CDL_CLASS_OTHER = "other"

MODEL_CONFIG = "./config/model_config.yml"


def recode_cdl_values(cdl_values, cdl_mapping, label_encoder):

    encoded_other = label_encoder.transform([CDL_CLASS_OTHER])[0]

    # Note: preserve CDL dtype (uint8) to save memory
    cdl_recoded = np.full(cdl_values.shape, encoded_other, dtype=cdl_values.dtype)

    for cdl_class_string, cdl_class_ints in cdl_mapping.items():

        encoded_cdl_class = label_encoder.transform([cdl_class_string])[0]
        cdl_recoded[np.isin(cdl_values, cdl_class_ints)] = encoded_cdl_class

    return cdl_recoded


def get_cdl_label_encoder_and_mapping():

    with open(CDL_MAPPING_FILE, "r") as infile:

        cdl_mapping = yaml.safe_load(infile)

    cdl_label_encoder = LabelEncoder()

    # TODO Buggy if not all cdl_classes are present in the training set?
    cdl_classes = list(cdl_mapping.keys()) + [CDL_CLASS_OTHER]
    cdl_label_encoder.fit(cdl_classes)

    return cdl_label_encoder, cdl_mapping


def get_annotated_scenes(naip_paths, cdl_label_encoder, cdl_mapping):

    annotated_scenes = []

    for naip_path in naip_paths:

        print(f"reading {naip_path}")
        with rasterio.open(naip_path) as naip:

            X = naip.read()

        naip_file = os.path.split(naip_path)[1]
        cdl_annotation_path = os.path.join(CDL_ANNOTATION_DIR, naip_file)

        with rasterio.open(cdl_annotation_path) as naip_cdl:

            y_cdl = naip_cdl.read()

        # Note: shapes are (band, height, width)
        assert X.shape[1:] == y_cdl.shape[1:]

        y_cdl_recoded = recode_cdl_values(y_cdl, cdl_mapping, cdl_label_encoder)

        road_annotation_path = os.path.join(ROAD_ANNOTATION_DIR, naip_file)

        with rasterio.open(road_annotation_path) as naip_road:

            y_road = naip_road.read()

        # Note: swap NAIP and CDL shape from (band, height, width) to (width, height, band)
        annotated_scenes.append(
            [
                np.swapaxes(X, 0, 2),
                np.swapaxes(y_cdl_recoded, 0, 2),
                np.swapaxes(y_road, 0, 2),
            ]
        )

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

        for objective in [HAS_ROADS, IS_MAJORITY_FOREST]:
            labels[objective] = int(sample_batch[1][objective][image_index][0])

        outpath = outpath.replace(".png", ".txt")
        with open(outpath, "w") as outfile:
            outfile.write(json.dumps(labels))


def fit_model(config, cdl_label_encoder, cdl_mapping):

    training_scenes = get_annotated_scenes(
        config["training_scenes"], cdl_label_encoder, cdl_mapping
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
        config["validation_scenes"], cdl_label_encoder, cdl_mapping
    )

    X_mean_train, X_std_train = get_X_mean_and_std(training_scenes)

    print(f"X_mean_train = {X_mean_train}")
    print(f"X_std_train = {X_std_train}")

    normalize_scenes(training_scenes, X_mean_train, X_std_train)
    normalize_scenes(validation_scenes, X_mean_train, X_std_train)

    model = get_keras_model(IMAGE_SHAPE, len(cdl_label_encoder.classes_))

    # plot_model(model, to_file='model.png')

    training_generator = get_generator(training_scenes, cdl_label_encoder, IMAGE_SHAPE)

    sample_batch = next(training_generator)

    for name, values in sample_batch[1].items():
        print(f"Sample batch of {name}: {Counter(values.flatten().tolist())}")

    print(f"Shape of sample batch X: {sample_batch[0][0].shape}")

    save_sample_images(sample_batch, X_mean_train, X_std_train, cdl_label_encoder)

    validation_generator = get_generator(
        validation_scenes, cdl_label_encoder, IMAGE_SHAPE
    )

    # TODO Tensorboard
    history = model.fit_generator(
        generator=training_generator,
        steps_per_epoch=50,
        epochs=100,
        verbose=True,
        callbacks=[
            callbacks.EarlyStopping(
                patience=20, monitor="val_loss", restore_best_weights=True
            )
        ],
        validation_data=validation_generator,
        validation_steps=10,
    )

    return model, X_mean_train, X_std_train


def get_config(model_config):

    with open(model_config, "r") as infile:

        config = yaml.safe_load(infile)

    assert len(set(config["training_scenes"])) == len(config["training_scenes"])
    assert len(set(config["validation_scenes"])) == len(config["validation_scenes"])
    assert len(set(config["test_scenes"])) == len(config["test_scenes"])

    assert (
        len(set(config["training_scenes"]).intersection(config["validation_scenes"]))
        == 0
    )
    assert (
        len(set(config["test_scenes"]).intersection(config["validation_scenes"])) == 0
    )

    # TODO Also assert that training scenes don't intersect test or validation scenes
    # NAIP scenes can overlap by a few hundred meters

    return config


def print_classification_reports(test_X, test_y, model, cdl_label_encoder):

    test_predictions = model.predict(test_X)

    output_names = get_output_names(model)

    for index, name in enumerate(output_names):

        print(f"Classification report for {name}:")
        if name == MODAL_LAND_COVER:

            print(
                classification_report(
                    y_pred=test_predictions[index].argmax(axis=-1),
                    y_true=test_y[name].argmax(axis=-1),
                    target_names=cdl_label_encoder.classes_,
                    labels=range(len(cdl_label_encoder.classes_)),
                )
            )

        elif name == PIXELS:

            print(
                classification_report(
                    y_pred=test_predictions[index].argmax(axis=-1).flatten(),
                    y_true=test_y[name].argmax(axis=-1).flatten(),
                    target_names=PIXEL_CLASSES,
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


def main():

    config = get_config(MODEL_CONFIG)

    cdl_label_encoder, cdl_mapping = get_cdl_label_encoder_and_mapping()

    model, X_mean_train, X_std_train = fit_model(config, cdl_label_encoder, cdl_mapping)

    # TODO Filename  # TODO Also save X_{mean,std}_train
    model.save("my_model.h5")

    test_scenes = get_annotated_scenes(
        config["test_scenes"], cdl_label_encoder, cdl_mapping
    )
    normalize_scenes(test_scenes, X_mean_train, X_std_train)

    test_generator = get_generator(
        test_scenes, cdl_label_encoder, IMAGE_SHAPE, batch_size=600
    )
    test_X, test_y = next(test_generator)

    print_classification_reports(test_X, test_y, model, cdl_label_encoder)

    for test_scene in config["test_scenes"]:

        predict_pixels_entire_scene(
            model, test_scene, X_mean_train, X_std_train, IMAGE_SHAPE
        )


if __name__ == "__main__":
    main()
