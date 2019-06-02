from collections import Counter
import glob
import json
import os
import yaml

import fiona
from keras import callbacks
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import stats
from shapely.geometry import Polygon
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from annotate_naip_scenes import CDL_ANNOTATION_DIR, NAIP_DIR, ROAD_ANNOTATION_DIR
from cnn import get_keras_model, HAS_ROADS, IS_MAJORITY_FOREST, MODAL_LAND_COVER


# Note: any CDL class absent from CDL_MAPPING_FILE is coded as CDL_CLASS_OTHER
CDL_MAPPING_FILE = "./cdl/cdl_classes.yml"
CDL_CLASS_OTHER = "other"


def recode_cdl_values(cdl_values, cdl_mapping, label_encoder):

    encoded_other = label_encoder.transform([CDL_CLASS_OTHER])[0]
    cdl_recoded = np.full(cdl_values.shape, encoded_other)

    for cdl_class_string, cdl_class_ints in cdl_mapping.items():

        encoded_cdl_class = label_encoder.transform([cdl_class_string])[0]
        cdl_recoded[np.isin(cdl_values, cdl_class_ints)] = encoded_cdl_class

    return cdl_recoded


def get_random_window(annotated_scene, image_shape, label_encoder):

    naip_values, cdl_values, road_values = annotated_scene

    # Note: both values and image_shape are (x, y, band) after call to np.swapaxes
    x_start = np.random.choice(range(naip_values.shape[0] - image_shape[0]))
    y_start = np.random.choice(range(naip_values.shape[1] - image_shape[1]))

    x_end = x_start + image_shape[0]
    y_end = y_start + image_shape[1]

    naip_window = naip_values[x_start:x_end, y_start:y_end, 0 : image_shape[2]]

    forest = label_encoder.transform(["forest"])[0]
    is_majority_forest = int(
        np.mean(cdl_values[x_start:x_end, y_start:y_end] == forest) > 0.5
    )

    # TODO This fraction should change if roads are buffered
    has_roads = int(np.mean(road_values[x_start:x_end, y_start:y_end]) > 0.001)

    modal_land_cover = stats.mode(cdl_values[x_start:x_end, y_start:y_end], axis=None).mode[0]

    return naip_window, is_majority_forest, has_roads, modal_land_cover


def generator(annotated_scenes, label_encoder, image_shape, batch_size=20):

    while True:

        batch_X = np.empty((batch_size,) + image_shape)
        batch_forest = np.empty((batch_size, 1), dtype=int)
        batch_roads = np.empty((batch_size, 1), dtype=int)
        batch_land_cover = np.zeros((batch_size, len(label_encoder.classes_)), dtype=int)

        scene_indices = np.random.choice(range(len(annotated_scenes)), size=batch_size)

        for batch_index, scene_index in enumerate(scene_indices):

            # Note: annotated scenes are tuple of (NAIP, CDL annotation, road annotation)
            window = get_random_window(
                annotated_scenes[scene_index], image_shape, label_encoder
            )

            batch_X[batch_index], batch_forest[batch_index], batch_roads[batch_index], land_cover = window

            # Note: this one-hot encodes land cover
            batch_land_cover[batch_index, land_cover] = 1

        # Note: generator returns tuples of (inputs, targets)
        yield (batch_X, {IS_MAJORITY_FOREST: batch_forest, HAS_ROADS: batch_roads, MODAL_LAND_COVER: batch_land_cover})


def get_X_mean_and_std(annotated_scenes):

    # Note: annotated scene shapres are (x, y, band)

    X_sizes = [X.size for X, *y in annotated_scenes]
    X_means = [X.mean(axis=(0, 1)) for X, *y in annotated_scenes]
    X_vars = [X.var(axis=(0, 1)) for X, *y in annotated_scenes]

    # Note: this produces the same result as
    #  np.hstack((X.flatten() for X, Y in annotated_scenes)).mean()
    # but uses less memory
    X_mean = np.average(X_means, weights=X_sizes, axis=0)
    X_var = np.average(X_vars, weights=X_sizes, axis=0)
    X_std = np.sqrt(X_var)

    X_mean = X_mean.reshape((1, 1, X_mean.size))
    X_std = X_std.reshape((1, 1, X_std.size))

    return X_mean, X_std


def normalize_scenes(annotated_scenes, X_mean, X_std):

    # Note: use training mean and std when normalizing validation and test scenes (avoid leakage)!

    for index, (X, *y) in enumerate(annotated_scenes):

        X_normalized = (X - X_mean) / X_std

        # Note: this modifies annotated_scenes in place
        annotated_scenes[index][0] = X_normalized.astype(np.float32)


def get_cdl_label_encoder_and_mapping():

    with open(CDL_MAPPING_FILE, "r") as infile:

        cdl_mapping = yaml.safe_load(infile)

    cdl_label_encoder = LabelEncoder()

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

        # Note: swap NAIP and CDL shape from (band, y, x) to (x, y, band)
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
        rgb_image = (batch_X[image_index, :, :, :3] * X_std_train[:, :, :3] + X_mean_train[:, :, :3]).astype(int)

        outfile = f"./sample_images/sample_{str(image_index).rjust(2, '0')}.png"
        print(f"Saving {outfile}")
        plt.imsave(outfile, rgb_image)

        # Note: convert land cover from a one-hot vector back into a string (e.g. "forest" or "water")
        one_hot_vector = sample_batch[1][MODAL_LAND_COVER][image_index]
        land_cover = label_encoder.inverse_transform([one_hot_vector.argmax()])[0]

        labels = {MODAL_LAND_COVER: land_cover}

        for objective in [HAS_ROADS, IS_MAJORITY_FOREST]:
            labels[objective] = int(sample_batch[1][objective][image_index][0])

        outfile = outfile.replace(".png", ".txt")
        with open(outfile, "w") as f:
            f.write(json.dumps(labels))


def fit_model(config, cdl_label_encoder, cdl_mapping, image_shape):

    training_scenes = get_annotated_scenes(
        config["training_scenes"], cdl_label_encoder, cdl_mapping
    )
    validation_scenes = get_annotated_scenes(
        config["validation_scenes"], cdl_label_encoder, cdl_mapping
    )

    X_mean_train, X_std_train = get_X_mean_and_std(training_scenes)

    print(f"X_mean_train = {X_mean_train}")
    print(f"X_std_train = {X_std_train}")

    normalize_scenes(training_scenes, X_mean_train, X_std_train)
    normalize_scenes(validation_scenes, X_mean_train, X_std_train)

    model = get_keras_model(image_shape, len(cdl_label_encoder.classes_))

    training_generator = generator(training_scenes, cdl_label_encoder, image_shape)

    sample_batch = next(training_generator)

    for name, values in sample_batch[1].items():
        print(f"Sample batch of {name}: {Counter(values.flatten().tolist())}")

    print(f"Shape of sample batch X: {sample_batch[0][0].shape}")

    save_sample_images(sample_batch, X_mean_train, X_std_train, cdl_label_encoder)

    validation_generator = generator(validation_scenes, cdl_label_encoder, image_shape)

    # TODO Tensorboard
    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=50,
        epochs=100,
        verbose=True,
        callbacks=[callbacks.EarlyStopping(patience=20, monitor="val_loss", restore_best_weights=True)],
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

    assert len(set(config["training_scenes"]).intersection(config["validation_scenes"])) == 0
    assert len(set(config["test_scenes"]).intersection(config["validation_scenes"])) == 0

    # TODO Also assert that training scenes don't intersect test or validation scenes
    # NAIP scenes can overlap by a few hundred meters

    return config

def main(image_shape=(256, 256, 4), model_config="config.yml"):

    config = get_config(model_config)

    cdl_label_encoder, cdl_mapping = get_cdl_label_encoder_and_mapping()

    model, X_mean_train, X_std_train = fit_model(
        config, cdl_label_encoder, cdl_mapping, image_shape
    )

    test_scenes = get_annotated_scenes(config["test_scenes"], cdl_label_encoder, cdl_mapping)
    normalize_scenes(test_scenes, X_mean_train, X_std_train)

    test_generator = generator(
        test_scenes, cdl_label_encoder, image_shape, batch_size=500
    )
    test_X, test_y = next(test_generator)

    test_predictions = model.predict(test_X)

    # Note: example name is "is_majority_forest/Sigmoid", need the part before the /
    output_names = [x.op.name.split("/")[0] for x in model.outputs]

    for index, name in enumerate(output_names):

        test_predictions[index] = (test_predictions[index] > 0.5).astype(
            test_y[name].dtype
        )

        print(f"Classification report for {name}:")
        if name == MODAL_LAND_COVER:

            print(classification_report(test_predictions[index], test_y[name], target_names=cdl_label_encoder.classes_))

        else:

            print(classification_report(test_predictions[index], test_y[name]))


if __name__ == "__main__":
    main()
