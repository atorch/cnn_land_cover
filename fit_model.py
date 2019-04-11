from collections import Counter
import glob
import os
import yaml

import fiona
import numpy as np
import rasterio
from shapely.geometry import Polygon
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from annotate_naip_scenes import CDL_ANNOTATION_DIR, NAIP_DIR, ROAD_ANNOTATION_DIR
from cnn import get_keras_model, HAS_ROADS, IS_MAJORITY_FOREST


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

    # Note: this target variable is an indicator for whether the window is >50% forest
    forest_code = label_encoder.transform(["forest"])[0]
    cdl_window = int(
        np.mean(cdl_values[x_start:x_end, y_start:y_end] == forest_code) > 0.5
    )

    # Note: this target variable is an indivator for whether the window is >(1/1000) roads
    road_window = int(np.mean(road_values[x_start:x_end, y_start:y_end]) > 0.001)

    return naip_window, cdl_window, road_window


def generator(annotated_scenes, label_encoder, image_shape, batch_size=16):

    while True:

        X = np.empty((batch_size,) + image_shape)
        forest = np.empty((batch_size, 1), dtype=int)
        roads = np.empty((batch_size, 1), dtype=int)

        scene_indices = np.random.choice(range(len(annotated_scenes)), size=batch_size)

        for batch_index, scene_index in enumerate(scene_indices):

            # Note: annotated scenes are tuple of (NAIP, CDL annotation, road annotation)
            window = get_random_window(
                annotated_scenes[scene_index], image_shape, label_encoder
            )

            X[batch_index], forest[batch_index], roads[batch_index] = window

        # Note: generator returns tuples of (inputs, targets)
        yield (X, {IS_MAJORITY_FOREST: forest, HAS_ROADS: roads})


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


def main(image_shape=(128, 128, 4)):

    cdl_label_encoder, cdl_mapping = get_cdl_label_encoder_and_mapping()

    naip_paths = glob.glob(os.path.join(NAIP_DIR, "m_*tif"))
    print(f"found {len(naip_paths)} naip scenes")

    # TODO random train/val/test split
    training_scenes = get_annotated_scenes(
        naip_paths[:2], cdl_label_encoder, cdl_mapping
    )
    validation_scenes = get_annotated_scenes(
        naip_paths[2:4], cdl_label_encoder, cdl_mapping
    )

    X_mean_train, X_std_train = get_X_mean_and_std(training_scenes)

    print(f"X_mean_train = {X_mean_train}")
    print(f"X_std_train = {X_std_train}")

    normalize_scenes(training_scenes, X_mean_train, X_std_train)
    normalize_scenes(validation_scenes, X_mean_train, X_std_train)

    model = get_keras_model(image_shape)

    training_generator = generator(training_scenes, cdl_label_encoder, image_shape)
    sample_batch = next(training_generator)

    for name, values in sample_batch[1].items():
        print(f"Sample batch of {name}: {Counter(values.flatten().tolist())}")

    print(sample_batch[0][0].shape)

    validation_generator = generator(validation_scenes, cdl_label_encoder, image_shape)

    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=16,
        epochs=20,
        verbose=True,
        callbacks=None,
        validation_data=validation_generator,
        validation_steps=4,
    )

    test_scenes = get_annotated_scenes(naip_paths[4:6], cdl_label_encoder, cdl_mapping)
    normalize_scenes(test_scenes, X_mean_train, X_std_train)

    test_generator = generator(
        test_scenes, cdl_label_encoder, image_shape, batch_size=128
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
        print(classification_report(test_predictions[index], test_y[name]))


if __name__ == "__main__":
    main()
