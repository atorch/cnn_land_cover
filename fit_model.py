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

from annotate_naip_scenes import CDL_ANNOTATION_DIR, ROAD_ANNOTATION_DIR
from cnn import get_keras_model, HAS_ROADS, IS_MAJORITY_FOREST


CDL_DIR = "./cdl"
COUNTY_DIR = "./county"
NAIP_DIR = "./naip"

# TODO Make sure CDL year matches NAIP year
CDL_FILE = "2017_30m_cdls.img"
COUNTY_FILE = "tl_2018_us_county.shp"

CDL_ANNOTATION_PREFIX = "cdl_for_"

# Note: any CDL class absent from CDL_MAPPING_FILE is coded as CDL_CLASS_OTHER
CDL_MAPPING_FILE = "cdl_classes.yml"
CDL_CLASS_OTHER = "other"


def recode_cdl_values(cdl_values, cdl_mapping, label_encoder):

    encoded_other = label_encoder.transform([CDL_CLASS_OTHER])[0]
    cdl_recoded = np.full(cdl_values.shape, encoded_other)

    for cdl_class_string, cdl_class_ints in cdl_mapping.items():

        encoded_cdl_class = label_encoder.transform([cdl_class_string])[0]
        cdl_recoded[np.isin(cdl_values, cdl_class_ints)] = encoded_cdl_class

    return cdl_recoded


def get_random_crop(naip_values, cdl_values, image_shape, label_encoder):
    # Note: both values and image_shape are (x, y, band) after call to np.swapaxes
    x_start = np.random.choice(range(naip_values.shape[0] - image_shape[0]))
    y_start = np.random.choice(range(naip_values.shape[1] - image_shape[1]))

    x_end = x_start + image_shape[0]
    y_end = y_start + image_shape[1]

    naip_crop = naip_values[x_start:x_end, y_start:y_end, 0 : image_shape[2]]

    # Note: target variable is indicator for whether NAIP scene is >50% forest
    forest_code = label_encoder.transform(["forest"])[0]
    cdl_crop = int(
        np.mean(cdl_values[x_start:x_end, y_start:y_end] == forest_code) > 0.5
    )

    return naip_crop, cdl_crop


def generator(annotated_scenes, label_encoder, image_shape, batch_size=16):

    while True:

        batch_X = np.empty((batch_size,) + image_shape)
        batch_forest = np.empty((batch_size, 1), dtype=int)
        batch_roads = np.empty((batch_size, 1), dtype=int)

        scene_indices = np.random.choice(range(len(annotated_scenes)), size=batch_size)

        for batch_index, scene_index in enumerate(scene_indices):

            # Note: annotated scenes are tuple of (NAIP, CDL annotation, road annotation)
            batch_X[batch_index], batch_forest[batch_index] = get_random_crop(
                annotated_scenes[scene_index][0],
                annotated_scenes[scene_index][1],
                image_shape,
                label_encoder,
            )

        # Note: generator returns tuples of (inputs, targets)
        yield (batch_X, {IS_MAJORITY_FOREST: batch_forest, HAS_ROADS: batch_roads})


def normalize_scenes(annotated_scenes):

    # TODO Use same mean and std when normalizing validation and test scenes (avoid leakage)!

    X_sizes = [X.size for X, *y in annotated_scenes]
    X_means = [X.mean() for X, *y in annotated_scenes]
    X_vars = [X.var() for X, *y in annotated_scenes]

    # Note: this produces the same result as
    #  np.hstack((X.flatten() for X, Y in annotated_scenes)).mean()
    # but uses less memory
    X_mean = np.average(X_means, weights=X_sizes)
    X_var = np.average(X_vars, weights=X_sizes)
    X_std = np.sqrt(X_var)

    for index, (X, *y) in enumerate(annotated_scenes):

        X_normalized = (X - X_mean) / X_std

        # Note: this modifies annotated_scenes in place
        annotated_scenes[index][0] = X_normalized.astype(np.float32)


def get_cdl_label_encoder_and_mapping():

    with open(CDL_MAPPING_FILE, "r") as infile:

        cdl_mapping = yaml.load(infile)

    cdl_label_encoder = LabelEncoder()

    cdl_classes = list(cdl_mapping.keys()) + [CDL_CLASS_OTHER]
    cdl_label_encoder.fit(cdl_classes)

    return cdl_label_encoder, cdl_mapping


def main(image_shape=(128, 128, 4)):

    naip_paths = glob.glob(os.path.join(NAIP_DIR, "m_*tif"))[:2]
    print(f"found {len(naip_paths)} naip scenes")

    annotated_scenes = []

    cdl_label_encoder, cdl_mapping = get_cdl_label_encoder_and_mapping()

    for naip_path in naip_paths:

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

    normalize_scenes(annotated_scenes)

    model = get_keras_model(image_shape)

    training_generator = generator(annotated_scenes, cdl_label_encoder, image_shape)
    sample_batch = next(training_generator)
    print(Counter(sample_batch[1].flatten().tolist()))
    print(sample_batch[0][0].shape)

    # TODO Validation generator should use different scenes from training generator
    validation_generator = generator(annotated_scenes, cdl_label_encoder, image_shape)

    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=16,
        epochs=20,
        verbose=True,
        callbacks=None,
        validation_data=validation_generator,
        validation_steps=4,
    )

    # TODO Test generator should use different scenes from training & validation generators
    test_generator = generator(
        annotated_scenes, cdl_label_encoder, image_shape, batch_size=128
    )
    test_X, test_y = next(test_generator)

    test_predictions = model.predict(test_X)
    test_predictions = (test_predictions > 0.5).astype(test_y.dtype)

    print(classification_report(test_predictions, test_y))


if __name__ == "__main__":
    main()
