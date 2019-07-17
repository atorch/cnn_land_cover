import numpy as np
from keras.utils.np_utils import to_categorical
from scipy import stats

from cnn import HAS_BUILDINGS, HAS_ROADS, IS_MAJORITY_FOREST, MODAL_LAND_COVER, PIXELS

from constants import (
    CDL_CLASS_BUILDING,
    CDL_CLASS_ROAD,
    HAS_BUILDINGS,
    HAS_ROADS,
    PIXELS,
    IS_MAJORITY_FOREST,
    MODAL_LAND_COVER,
)


def get_generator(annotated_scenes, label_encoder, image_shape, batch_size=20):

    # TODO Better sampling scheme than uniform, compare accuracy
    # Try distribution with a higher probability of seeing entire NAIP scene (uniform will overlap)
    # TODO Augmentation (flips)? Worth the extra computation?

    n_pixel_classes = len(label_encoder.classes_)

    while True:

        batch_X = np.empty((batch_size,) + image_shape)
        batch_forest = np.empty((batch_size, 1), dtype=int)
        batch_has_buildings = np.empty((batch_size, 1), dtype=int)
        batch_has_roads = np.empty((batch_size, 1), dtype=int)
        batch_pixels = np.empty(
            (batch_size,) + image_shape[:2] + (n_pixel_classes,), dtype=int
        )
        batch_land_cover = np.zeros(
            (batch_size, len(label_encoder.classes_)), dtype=int
        )

        scene_indices = np.random.choice(range(len(annotated_scenes)), size=batch_size)

        for batch_index, scene_index in enumerate(scene_indices):

            # Note: annotated scenes are tuple of (NAIP, CDL annotation, road annotation)
            batch_X[batch_index], labels = get_random_patch(
                annotated_scenes[scene_index], image_shape, label_encoder
            )

            batch_forest[batch_index] = labels[IS_MAJORITY_FOREST]
            batch_has_buildings[batch_index] = labels[HAS_BUILDINGS]
            batch_has_roads[batch_index] = labels[HAS_ROADS]
            batch_pixels[batch_index] = labels[PIXELS]

            # Note: this one-hot encodes land cover (array was initialized to np.zeros)
            land_cover = labels[MODAL_LAND_COVER]
            batch_land_cover[batch_index, land_cover] = 1

        # Note: generator returns tuples of (inputs, targets)
        yield (
            batch_X,
            {
                HAS_BUILDINGS: batch_has_buildings,
                HAS_ROADS: batch_has_roads,
                IS_MAJORITY_FOREST: batch_forest,
                MODAL_LAND_COVER: batch_land_cover,
                PIXELS: batch_pixels,
            },
        )


def get_one_hot_encoded_pixels(image_shape, y_patch, label_encoder):

    return to_categorical(y_patch, num_classes=len(label_encoder.classes_)).astype(int)


def get_random_patch(annotated_scene, image_shape, label_encoder):

    naip_values, y_values = annotated_scene

    # Note: both values and image_shape are (x, y, band) after call to np.swapaxes
    x_start = np.random.choice(range(naip_values.shape[0] - image_shape[0]))
    y_start = np.random.choice(range(naip_values.shape[1] - image_shape[1]))

    x_end = x_start + image_shape[0]
    y_end = y_start + image_shape[1]

    naip_patch = naip_values[x_start:x_end, y_start:y_end, 0 : image_shape[2]]
    y_patch = y_values[x_start:x_end, y_start:y_end]

    forest = label_encoder.transform(["forest"])[0]
    is_majority_forest = int(np.mean(y_patch == forest) > 0.5)

    road = label_encoder.transform([CDL_CLASS_ROAD])[0]
    has_roads = int(np.mean(y_patch == road) > 0.001)

    building = label_encoder.transform([CDL_CLASS_BUILDING])[0]
    has_buildings = int(np.mean(y_patch == building) > 0.0001)

    modal_land_cover = stats.mode(y_patch, axis=None).mode[0]

    pixels = get_one_hot_encoded_pixels(image_shape, y_patch, label_encoder)

    labels = {
        HAS_BUILDINGS: has_buildings,
        HAS_ROADS: has_roads,
        IS_MAJORITY_FOREST: is_majority_forest,
        MODAL_LAND_COVER: modal_land_cover,
        PIXELS: pixels,
    }

    return naip_patch, labels
