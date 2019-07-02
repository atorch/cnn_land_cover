import numpy as np
from scipy import stats

from cnn import (
    HAS_BUILDINGS,
    HAS_ROADS,
    IS_MAJORITY_FOREST,
    MODAL_LAND_COVER,
    PIXELS,
    PIXEL_CLASSES,
    N_PIXEL_CLASSES,
)


# TODO This is essentially a label encoder for PIXEL_CLASSES, just use that
PIXEL_INDEX = {
    pixel_class: np.where(np.array(PIXEL_CLASSES) == pixel_class)[0][0]
    for pixel_class in PIXEL_CLASSES
}


def get_generator(annotated_scenes, label_encoder, image_shape, batch_size=20):

    # TODO Better sampling scheme than uniform, compare accuracy
    # Try distribution with a higher probability of seeing entire NAIP scene (uniform will overlap)
    # TODO Augmentation (flips)? Worth the extra computation?

    while True:

        batch_X = np.empty((batch_size,) + image_shape)
        batch_forest = np.empty((batch_size, 1), dtype=int)
        batch_has_buildings = np.empty((batch_size, 1), dtype=int)
        batch_has_roads = np.empty((batch_size, 1), dtype=int)
        batch_pixels = np.empty(
            (batch_size,) + image_shape[:2] + (N_PIXEL_CLASSES,), dtype=int
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


def get_one_hot_encoded_pixels(image_shape, road_patch, cdl_patch, label_encoder):

    # TODO One-hot encode pixels such that pixels.sum(axis=2) is 1 everywhere -- put this in a function, clean it up
    pixels = np.zeros(image_shape[:2] + (N_PIXEL_CLASSES,))

    forest = label_encoder.transform(["forest"])[0]
    corn_soy = label_encoder.transform(["corn_soy"])[0]
    water = label_encoder.transform(["water"])[0]

    pixels[:, :, PIXEL_INDEX["road"]][np.where(road_patch[:, :, 0])] = 1
    pixels[:, :, PIXEL_INDEX["forest"]][
        np.where(
            np.logical_and(
                cdl_patch[:, :, 0] == forest, np.logical_not(pixels[:, :, 1])
            )
        )
    ] = 1
    pixels[:, :, PIXEL_INDEX["corn_soy"]][
        np.where(
            np.logical_and(
                cdl_patch[:, :, 0] == corn_soy, np.logical_not(pixels[:, :, 1])
            )
        )
    ] = 1
    pixels[:, :, PIXEL_INDEX["water"]][
        np.where(
            np.logical_and(cdl_patch[:, :, 0] == water, np.logical_not(pixels[:, :, 1]))
        )
    ] = 1

    # Note: pixels that are not in {roads, forest, corn_soy, water} are coded as other
    pixels[:, :, PIXEL_INDEX["other"]][np.where(pixels.sum(axis=2) == 0)] = 1

    return pixels


def get_random_patch(annotated_scene, image_shape, label_encoder):

    naip_values, cdl_values, road_values, building_values = annotated_scene

    # Note: both values and image_shape are (x, y, band) after call to np.swapaxes
    x_start = np.random.choice(range(naip_values.shape[0] - image_shape[0]))
    y_start = np.random.choice(range(naip_values.shape[1] - image_shape[1]))

    x_end = x_start + image_shape[0]
    y_end = y_start + image_shape[1]

    naip_patch = naip_values[x_start:x_end, y_start:y_end, 0 : image_shape[2]]

    forest = label_encoder.transform(["forest"])[0]
    is_majority_forest = int(
        np.mean(cdl_values[x_start:x_end, y_start:y_end] == forest) > 0.5
    )

    road_patch = road_values[x_start:x_end, y_start:y_end]
    has_roads = int(np.mean(road_patch) > 0.001)

    building_patch = building_values[x_start:x_end, y_start:y_end]
    has_buildings = int(np.mean(building_patch) > 0.0001)

    cdl_patch = cdl_values[x_start:x_end, y_start:y_end]
    modal_land_cover = stats.mode(cdl_patch, axis=None).mode[0]

    pixels = get_one_hot_encoded_pixels(
        image_shape, road_patch, cdl_patch, label_encoder
    )

    labels = {
        HAS_BUILDINGS: has_buildings,
        HAS_ROADS: has_roads,
        IS_MAJORITY_FOREST: is_majority_forest,
        MODAL_LAND_COVER: modal_land_cover,
        PIXELS: pixels,
    }

    return naip_patch, labels
