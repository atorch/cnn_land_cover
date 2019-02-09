from collections import Counter
import glob
import os
import yaml

import fiona
import keras
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D
import numpy as np
import pyproj
import rasterio
from rasterio.windows import Window
from shapely.geometry import Polygon
from sklearn.preprocessing import LabelEncoder

CDL_DIR = './cdl'
COUNTY_DIR = './county'
NAIP_DIR = './naip'

# TODO Make sure CDL year matches NAIP year
CDL_FILE = '2017_30m_cdls.img'
COUNTY_FILE = 'tl_2018_us_county.shp'

CDL_ANNOTATION_PREFIX = 'cdl_for_'

# Note: any CDL class absent from CDL_MAPPING_FILE is coded as CDL_CLASS_OTHER
CDL_MAPPING_FILE = 'cdl_classes.yml'
CDL_CLASS_OTHER = 'other'

def get_y_x_at_pixel_centers(raster):
    yx_mesh = np.meshgrid(
        range(raster.meta['height']),
        range(raster.meta['width']),
        indexing='ij',
    )

    y_flat = yx_mesh[0].flatten()
    x_flat = yx_mesh[1].flatten()

    y_raster_proj = raster.meta['transform'].f + (y_flat + 0.5) * raster.meta['transform'].e
    x_raster_proj = raster.meta['transform'].c + (x_flat + 0.5) * raster.meta['transform'].a

    return y_raster_proj, x_raster_proj


def get_raster_values(raster, x_raster_proj, y_raster_proj):

    # raster_values = raster.sample(zip(x_raster_proj.tolist(), y_raster_proj.tolist()))
    # raster_values = [x[0] for x in raster_values]

    min_x = min(x_raster_proj)
    max_x = max(x_raster_proj)

    min_y = min(y_raster_proj)
    max_y = max(y_raster_proj)

    y_offset = (max_y - raster.meta['transform'].f) / raster.meta['transform'].e
    y_offset = np.floor(y_offset).astype(int)
    assert y_offset > 0

    height = (min_y - raster.meta['transform'].f) / raster.meta['transform'].e - y_offset
    height = np.ceil(height).astype(int)
    assert height > 0

    x_offset = (min_x - raster.meta['transform'].c) / raster.meta['transform'].a
    x_offset = np.floor(x_offset).astype(int)
    assert x_offset > 0

    width = (max_x - raster.meta['transform'].c) / raster.meta['transform'].a - x_offset
    width = np.ceil(width).astype(int)
    assert width > 0

    window = Window(x_offset, y_offset, width, height)

    # Note: shape is (band, y, x), i.e. (band, height, width)
    raster_window_values = raster.read(window=window)

    x_window_index = (x_raster_proj - raster.meta['transform'].c) / raster.meta['transform'].a - x_offset
    x_window_index = x_window_index.astype(int)

    y_window_index = (y_raster_proj - raster.meta['transform'].f) / raster.meta['transform'].e  - y_offset
    y_window_index = y_window_index.astype(int)

    return raster_window_values[0, y_window_index, x_window_index]

def save_cdl_values_for_naip_raster(x_cdl, y_cdl, cdl, naip_file, naip):

    cdl_values = get_raster_values(cdl, x_cdl, y_cdl)

    cdl_values = cdl_values.reshape((naip.meta['height'], naip.meta['width']))

    output_file = CDL_ANNOTATION_PREFIX + naip_file

    profile = naip.profile.copy()

    # Note: the output has the same width, height, and transform as the NAIP raster,
    # but contains a single band of CDL codes (whereas the NAIP raster contains 4 bands)
    profile['dtype'] = cdl.profile['dtype']
    profile['count'] = 1

    output_path = os.path.join(NAIP_DIR, output_file)

    print 'writing ' + output_file

    with rasterio.open(output_path, 'w', **profile) as output:
        output.write(cdl_values.astype(profile['dtype']), 1)


def get_counties(raster):
    # TODO Finish this function
    county_shp = fiona.open(os.path.join(COUNTY_DIR, COUNTY_FILE))

    raster_poly = Polygon()

    counties = []

    return counties

def preprocess_naip(naip_paths):

    cdl_path = os.path.join(CDL_DIR, CDL_FILE)
    cdl = rasterio.open(cdl_path)
    proj_cdl = pyproj.Proj(cdl.crs)

    for naip_path in naip_paths:

        print 'processing ' + naip_path

        naip = rasterio.open(naip_path)
        proj_naip = pyproj.Proj(naip.crs)

        # counties = get_counties(naip)

        y_naip, x_naip = get_y_x_at_pixel_centers(naip)

        x_cdl, y_cdl = pyproj.transform(proj_naip, proj_cdl, x_naip, y_naip)

        naip_file = os.path.split(naip_path)[-1]

        save_cdl_values_for_naip_raster(x_cdl, y_cdl, cdl, naip_file, naip)

def get_cdl_annotation_path_from_naip_path(naip_path):
    head, tail = os.path.split(naip_path)

    return os.path.join(head, CDL_ANNOTATION_PREFIX + tail)

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

    naip_crop = naip_values[x_start:x_end, y_start:y_end, 0:image_shape[2]]

    # Note: target variable is indicator for whether NAIP scene is >50% forest
    forest_code = label_encoder.transform(['forest'])[0]
    cdl_crop = int(np.mean(cdl_values[x_start:x_end, y_start:y_end] == forest_code) > 0.5)

    return naip_crop, cdl_crop


def generator(annotated_scenes, label_encoder, image_shape, batch_size=16):

    while True:

        batch_X = np.empty((batch_size, ) + image_shape)
        batch_Y = np.empty((batch_size, 1), dtype=int)  # TODO dtype, populate

        scene_indices = np.random.choice(range(len(annotated_scenes)), size=batch_size)

        for batch_index, scene_index in enumerate(scene_indices):

            # Note: annotated scenes are tuple of (NAIP, CDL annotation)
            batch_X[batch_index], batch_Y[batch_index] = get_random_crop(
                annotated_scenes[scene_index][0],
                annotated_scenes[scene_index][1],
                image_shape,
                label_encoder,
            )

        # Note: generator returns tuples of (inputs, targets)
        yield(batch_X, batch_Y)

def add_keras_model_block(input_layer):

    conv = Conv2D(32, kernel_size=3, padding="same", activation="relu")(input_layer)

    maxpool = MaxPooling2D()(conv)

    return BatchNormalization()(maxpool)

def get_keras_model(image_shape):
    input_layer = Input(shape=image_shape)

    current_last_layer = input_layer
    for _index in range(3):

        current_last_layer = add_keras_model_block(current_last_layer)

    flat = Flatten()(current_last_layer)
    final_layer = Dense(1, activation="sigmoid")(flat)

    model = Model(inputs=input_layer, outputs=final_layer)

    print(model.summary())

    nadam = keras.optimizers.Nadam()

    model.compile(
        optimizer=nadam,
        loss=keras.losses.binary_crossentropy,
        metrics=['accuracy'],
    )

    return model

def normalize_scenes(annotated_scenes):

    X_sizes = [X.size for X, y in annotated_scenes]
    X_means = [X.mean() for X, y in annotated_scenes]
    X_vars = [X.var() for X, y in annotated_scenes]

    # Note: this produces the same result as
    #  np.hstack((X.flatten() for X, Y in annotated_scenes)).mean()
    # but uses less memory
    X_mean = np.average(X_means, weights=X_sizes)
    X_var = np.average(X_vars, weights=X_sizes)
    X_std = np.sqrt(X_var)

    for index, X_y in enumerate(annotated_scenes):

        X, y = X_y
        X_normalized = (X - X_mean) / X_std

        # Note: this modifies annotated_scenes in place
        annotated_scenes[index] = (X_normalized.astype(np.float32), y)

def main(image_shape=(128, 128, 4)):

    naip_paths = glob.glob(os.path.join(NAIP_DIR, 'm_*tif'))
    print 'found ' + str(len(naip_paths)) + ' naip scenes'

    # preprocess_naip(naip_paths)

    annotated_scenes = []

    with open(CDL_MAPPING_FILE, 'r') as infile:

        cdl_mapping = yaml.load(infile)

    cdl_label_encoder = LabelEncoder()

    cdl_classes = cdl_mapping.keys() + [CDL_CLASS_OTHER]
    cdl_label_encoder.fit(cdl_classes)

    for naip_path in naip_paths:

        with rasterio.open(naip_path) as naip:

            X = naip.read()

        cdl_annotation_path = get_cdl_annotation_path_from_naip_path(naip_path)

        with rasterio.open(cdl_annotation_path) as naip_cdl:

            y_cdl = naip_cdl.read()

        # Note: shapes are (band, height, width)
        assert X.shape[1:] == y_cdl.shape[1:]

        y_cdl_recoded = recode_cdl_values(y_cdl, cdl_mapping, cdl_label_encoder)

        # Note: swap NAIP and CDL shape from (band, y, x) to (x, y, band)
        annotated_scenes.append(
            (np.swapaxes(X, 0, 2),
             np.swapaxes(y_cdl_recoded, 0, 2)),
        )

    normalize_scenes(annotated_scenes)

    model = get_keras_model(image_shape)

    my_generator = generator(annotated_scenes, cdl_label_encoder, image_shape)
    sample_batch = next(my_generator)
    print(Counter(sample_batch[1].flatten().tolist()))
    print(sample_batch[0][0].shape)

    # TODO validation, test
    model.fit_generator(
        generator=my_generator,
        steps_per_epoch=16,
        epochs=20,
        verbose=True,
        callbacks=None,
        validation_data=None,
    )

if __name__ == '__main__':
    main()
