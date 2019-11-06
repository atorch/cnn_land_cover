from functools import partial

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
)

from constants import (
    CDL_CLASSES_TO_MASK,
    HAS_BUILDINGS,
    HAS_ROADS,
    PIXELS,
    IS_MAJORITY_FOREST,
    MODAL_LAND_COVER,
)


# TODO Tune
N_BLOCKS = 6
BASE_N_FILTERS = 32
ADDITIONAL_FILTERS_PER_BLOCK = 16
DROPOUT_RATE = 0.15


def add_downsampling_block(input_layer, block_index):

    n_filters = BASE_N_FILTERS + ADDITIONAL_FILTERS_PER_BLOCK * block_index

    conv1 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(
        input_layer
    )
    dropout = Dropout(rate=DROPOUT_RATE)(conv1)
    conv2 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(dropout)

    batchnorm = BatchNormalization()(conv2)

    # Note: Don't MaxPool in last downsampling block
    if block_index == N_BLOCKS - 1:

        return batchnorm, conv2

    return MaxPooling2D()(batchnorm), conv2


def add_upsampling_block(input_layer, block_index, downsampling_conv2_layers):

    n_filters = BASE_N_FILTERS + ADDITIONAL_FILTERS_PER_BLOCK * block_index

    upsample = UpSampling2D()(input_layer)

    concat = concatenate([upsample, downsampling_conv2_layers[block_index - 1]])

    conv1 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(concat)
    dropout = Dropout(rate=DROPOUT_RATE)(conv1)
    conv2 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(dropout)

    return BatchNormalization()(conv2)


def get_masked_categorical_crossentropy(cdl_indices_to_mask):
    def masked_categorical_crossentropy(y_true, y_pred):

        # Used for pixel classifications: mask pixels whose class is in cdl_indices_to_mask
        # Note: shapes are (batch, image_shape, image_shape, n_classes)
        # TODO Speed this up, test it

        mask = K.ones_like(y_true[:, :, :, 0])

        for cdl_index in cdl_indices_to_mask:
            cdl_index_indicators = K.cast(
                K.equal(y_true[:, :, :, cdl_index], 1), "float32"
            )
            mask -= cdl_index_indicators

        return K.categorical_crossentropy(y_true, y_pred) * mask

    return masked_categorical_crossentropy


def get_keras_model(image_shape, label_encoder):

    # Note: model is fully convolutional, so image width and height can be arbitrary
    input_layer = Input(shape=(None, None, image_shape[2]))

    # Note: Keep track of conv2 layers so that they can be connected to the upsampling blocks
    downsampling_conv2_layers = []

    current_last_layer = input_layer
    for index in range(N_BLOCKS):

        current_last_layer, conv2_layer = add_downsampling_block(
            current_last_layer, index
        )

        downsampling_conv2_layers.append(conv2_layer)

    maxpool = GlobalAveragePooling2D()(current_last_layer)

    n_classes = len(label_encoder.classes_)

    output_buildings = Dense(1, activation="sigmoid", name=HAS_BUILDINGS)(maxpool)
    output_forest = Dense(1, activation="sigmoid", name=IS_MAJORITY_FOREST)(maxpool)
    output_roads = Dense(1, activation="sigmoid", name=HAS_ROADS)(maxpool)
    output_land_cover = Dense(n_classes, activation="softmax", name=MODAL_LAND_COVER)(
        maxpool
    )

    for index in range(N_BLOCKS - 1, 0, -1):

        current_last_layer = add_upsampling_block(
            current_last_layer, index, downsampling_conv2_layers
        )

    output_pixels = Conv2D(n_classes, 1, activation="softmax", name=PIXELS)(
        current_last_layer
    )

    model = Model(
        inputs=input_layer,
        outputs=[
            output_buildings,
            output_forest,
            output_roads,
            output_land_cover,
            output_pixels,
        ],
    )

    print(model.summary())

    nadam = keras.optimizers.Nadam()

    cdl_indices_to_mask = label_encoder.transform(CDL_CLASSES_TO_MASK)
    masked_categorical_crossentropy = get_masked_categorical_crossentropy(
        cdl_indices_to_mask
    )

    model.compile(
        optimizer=nadam,
        loss={
            HAS_BUILDINGS: keras.losses.binary_crossentropy,
            HAS_ROADS: keras.losses.binary_crossentropy,
            IS_MAJORITY_FOREST: keras.losses.binary_crossentropy,
            MODAL_LAND_COVER: keras.losses.categorical_crossentropy,
            PIXELS: masked_categorical_crossentropy,
        },
        metrics=["accuracy"],
    )

    return model


def get_output_names(model):

    # Note: example name is "is_majority_forest/Sigmoid" -- get the part before the "/"
    # TODO Why do names have an extra "_1" suffix after model is saved (but not before?)
    return [x.op.name.split("/")[0].replace("_1", "") for x in model.outputs]
