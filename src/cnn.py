import keras
from keras.models import Model
from keras.layers import (
    BatchNormalization,
    concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    UpSampling2D,
)


HAS_ROADS = "has_roads"
IS_MAJORITY_FOREST = "is_majority_forest"
MODAL_LAND_COVER = "modal_land_cover"
PIXELS = "pixels"

# TODO Clean this up
# For now, pixel classes are road, forest, other
N_PIXEL_CLASSES = 3

BASE_N_FILTERS = 16
ADDITIONAL_FILTERS_PER_BLOCK = 16

N_BLOCKS = 5


def add_downsampling_block(input_layer, block_index, downsampling_conv2_layers):

    n_filters = BASE_N_FILTERS + ADDITIONAL_FILTERS_PER_BLOCK * block_index

    conv1 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(
        input_layer
    )
    conv2 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(conv1)

    downsampling_conv2_layers[block_index] = conv2

    batchnorm = BatchNormalization()(conv2)

    # Note: Don't MaxPool in last downsampling block
    if block_index == N_BLOCKS - 1:

        return batchnorm

    return MaxPooling2D()(batchnorm)


def add_upsampling_block(input_layer, block_index, downsampling_conv2_layers):

    n_filters = BASE_N_FILTERS + ADDITIONAL_FILTERS_PER_BLOCK * block_index

    upsample = UpSampling2D()(input_layer)

    concat = concatenate([upsample, downsampling_conv2_layers[block_index - 1]])

    conv1 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(
        concat
    )
    conv2 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(conv1)

    return BatchNormalization()(conv2)


def get_keras_model(image_shape, n_land_cover_classes):

    input_layer = Input(shape=image_shape)

    # Note: Keep track of conv2 layers so that they can be connected to the upsampling blocks
    downsampling_conv2_layers = {}

    current_last_layer = input_layer
    for index in range(N_BLOCKS):

        current_last_layer = add_downsampling_block(current_last_layer, index, downsampling_conv2_layers)

    flat = Flatten()(current_last_layer)

    dropout_1 = Dropout(rate=0.2)(flat)
    dense_1 = Dense(512, activation="relu")(dropout_1)

    dropout_2 = Dropout(rate=0.2)(dense_1)
    dense_2 = Dense(128, activation="relu")(dropout_2)

    output_forest = Dense(1, activation="sigmoid", name=IS_MAJORITY_FOREST)(dense_2)
    output_roads = Dense(1, activation="sigmoid", name=HAS_ROADS)(dense_2)
    output_land_cover = Dense(
        n_land_cover_classes, activation="softmax", name=MODAL_LAND_COVER
    )(dense_2)

    for index in range(N_BLOCKS - 1, 0, -1):

        current_last_layer = add_upsampling_block(current_last_layer, index, downsampling_conv2_layers)

    output_pixels = Conv2D(N_PIXEL_CLASSES, 1, activation="softmax", name=PIXELS)(
        current_last_layer
    )

    model = Model(
        inputs=input_layer,
        outputs=[output_forest, output_roads, output_land_cover, output_pixels],
    )

    print(model.summary())

    nadam = keras.optimizers.Nadam()

    model.compile(
        optimizer=nadam,
        loss={
            HAS_ROADS: keras.losses.binary_crossentropy,
            IS_MAJORITY_FOREST: keras.losses.binary_crossentropy,
            MODAL_LAND_COVER: keras.losses.categorical_crossentropy,
            PIXELS: keras.losses.categorical_crossentropy,
        },
        metrics=["accuracy"],
    )

    return model
