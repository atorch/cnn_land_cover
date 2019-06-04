import keras
from keras.models import Model
from keras.layers import (
    BatchNormalization,
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
ROAD_PIXELS = "road_pixels"

BASE_N_FILTERS = 16
ADDITIONAL_FILTERS_PER_BLOCK = 16


def add_downsampling_block(input_layer, block_index):

    n_filters = BASE_N_FILTERS + ADDITIONAL_FILTERS_PER_BLOCK * block_index

    conv1 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(
        input_layer
    )
    conv2 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(conv1)

    batchnorm = BatchNormalization()(conv2)

    # TODO Don't MaxPool in last downsampling block; don't UpSample in first upsampling block
    return MaxPooling2D()(batchnorm)


def add_upsampling_block(input_layer, block_index):

    n_filters = BASE_N_FILTERS + ADDITIONAL_FILTERS_PER_BLOCK * block_index

    upsample = UpSampling2D()(input_layer)

    conv1 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(
        upsample
    )
    conv2 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(conv1)

    return BatchNormalization()(conv2)


def get_keras_model(image_shape, n_land_cover_classes, n_blocks=5):

    input_layer = Input(shape=image_shape)

    current_last_layer = input_layer
    for index in range(n_blocks):

        current_last_layer = add_downsampling_block(current_last_layer, index)

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

    for index in range(n_blocks - 1, -1, -1):

        current_last_layer = add_upsampling_block(current_last_layer, index)

    output_road_pixels = Conv2D(1, 1, activation="sigmoid", name=ROAD_PIXELS)(
        current_last_layer
    )

    model = Model(
        inputs=input_layer,
        outputs=[output_forest, output_roads, output_land_cover, output_road_pixels],
    )

    print(model.summary())

    nadam = keras.optimizers.Nadam()

    model.compile(
        optimizer=nadam,
        loss={
            HAS_ROADS: keras.losses.binary_crossentropy,
            IS_MAJORITY_FOREST: keras.losses.binary_crossentropy,
            MODAL_LAND_COVER: keras.losses.categorical_crossentropy,
            ROAD_PIXELS: keras.losses.binary_crossentropy,
        },
        metrics=["accuracy"],
    )

    return model
