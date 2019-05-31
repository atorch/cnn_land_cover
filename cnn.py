import keras
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D


HAS_ROADS = "has_roads"
IS_MAJORITY_FOREST = "is_majority_forest"
MODAL_LAND_COVER = "modal_land_cover"


def add_keras_model_block(input_layer, index):

    n_filters = 16 * (index + 1)

    conv1 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(input_layer)
    conv2 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(conv1)

    batchnorm = BatchNormalization()(conv2)

    return MaxPooling2D()(batchnorm)


def get_keras_model(image_shape, n_land_cover_classes):

    input_layer = Input(shape=image_shape)

    current_last_layer = input_layer
    for index in range(5):

        current_last_layer = add_keras_model_block(current_last_layer, index)

    flat = Flatten()(current_last_layer)

    dropout_1 = Dropout(rate=0.2)(flat)
    dense_1 = Dense(512, activation="relu")(dropout_1)

    dropout_2 = Dropout(rate=0.2)(dense_1)
    dense_2 = Dense(64, activation="relu")(dropout_2)

    output_forest = Dense(1, activation="sigmoid", name=IS_MAJORITY_FOREST)(dense_2)
    output_roads = Dense(1, activation="sigmoid", name=HAS_ROADS)(dense_2)
    output_land_cover = Dense(n_land_cover_classes, activation="softmax", name=MODAL_LAND_COVER)(dense_2)

    model = Model(inputs=input_layer, outputs=[output_forest, output_roads, output_land_cover])

    print(model.summary())

    nadam = keras.optimizers.Nadam()

    model.compile(
        optimizer=nadam,
        loss={
            IS_MAJORITY_FOREST: keras.losses.binary_crossentropy,
            HAS_ROADS: keras.losses.binary_crossentropy,
            MODAL_LAND_COVER: keras.losses.categorical_crossentropy,
        },
        metrics=["accuracy"],
    )

    return model
