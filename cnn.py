import keras
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D


HAS_ROADS = "has_roads"
IS_MAJORITY_FOREST = "is_majority_forest"


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

    output_forest = Dense(1, activation="sigmoid", name=IS_MAJORITY_FOREST)(flat)
    output_roads = Dense(1, activation="sigmoid", name=HAS_ROADS)(flat)

    model = Model(inputs=input_layer, outputs=[output_forest, output_roads])

    print(model.summary())

    nadam = keras.optimizers.Nadam()

    model.compile(
        optimizer=nadam,
        loss={
            IS_MAJORITY_FOREST: keras.losses.binary_crossentropy,
            HAS_ROADS: keras.losses.binary_crossentropy,
        },
        metrics=["accuracy"],
    )

    return model
