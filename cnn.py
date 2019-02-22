import keras
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D

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
