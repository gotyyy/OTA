import os
import numpy as np
import keras
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution2D
import keras.models as models
from keras.layers import Input, LSTM, Dense

def LSTMLikeModel(weights=None,
             input_shape=[1024,2],
             classes=24,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate (%)
    model = models.Sequential()
    model.add(Reshape(input_shape + [1], input_shape=input_shape))

    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(50, (1, 8), padding='valid', activation="relu", name="conv11",
                            kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))

    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(50, (1, 8), padding="valid", activation="relu", name="conv12",
                            kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))

    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(50, (1, 8), padding="valid", activation="relu", name="conv13",
                     kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    model.add(Dropout(dr))

    model.add(Dense(classes, kernel_initializer='he_normal', name="dense2"))
    model.add(Activation('softmax'))

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

