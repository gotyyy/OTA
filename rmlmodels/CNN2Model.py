from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
#os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(0)
WEIGHTS_PATH = ('cnn2_like_weights_tf_dim_ordering_tf_kernels.h5')
import numpy as np


import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation


# Build VT-CNN2 Neural Net model using Keras primitives --
#  - Reshape [N,2,128] to [N,2,128,1] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization



def CNN2Model(weights=None,
             input_shape=[1024,2],
             classes=24,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    dr = 0.5 # dropout rate (%)
    model = models.Sequential()
    model.add(Reshape(input_shape + [1], input_shape=input_shape))

    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(256, (1, 3), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))

    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(80, (2, 3), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    model.add(Dropout(dr))

    model.add(Dense( classes, kernel_initializer='he_normal', name="dense2" ))
    model.add(Activation('softmax'))

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

