import os
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Flatten
from keras.layers.convolutional import Conv1D,Conv2D ,MaxPooling1D,ZeroPadding2D
from keras.layers.normalization import  BatchNormalization
from keras.layers.core import Activation
from keras import regularizers



def CNN2Model(weights='cnnLike-64k.wts.h5',
             input_shape=[1024,2],
             classes=24,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    dr = 0.5
    model = models.Sequential()
    model.add(Reshape(input_shape+[1], input_shape=input_shape))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(128, (1, 8),strides=(1,4), padding='valid', activation="relu", name="conv1", init='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(80, (2, 4),strides=(1,2), padding='valid', activation="relu", name="conv2", init='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation='relu',init='he_normal'))
    model.add(Dropout(dr))
    model.add(Dense((classes),activation='softmax',init='he_normal'))
    model.add(Reshape(classes))

    if weights is not None:
        model.load_weights(weights)

    return model


