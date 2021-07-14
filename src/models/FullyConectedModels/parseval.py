from tensorflow.data import Dataset
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    MaxPooling2D,
    Dense,
    Dropout,
    MaxPool1D,
    Flatten,
    AveragePooling1D,
    BatchNormalization,
)
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
import warnings
from constraint import tight_frame

warnings.filterwarnings("ignore")


def model_parseval(weight_decay):

    model_input = Input(
        shape=(
            32,
            32,
            1,
        )
    )
    model = Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(32, 32, 1),
        kernel_regularizer=l2(weight_decay),
        kernel_constraint=tight_frame(0.001),
        kernel_initializer="Orthogonal",
    )(model_input)
    model = Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="Orthogonal",
        kernel_constraint=tight_frame(0.001),
    )(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)
    model = Conv2D(
        128,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="Orthogonal",
        kernel_regularizer=l2(weight_decay),
        kernel_constraint=tight_frame(0.001),
    )(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)
    model = Conv2D(
        256,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="Orthogonal",
        kernel_regularizer=l2(weight_decay),
        kernel_constraint=tight_frame(0.001),
    )(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)
    model = Flatten()(model)
    model = Dense(4, activation="softmax", kernel_regularizer=l2(weight_decay))(model)
    model = Model(inputs=model_input, outputs=model)
    return model
