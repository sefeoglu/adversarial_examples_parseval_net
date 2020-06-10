
import pandas as pd
import numpy as np
from read_data import read_data
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def preprocessing():
    # creating initial dataframe
    X, Y = read_data()
    y_df = pd.DataFrame(Y, columns=['Label'])
    labelencoder = LabelEncoder()
    y_df['Cat_'] = labelencoder.fit_transform(y_df['Label'])


    img_rows, img_cols = X[0].shape


    # transform data set
    if K.common.image_data_format() == 'channels_first':
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    Y_cat = to_categorical(y_df['Cat_'])
    return X, Y_cat