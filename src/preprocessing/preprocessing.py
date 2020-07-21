import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

def preprocessing_data(data):
    data_X = []
    Y_data = []
    for row in data:
        data_X.append(cv2.resize(row['crop'], (32,32)))
        Y_data.append(row['label'])
    data_X = np.array(data_X)
    X = transform_X(data_X.astype('float32'))
    Y = transform_Y(Y_data)
    return X, Y

def transform_Y(Y):
    labelencoder = LabelEncoder()
    y_df = pd.DataFrame(Y, columns=['Label'])
    y_df['Encoded'] = labelencoder.fit_transform(y_df['Label'])
    y_cat = to_categorical(y_df['Encoded'])
    return y_cat

def transform_X(X):
    img_rows, img_cols = X[0].shape

    # transform data set
    if K.image_data_format() == 'channels_first':
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    return X