import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K


def preprocessing_data(data):
    """[summary]

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    x_input = []
    y_input = []

    for row in data:
        x_input.append(cv2.resize(row['crop'], (32, 32)))
        y_input.append(row['label'])

    x_input = np.array(x_input)

    transformed_x = transform_imput(x_input.astype('float32'))

    transformed_y = transform_output(y_input)
    return transformed_x, transformed_y



def transform_output(output):
    """[summary]

    Args:
        output ([type]): [description]

    Returns:
        [type]: [description]
    """
    labelencoder = LabelEncoder()
    y_df = pd.DataFrame(output, columns=['Label'])
    y_df['Encoded'] = labelencoder.fit_transform(y_df['Label'])
    y_cat = to_categorical(y_df['Encoded'])
    return y_cat


def transform_imput(x_input):
    """[summary]

    Args:
        X ([type]): [description]

    Returns:
        [type]: [description]
    """
    img_rows, img_cols = x_input[0].shape

    # transform data set
    if K.image_data_format() == 'channels_first':
        x_input = x_input.reshape(x_input.shape[0], 1, img_rows, img_cols)
    else:
        x_input = x_input.reshape(x_input.shape[0], img_rows, img_cols, 1)
    return x_input
