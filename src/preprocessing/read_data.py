import gzip
import pickle
import numpy as np
import cv2


def read_data():
    """[summary]

    Returns:
        new_data_x: cropped image data which was already transformed
        y_data: label of images, open Closed, PartlyOpen, NotVisible
    """
    with open("data.pz", "rb") as file_:
        with gzip.GzipFile(fileobj=file_) as gzf:
            data = pickle.load(gzf, encoding="latin1", fix_imports=True)
    new_data_x = []

    y_data = []

    for row in data:
        new_data_x.append(cv2.resize(row["crop"], (32, 32)))
        y_data.append(row["label"])

    new_data_x = np.array(new_data_x)

    return new_data_x, y_data
