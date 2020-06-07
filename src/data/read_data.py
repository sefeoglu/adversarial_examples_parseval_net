import gzip
import pickle
import numpy as np

def read_data():
    with open("data.pz", 'rb') as file_:
        with gzip.GzipFile(fileobj=file_) as gzf:
            data = pickle.load(gzf, encoding='latin1', fix_imports=True)
    return data

if __name__ == "__main__":
    data = read_data()
    new_data_X = []
    Y_data = []
    for row in data:
        new_data_X.append(row['crop'])
        Y_data.append(row['label'])
    new_data_X = np.array(new_data_X)
    new_data_X.shape