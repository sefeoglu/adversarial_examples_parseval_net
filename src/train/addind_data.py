import pandas as pd
import pickle
import tensorflow.keras.backend as K
import tensorflow as tf

import tensorflow as tf
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
import gzip
from train_utily import noise
import warnings

warnings.filterwarnings("ignore")
import tensorflow

print("\nTensorflow Version: " + tf.__version__)
from _utility import lrate, get_adversarial_examples, print_test
from wresnet import WideResidualNetwork
import os

## globals
epsilons = [0.001, 0.003, 0.005, 0.01, 0.03]
percents = [0.25, 0.5, 0.75, 1.0]
os.mkdir("RandomnoiseModels")
os.mkdir("AEModels")
folder_list = ["RandomnoiseModels", "AEModels"]


def data_augmentation(epsilon, percent, X, Y, perturbation_type):
    split = int(len(X) * percent)
    file_name = str(epsilon) + ".pickle"
    X_adv_percent = list()
    if perturbation_type[0] == "FGSM":
        X_adv_percent = get_adversarial_examples(model, X[:split], Y[:split], epsilon)
    else:
        X_adv_percent = noise(X[:split], eps=epsilon)

    aug_X = np.concatenate((X, X_adv_percent), axis=0)
    Y_adv = Y[:split]
    aug_Y = np.concatenate((Y, Y_adv), axis=0)

    return aug_X, aug_Y


def experiments(X, Y, folder):

    perturbation_type = ["FGSM" if folder == "AEModels" else "Random"]

    for epsilon in epsilons:
        for percent in percents:
            aug_X, aug_Y = data_augmentation(epsilon, percent, X, Y, perturbation_type)
            train(aug_X, aug_Y, percent, epsilon, folder)


def train(X, Y, percent, epsilon, folder):

    "Ten fold CVs of ResNet"
    BS = 64
    init = (32, 32, 1)
    sgd = SGD(lr=0.1, momentum=0.9)
    kfold = KFold(n_splits=10, random_state=42, shuffle=False)
    model_name = folder + "/ResNet_" + str(epsilon) + "_" + str(percent)

    for j, (train, val) in enumerate(kfold.split(X)):

        resnet = WideResidualNetwork(
            init, 0.0001, 0.9, nb_classes=4, N=2, k=1, dropout=0.0
        )
        model = resnet.create_wide_residual_network()

        x_train, y_train = X[train], Y[train]
        x_val, y_val = X[val], Y[val]

        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["acc"])

        hist = model.fit(
            generator.flow(x_train, y_train, batch_size=64),
            steps_per_epoch=len(x_train) // 64,
            epochs=50,
            validation_data=(x_val, y_val),
            validation_steps=len(x_val) // 64,
            callbacks=[lrate],
        )

        name = model_name + "_" + str(j) + ".h5"
        hist_name = model_name + "_acc" + "_" + str(j) + ".pickle"
        hist_name_loss = model_name + "_loss" + "_" + str(j) + ".pickle"

        with open(hist_name, "wb") as f:
            pickle.dump(hist.history["val_acc"], f)

        with open(hist_name_loss, "wb") as f:
            pickle.dump(hist.history["val_loss"], f)

        model.save_weights(name)


data = hkl.load("data.hkl")

X_train, X_test, Y_train, y_test = (
    data["xtrain"],
    data["xtest"],
    data["ytrain"],
    data["ytest"],
)

for folder in folder_list:
    experiments(X_train, Y_train, folder)
