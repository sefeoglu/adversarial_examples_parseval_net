import numpy as np
from cleverhans.future.tf2.attacks import fast_gradient_method
import pandas as pd
from sklearn.model_selection import KFold
import sys
import tensorflow
import tensorflow as tf

from _utility import print_test, get_adversarial_examples

import pickle

folder_name = "./adversarial_examples_parseval_net/src/logs/saved_models/"


def train(
    instance,
    X_train,
    Y_train,
    X_test,
    y_test,
    epochs,
    BS,
    sgd,
    generator,
    callbacks_list,
    model_name="ResNet",
):

    kfold = KFold(n_splits=10, random_state=42, shuffle=False)

    for j, (train, val) in enumerate(kfold.split(X_train)):

        model = instance.create_wide_residual_network()
        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["acc"])

        print("Finished compiling")

        x_train, y_train = X_train[train], Y_train[train]
        x_val, y_val = X_train[val], Y_train[val]

        hist = model.fit(
            generator.flow(x_train, y_train, batch_size=BS),
            steps_per_epoch=len(x_train) // BS,
            epochs=epochs,
            callbacks=callbacks_list,
            validation_data=(x_val, y_val),
            validation_steps=x_val.shape[0] // BS,
        )
        ## write the history

        with open("history_" + model_name + str(j), "wb") as file_pi:
            pickle.dump(hist.history, file_pi)

        model_name = folder_name + model_name + "_" + str(j) + ".h5"
        model.save(model_name)
