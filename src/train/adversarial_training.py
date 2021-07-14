import numpy as np
from cleverhans.future.tf2.attacks import fast_gradient_method
import pandas as pd
from sklearn.model_selection import KFold
import sys
import tensorflow
import tensorflow as tf
from multiprocessing import Pool

from _utility import lrate, get_adversarial_examples, print_test, step_decay
import hickle as hkl
import pickle

model_name = "ResNet_da"


class AdversarialTraining(object):
    """
    The class provides an adversarial training for a given model and epsilon values.
    In addition to this, the class changes the half of the batch with their adversarial examples.
    The adversarial exaples obtain using fast gradient sign method of CleverHans framework.
    """

    def __init__(self, parameter):
        self.epochs = parameter["epochs"]
        self.batch_size = parameter["batch_size"]
        self.optimizer = parameter["optimizer"]

        self.generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=5.0 / 32,
            height_shift_range=5.0 / 32,
        )

    def train(self, model, train_dataset, val_dataset, epsilon_list):

        # Ten fold cross validation
        for epoch in range(self.epochs):
            lr_rate = step_decay(epoch)
            tf.keras.backend.set_value(model.optimizer.learning_rate, lr_rate)

            for step, (x_train, y_train) in enumerate(train_dataset):
                print(step)
                x_train = self.data_augmentation(x_train, y_train, model, epsilon_list)
                model.fit(
                    self.generator.flow(x_train, y_train, self.batch_size),
                    batch_size=self.batch_size,
                    verbose=0.0,
                )

    def data_augmentation(self, X_train, Y_train, pretrained_model, epsilon_list):
        """[summary]

        Args:
            X_train ([type]): Training inputs
            Y_train ([type]): outputs
            epsilon_list ([type]): according to SNR

        Returns:
            augmented batch which consists of the adversarial and clean examples.
        """
        first_half_end = int(len(X_train) / 2)
        second_half_end = int(len(X_train))
        x_clean = X_train[0:first_half_end, :, :, :]
        x_adv = self.get_adversarial(
            pretrained_model,
            X_train[first_half_end:second_half_end, :, :, :],
            Y_train[first_half_end:second_half_end],
            epsilon_list,
        )
        x_mix = self.merge_data(x_clean, x_adv)
        y_mix = Y_train[0:second_half_end]

        return x_mix, y_mix

    def merge_data(self, x_clean, x_adv):
        """[summary]

        Args:
            x_clean ([type]): [description]
            x_adv ([type]): [description]

        Returns:
            combine the clean and adversarial inputs.
        """
        x_mix = []
        for i in range(len(x_clean)):
            x_mix.append(x_clean[i])
        for j in range(len(x_adv)):
            x_mix.append(x_adv[j])
        x_mix = np.array(x_mix)

        return x_mix

    def get_adversarial(self, logits_model, X_true, y_true, epsilon_list):
        return self.adversarial_example(logits_model, X_true, y_true, epsilon_list)

    def adversarial_example(self, logits_model, X_true, y_true, epsilon_list):
        X_adv = []

        for index, x_true in enumerate(X_true):
            epsilon = epsilon_list[index]

            original_image = x_true
            original_image = tf.reshape(original_image, (1, 32, 32))
            original_label = y_true[index]
            original_label = np.reshape(np.argmax(original_label), (1,)).astype("int64")
            adv_example_targeted_label = fast_gradient_method(
                logits_model,
                original_image,
                epsilon,
                np.inf,
                y=original_label,
                targeted=False,
            )
            X_adv.append(np.array(adv_example_targeted_label).reshape(32, 32, 1))
        X_adv = np.array(X_adv)

        return X_adv


def simulate_train(s):

    for j, (train, val) in enumerate(kfold.split(X_train)):
        if j == s:
            print(s)
            model = wideresnet.create_wide_residual_network()
            model.compile(
                loss="categorical_crossentropy", optimizer=sgd, metrics=["acc"]
            )
            print("Finished compiling")
            x_train, y_train = X_train[train], Y_train[train]
            x_val, y_val = X_train[val], Y_train[val]
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_dataset = train_dataset.batch(BS)
            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            val_dataset = val_dataset.batch(BS)
            adversarial_training.train(model, train_dataset, val_dataset, epsilons)
            name = model_name + "_" + str(j) + ".h5"
            model.save_weights(name)


if __name__ == "__main__":

    data = hkl.load("data.hkl")
    X_train, X_test, Y_train, y_test = (
        data["xtrain"],
        data["xtest"],
        data["ytrain"],
        data["ytest"],
    )
    epsilons = [i / 1000 for i in range(1, 33)]  # factor for fast gradient sign method

    kfold = KFold(n_splits=10, random_state=42, shuffle=False)
    EPOCHS = 50
    BS = 64
    init = (32, 32, 1)
    sgd = SGD(lr=0.1, momentum=0.9)
    parameter = {"epochs": EPOCHS, "batch_size": BS, "optimizer": sgd}
    # change here depending on your model
    wideresnet = WideResidualNetwork(
        init, 0.0001, 0.9, nb_classes=4, N=2, k=1, dropout=0.0
    )

    with Pool(10) as p:
        print(p.map(f, np.range(10)))
