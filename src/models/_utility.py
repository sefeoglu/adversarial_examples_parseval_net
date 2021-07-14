from tensorflow.keras.callbacks import LearningRateScheduler

# Define configuration parameters
import math
import cleverhans
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import tensorflow as tf

import numpy as np


def step_decay(epoch):
    """[summary]

    Args:
        epoch (int): epoch number

    Returns:
        lrate(float): new learning rate
    """
    initial_lrate = 0.1
    factor = 0.1
    if epoch < 10:
        lrate = initial_lrate
    elif epoch < 20:
        lrate = initial_lrate * math.pow(factor, 1)
    elif epoch < 30:
        lrate = initial_lrate * math.pow(factor, 2)
    elif epoch < 40:
        lrate = initial_lrate * math.pow(factor, 3)
    else:
        lrate = initial_lrate * math.pow(factor, 4)
    return lrate


def step_decay_conv(epoch):
    """step decay for learning rate in convolutional networks

    Args:
        epoch (int): epoch number

    Returns:
        lrate(float): new learning rate
    """
    initial_lrate = 0.01
    factor = 0.1
    if epoch < 10:
        lrate = initial_lrate
    elif epoch < 20:
        lrate = initial_lrate * math.pow(factor, 1)
    elif epoch < 30:
        lrate = initial_lrate * math.pow(factor, 2)
    elif epoch < 40:
        lrate = initial_lrate * math.pow(factor, 3)
    else:
        lrate = initial_lrate * math.pow(factor, 4)
    return lrate


def print_test(model, X_adv, X_test, y_test, epsilon):
    """
    returns the test results and show the SNR and evaluation results
    """
    loss, acc = model.evaluate(X_adv, y_test)
    print("epsilon: {} and test evaluation : {}, {}".format(epsilon, loss, acc))
    SNR = 20 * np.log10(np.linalg.norm(X_test) / np.linalg.norm(X_test - X_adv))
    print("SNR: {}".format(SNR))
    return loss, acc


def get_adversarial_examples(pretrained_model, X_true, y_true, epsilon):
    """
    The attack requires the model to ouput the logits
    returns the adversarial example/s of a given image/s for epsilon value using
    fast gradient sign method
    """
    logits_model = tf.keras.Model(
        pretrained_model.input, pretrained_model.layers[-1].output
    )
    X_adv = []

    for i in range(len(X_true)):

        random_index = i

        original_image = X_true[random_index]
        original_image = tf.convert_to_tensor(
            original_image.reshape((1, 32, 32))
        )  # The .reshape just gives it the proper form to input into the model, a batch of 1 a.k.a a tensor
        original_label = y_true[random_index]
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


lrate_conv = LearningRateScheduler(step_decay_conv)
lrate = LearningRateScheduler(step_decay)
