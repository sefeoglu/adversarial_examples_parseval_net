from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from scipy import interp

# import cleverhans
import sys
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import os

plt.rcParams.update({"font.size": 14})

## add your path
prefix = +"logs/"


def learning_curves(percent, epsilon, exp, curve_type):
    acc_list = []
    "Ten fold CVs of ResNet"
    BS = 64
    init = (32, 32, 1)
    sgd = SGD(lr=0.1, momentum=0.9)
    model_name = prefix + exp + "/ResNet_" + str(epsilon) + "_" + str(percent)
    for j in range(10):
        hist_name = model_name + "_" + curve_type + "" + "_" + str(j) + ".pickle"
        with open(hist_name, "rb") as f:
            acc = pickle.load(f)
        acc_list.append(acc)
    return acc_list


def learning_curves_plot(train_sizes, mean_acc, std_acc, exp, curve_type):

    plt.figure(dpi=80)
    # Draw lines
    plt.fill_between(
        train_sizes, mean_acc - std_acc, mean_acc + std_acc, alpha=0.1, color="r"
    )
    plt.plot(mean_acc, "-", color="r", label="ResNet")

    plt.fill_between(
        train_sizes,
        mean_list[0] - std_list[0],
        mean_list[0] + std_list[0],
        alpha=0.1,
        color="darkgreen",
    )
    plt.plot(mean_list[0], "-", color="darkgreen", label="ResNet-0.25 AEs")

    plt.fill_between(
        train_sizes,
        mean_list[1] - std_list[1],
        mean_list[1] + std_list[1],
        alpha=0.1,
        color="darkblue",
    )
    plt.plot(mean_list[1], "-", color="darkblue", label="ResNet-0.5 AEs")

    plt.fill_between(
        train_sizes,
        mean_list[2] - std_list[2],
        mean_list[2] + std_list[2],
        alpha=0.1,
        color="darkorange",
    )
    plt.plot(mean_list[2], "-", color="darkorange", label="ResNet-0.75 AEs")

    plt.fill_between(
        train_sizes,
        mean_list[3] - std_list[3],
        mean_list[3] + std_list[3],
        alpha=0.1,
        color="magenta",
    )
    plt.plot(mean_list[3], "-", color="magenta", label="ResNet-1.0 AEs")

    # # Create plot
    plt.title("Learning Curves of Models (epsilon={})".format(epsilon))
    plt.xlabel("Epoch"), plt.ylabel(curve_type), plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(prefix + exp + curve_type + "/" + str(epsilon) + ".png")


train_sizes = [i for i in range(50)]

epsilons = [0.001, 0.003, 0.005, 0.01, 0.03]

percents = [0.25, 0.5, 0.75, 1.0]

Curve_Types = ["loss", "acc"]

Experiment = ["AEModels", "RandomNoisemodels"]


for exp in Experiment:
    for curve_type in Curve_Types:

        for epsilon in epsilons:

            mean_list, std_list = [], []
            train_mean_list, train_std_list = [], []

            for percent in percents:

                acc_list = learning_curves(percent, epsilon, exp, curve_type)
                mean_list.append(np.mean(acc_list, axis=0))
                std_list.append(np.std(acc_list, axis=0))

            learning_curves_plot(train_sizes, mean_acc, std_list, exp, curve_type)
