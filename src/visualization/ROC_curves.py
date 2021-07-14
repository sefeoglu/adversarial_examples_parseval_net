from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from scipy import interp

# import cleverhans
import sys
import tensorflow as tf

# from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
import gzip
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import tensorflow

print("\nTensorflow Version: " + tf.__version__)
from wresnet import WideResidualNetwork
from parsevalnet import ParsevalNetwork
import hickle as hkl
import os

plt.rcParams.update({"font.size": 14})
## add your path
prefix = ""


def ROC_result(model):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = model.predict(X_test)
    # # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    return fpr, tpr


def plot_roc(
    fpr_l, tpr_l, roc_auc_l, std_tpr_list, std_roc_auc_list, epsilon, fig_name
):

    plt.figure(figsize=(10, 8))
    lw = 2

    plt.fill_between(
        fpr_l,
        tpr_l[0] - std_tpr_list[0],
        tpr_l[0] + std_tpr_list[0],
        alpha=0.1,
        color="purple",
    )
    plt.plot(
        fpr_l,
        tpr_l[0],
        color="purple",
        lw=lw,
        label="ResNet (area = %0.4f)" % roc_auc_l[0],
    )

    plt.fill_between(
        fpr_l,
        tpr_l[1] - std_tpr_list[1],
        tpr_l[1] + std_tpr_list[1],
        alpha=0.1,
        color="darkgreen",
    )
    plt.plot(
        fpr_l,
        tpr_l[1],
        color="darkgreen",
        lw=lw,
        label="ResNet-0.25 AEs(area = %0.4f)" % roc_auc_l[1],
    )

    plt.fill_between(
        fpr_l,
        tpr_l[2] - std_tpr_list[2],
        tpr_l[2] + std_tpr_list[2],
        alpha=0.1,
        color="darkblue",
    )
    plt.plot(
        fpr_l,
        tpr_l[2],
        color="darkblue",
        lw=lw,
        label="ResNet-0.50 AEs(area = %0.4f)" % roc_auc_l[2],
    )

    plt.fill_between(
        fpr_l,
        tpr_l[3] - std_tpr_list[3],
        tpr_l[3] + std_tpr_list[3],
        alpha=0.1,
        color="pink",
    )
    plt.plot(
        fpr_l,
        tpr_l[3],
        color="pink",
        lw=lw,
        label="ResNet-0.75 AEs(area = %0.4f)" % roc_auc_l[3],
    )

    plt.fill_between(
        fpr_l,
        tpr_l[4] - std_tpr_list[4],
        tpr_l[4] + std_tpr_list[4],
        alpha=0.1,
        color="darkorange",
    )
    plt.plot(
        fpr_l,
        tpr_l[4],
        color="darkorange",
        lw=lw,
        label="ResNet-1.0 AEs (area = %0.4f)" % roc_auc_l[4],
    )

    plt.fill_between(
        fpr_l,
        tpr_l[5] - std_tpr_list[5],
        tpr_l[5] + std_tpr_list[5],
        alpha=0.1,
        color="darkmagenta",
    )
    plt.plot(
        fpr_l,
        tpr_l[5],
        color="darkmagenta",
        lw=lw,
        label="Parseval (area = %0.4f)" % roc_auc_l[5],
    )

    plt.plot(np.arange(0, 1, 0.001), np.arange(0, 1, 0.001), linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC For Epsilon = {}".format(epsilon))
    plt.legend(loc="lower right")

    plt.xscale("log")

    plt.savefig(fig_name)


BS = 64
init = (32, 32, 1)
sgd = SGD(lr=0.1, momentum=0.9)

data = hkl.load("data.hkl")
X_train, X_test, Y_train, y_test = (
    data["xtrain"],
    data["xtest"],
    data["ytrain"],
    data["ytest"],
)

Experiment = ["AEModels", "RandomNoisemodels"]
epsilons_list = [0.03, 0.01, 0.005, 0.003, 0.001]
percent_list = [0, 0.25, 0.5, 0.75, 1.0]

for exp in Experiment:

    for epsilon in epsilons_list:
        for percent in percent_list:
            tprs = []
            aucs = []
            mean_fpr = np.arange(0, 1, 0.001)
            mean_fpr_list = []
            mean_tpr_list = []
            mean_roc_auc_list = []
            std_fpr_list = []
            std_tpr_list = []
            std_roc_auc_list = []
            micro_roc_auc = []
            fpr_list, tpr_list, roc_auc_list = [], [], []
            for i in range(10):
                resnet = WideResidualNetwork(
                    init, 0.0001, 0.9, nb_classes=4, N=2, k=1, dropout=0.0
                )
                model = resnet.create_wide_residual_network()
                model.compile(
                    loss="categorical_crossentropy", optimizer=sgd, metrics=["acc"]
                )
                if percent != 0:
                    model_path = (
                        prefix
                        + ex
                        + "/ResNet_"
                        + str(epsilon)
                        + "_"
                        + str(percent)
                        + "_"
                        + str(i)
                        + ".h5"
                    )
                    print(model_path)
                    model.load_weights(model_path)
                else:
                    model_path = prefix + "ResNet/ResNet_" + str(i) + ".h5"
                    print(model_path)
                    model.load_weights(model_path)
                    fpr, tpr = ROC_result(model)
                    tprs.append(interp(mean_fpr, fpr["micro"], tpr["micro"]))
                    roc_auc = auc(fpr["micro"], tpr["micro"])
                    aucs.append(roc_auc)
                    tpr_list.append(tprs)
                    roc_auc_list.append(aucs)
                    init = (32, 32, 1)
                    parseval_micro_fpr, parseval_micro_tpr, parseval_micro_roc_auc = (
                        [],
                        [],
                        [],
                    )
for i in range(10):
    parseval = ParsevalNetwork(init, 0.0001, 0.9, nb_classes=4, N=2, k=1, dropout=0.0)

    model = parseval.create_wide_residual_network()
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["acc"])
    model_name = prefix + "ResNet/Parseval_" + str(i) + ".h5"
    model.load_weights(model_name)
    fpr, tpr = ROC_result(model)
    parseval_micro_tpr.append(interp(mean_fpr, fpr["micro"], tpr["micro"]))
    roc_auc = auc(fpr["micro"], tpr["micro"])
    parseval_micro_roc_auc.append(roc_auc)
    parseval_micro_roc_auc.append(roc_auc)
    tpr_list.append(parseval_micro_tpr)
    roc_auc_list.append(parseval_micro_roc_auc)

    for i in range(6):
        mean_tpr_list.append(np.mean(tpr_list[i], axis=0))
        mean_roc_auc_list.append(auc(mean_fpr, mean_tpr_list[i]))
        std_tpr_list.append(np.std(tpr_list[i], axis=0))
        std_roc_auc_list.append(auc(mean_fpr, std_tpr_list[i]))

    fig_name = prefix + exp + "/ROC/Model_Epsilon" + str(epsilon) + ".png"
    plot_roc(
        mean_fpr,
        mean_tpr_list,
        mean_roc_auc_list,
        std_tpr_list,
        std_roc_auc_list,
        epsilon,
        fig_name,
    )
