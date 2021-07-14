#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 14})
## add your path
prefix = +"logs/"


def plot_figure(x_acc, x_size, y_err, title, epsilon, exp):
    plt.figure(figsize=(10, 8))
    plt.errorbar(x_001, y_001_acc, yerr=yerr, fmt="o", label="ACC")
    plt.title("{0} = {0}".format(title, epsilon))
    plt.xlabel("Percent of Adversarial Examples")
    plt.ylabel("Accuracy")

    plt.savefig(prefix + exp + "/performance/" + str(epsilon) + ".png")


Experiment = ["AEModels", "RandomNoisemodels"]


for exp in Experiment:
    columns = ["Model_ID", "percent", "epsilon", "accuracy", "attack"]

    table = pd.DataFrame(columns=columns)
    table
    for file in os.listdir(prefix + exp):
        format = file.split("_")[3].split(".")[-1]
        if format == "h5":
            epsilon = file.split("_")[1]
            percent = file.split("_")[2]
            ModelID = file.split("_")[3].split(".")[0]
            resnet = WideResidualNetwork(
                init, 0.0001, 0.9, nb_classes=4, N=2, k=1, dropout=0.0
            )
            resnet_model = resnet.create_wide_residual_network()
            resnet_model.compile(
                loss="categorical_crossentropy", optimizer=sgd, metrics=["acc"]
            )
            resnet_model.load_weights(prefix + exp + "/" + file)
            acc = resnet_model.evaluate(X_test, y_test)

            row = {
                "Model_ID": ModelID,
                "percent": percent,
                "epsilon": epsilon,
                "accuracy": acc[1],
                "attack": 0,
            }
            table = table.append(row, ignore_index=True)
    tab_25 = table[table["percent"] == "0.25"]
    tab_5 = table[table["percent"] == "0.5"]
    tab_75 = table[table["percent"] == "0.75"]
    tab_1 = table[table["percent"] == "1.0"]

    tab_25_001 = tab_25[tab_25["epsilon"] == "0.001"]
    tab_25_001["mean_acc"], tab_25_001["mean_att"] = (
        sum(tab_25_001["accuracy"]) / 10,
        sum(tab_25_001["attack"]) / 10,
    )
    tab_25_01 = tab_25[tab_25["epsilon"] == "0.01"]
    tab_25_01["mean_acc"], tab_25_01["mean_att"] = (
        sum(tab_25_01["accuracy"]) / 10,
        sum(tab_25_01["attack"]) / 10,
    )
    tab_25_03 = tab_25[tab_25["epsilon"] == "0.03"]
    tab_25_03["mean_acc"], tab_25_03["mean_att"] = (
        sum(tab_25_03["accuracy"]) / 10,
        sum(tab_25_03["attack"]) / 10,
    )
    tab_25_005 = tab_25[tab_25["epsilon"] == "0.005"]
    tab_25_005["mean_acc"], tab_25_005["mean_att"] = (
        sum(tab_25_005["accuracy"]) / 10,
        sum(tab_25_005["attack"]) / 10,
    )
    tab_25_003 = tab_25[tab_25["epsilon"] == "0.003"]
    tab_25_003["mean_acc"], tab_25_003["mean_att"] = (
        sum(tab_25_003["accuracy"]) / 10,
        sum(tab_25_003["attack"]) / 10,
    )

    tab_5_001 = tab_5[tab_5["epsilon"] == "0.001"]
    tab_5_001["mean_acc"], tab_5_001["mean_att"] = (
        sum(tab_5_001["accuracy"]) / 10,
        sum(tab_5_001["attack"]) / 10,
    )
    tab_5_01 = tab_5[tab_5["epsilon"] == "0.01"]
    tab_5_01["mean_acc"], tab_5_01["mean_att"] = (
        sum(tab_5_01["accuracy"]) / 10,
        sum(tab_5_01["attack"]) / 10,
    )

    tab_5_03 = tab_5[tab_5["epsilon"] == "0.03"]
    tab_5_03["mean_acc"], tab_5_03["mean_att"] = (
        sum(tab_5_03["accuracy"]) / 10,
        sum(tab_5_03["attack"]) / 10,
    )
    tab_5_005 = tab_5[tab_5["epsilon"] == "0.005"]
    tab_5_005["mean_acc"], tab_5_005["mean_att"] = (
        sum(tab_5_005["accuracy"]) / 10,
        sum(tab_5_005["attack"]) / 10,
    )
    tab_5_003 = tab_5[tab_5["epsilon"] == "0.003"]
    tab_5_003["mean_acc"], tab_5_003["mean_att"] = (
        sum(tab_5_003["accuracy"]) / 10,
        sum(tab_5_003["attack"]) / 10,
    )
    tab_1_001 = tab_1[tab_1["epsilon"] == "0.001"]
    tab_1_001["mean_acc"], tab_1_001["mean_att"] = (
        sum(tab_1_001["accuracy"]) / 10,
        sum(tab_1_001["attack"]) / 10,
    )
    tab_1_01 = tab_1[tab_1["epsilon"] == "0.01"]
    tab_1_01["mean_acc"], tab_1_01["mean_att"] = (
        sum(tab_1_01["accuracy"]) / 10,
        sum(tab_1_01["attack"]) / 10,
    )
    tab_1_03 = tab_1[tab_1["epsilon"] == "0.03"]
    tab_1_03["mean_acc"], tab_1_03["mean_att"] = (
        sum(tab_1_03["accuracy"]) / 10,
        sum(tab_1_03["attack"]) / 10,
    )
    tab_1_005 = tab_1[tab_1["epsilon"] == "0.005"]
    tab_1_005["mean_acc"], tab_1_005["mean_att"] = (
        sum(tab_1_005["accuracy"]) / 10,
        sum(tab_1_005["attack"]) / 10,
    )
    tab_1_003 = tab_1[tab_1["epsilon"] == "0.003"]
    tab_1_003["mean_acc"], tab_1_003["mean_att"] = (
        sum(tab_1_003["accuracy"]) / 10,
        sum(tab_1_003["attack"]) / 10,
    )

    tab_75_001 = tab_75[tab_75["epsilon"] == "0.001"]
    tab_75_001["mean_acc"], tab_75_001["mean_att"] = (
        sum(tab_75_001["accuracy"]) / 10,
        sum(tab_75_001["attack"]) / 10,
    )
    tab_75_01 = tab_75[tab_75["epsilon"] == "0.01"]
    tab_75_01["mean_acc"], tab_75_01["mean_att"] = (
        sum(tab_75_01["accuracy"]) / 10,
        sum(tab_75_01["attack"]) / 10,
    )
    tab_75_03 = tab_75[tab_75["epsilon"] == "0.03"]
    tab_75_03["mean_acc"], tab_75_03["mean_att"] = (
        sum(tab_75_03["accuracy"]) / 10,
        sum(tab_75_03["attack"]) / 10,
    )
    tab_75_005 = tab_75[tab_75["epsilon"] == "0.005"]
    tab_75_005["mean_acc"], tab_75_005["mean_att"] = (
        sum(tab_75_005["accuracy"]) / 10,
        sum(tab_75_005["attack"]) / 10,
    )
    tab_75_003 = tab_75[tab_75["epsilon"] == "0.003"]
    tab_75_003["mean_acc"], tab_75_003["mean_att"] = (
        sum(tab_75_003["accuracy"]) / 10,
        sum(tab_75_003["attack"]) / 10,
    )

    table_mean = table_mean.append(tab_25_001.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_25_01.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_25_03.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_25_005.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_25_003.head(1), ignore_index=True)
    ###
    table_mean = table_mean.append(tab_5_001.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_5_01.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_5_03.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_5_005.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_5_003.head(1), ignore_index=True)
    ###
    table_mean = table_mean.append(tab_75_001.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_75_01.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_75_03.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_75_005.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_75_003.head(1), ignore_index=True)
    ###
    table_mean = table_mean.append(tab_1_001.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_1_01.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_1_03.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_1_005.head(1), ignore_index=True)
    table_mean = table_mean.append(tab_1_003.head(1), ignore_index=True)

    print(
        table_mean[column].sort_values(by=["epsilon", "percent"]).to_latex(index=False)
    )

    y_001 = table_mean[table_mean["epsilon"] == "0.001"]["mean_att"]
    y_001_acc = table_mean[table_mean["epsilon"] == "0.001"]["mean_acc"]
    x_001 = table_mean[table_mean["epsilon"] == "0.001"]["percent"]

    y_003 = table_mean[table_mean["epsilon"] == "0.003"]["mean_att"]
    y_003_acc = table_mean[table_mean["epsilon"] == "0.003"]["mean_acc"]
    x_003 = table_mean[table_mean["epsilon"] == "0.003"]["percent"]

    y_005 = table_mean[table_mean["epsilon"] == "0.005"]["mean_att"]
    y_005_acc = table_mean[table_mean["epsilon"] == "0.005"]["mean_acc"]
    x_005 = table_mean[table_mean["epsilon"] == "0.005"]["percent"]

    y_01 = table_mean[table_mean["epsilon"] == "0.01"]["mean_att"]
    y_01_acc = table_mean[table_mean["epsilon"] == "0.01"]["mean_acc"]
    x_01 = table_mean[table_mean["epsilon"] == "0.01"]["percent"]

    y_03 = table_mean[table_mean["epsilon"] == "0.03"]["mean_att"]
    y_03_acc = table_mean[table_mean["epsilon"] == "0.03"]["mean_acc"]
    x_03 = table_mean[table_mean["epsilon"] == "0.03"]["percent"]

    l_25 = []
    l_25.append(np.std(tab_25_001["accuracy"], axis=0))
    l_25.append(np.std(tab_25_01["accuracy"], axis=0))
    l_25.append(np.std(tab_25_03["accuracy"], axis=0))
    l_25.append(np.std(tab_25_005["accuracy"], axis=0))
    l_25.append(np.std(tab_25_003["accuracy"], axis=0))
    # ###
    l_5 = []
    l_5.append(np.std(tab_5_001["accuracy"], axis=0))
    l_5.append(np.std(tab_5_01["accuracy"], axis=0))
    l_5.append(np.std(tab_5_03["accuracy"], axis=0))
    l_5.append(np.std(tab_5_005["accuracy"], axis=0))
    l_5.append(np.std(tab_5_003["accuracy"], axis=0))
    # ###
    l_75 = []
    l_75.append(np.std(tab_75_001["accuracy"], axis=0))
    l_75.append(np.std(tab_75_01["accuracy"], axis=0))
    l_75.append(np.std(tab_75_03["accuracy"], axis=0))
    l_75.append(np.std(tab_75_005["accuracy"], axis=0))
    l_75.append(np.std(tab_75_003["accuracy"], axis=0))
    # ###

    l_1 = []
    l_1.append(np.std(tab_1_001["accuracy"], axis=0))
    l_1.append(np.std(tab_1_01["accuracy"], axis=0))
    l_1.append(np.std(tab_1_03["accuracy"], axis=0))
    l_1.append(np.std(tab_1_005["accuracy"], axis=0))
    l_1.append(np.std(tab_1_003["accuracy"], axis=0))

    x_acc = [x_001_acc, x_01_acc, x_03_acc, x_005_acc, x_003_acc]
    x_sizes = [x_001, x_01, x_03, x_005, x_003]
    epsilons = [0.001, 0.01, 0.03, 0.005, 0.003]
    # example variable error bar values

    for i in range(5):
        yerr = [0.01, l_25[i], l_5[i], l_75[i], l_1[i]]
        plot_figure(x_acc[i], x_sizes[i], yerr, epsilons[i], exp)
