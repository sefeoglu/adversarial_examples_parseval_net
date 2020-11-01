import numpy as np
from cleverhans.future.tf2.attacks import fast_gradient_method
import pandas as pd
from sklearn.model_selection import KFold
import sys
import tensorflow
import tensorflow as tf

from _utility import print_test, get_adversarial_examples

import pickle

def train(instance, X_train, Y_train, X_test, y_test, epochs,
    BS, sgd, generator, callbacks_list, epsilon_list, model_name="ResNet"):
    """[summary]

    Args:
        instance ([type]): [description]
        X_train ([type]): [description]
        Y_train ([type]): [description]
        X_test ([type]): [description]
        y_test ([type]): [description]
        epochs ([type]): [description]
        BS ([type]): [description]
        sgd ([type]): [description]
        generator ([type]): [description]
        callbacks_list ([type]): [description]
        model_name (str, optional): [description]. Defaults to "ResNet".

    Returns:
        [type]: [description]
    """    

    res_df = pd.DataFrame(columns=["loss_clean","acc_clean","0.003_loss","0.003_acc","0.005_loss","0.005_acc","0.02_acc","0.02_loss", "0.01_acc","0.01_clean"])


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
            epochs= epochs, callbacks = callbacks_list,
            validation_data=(x_val, y_val),
            validation_steps=x_val.shape[0] // BS,
            )
                ## write the history
        with open('history_'+model_name+str(j), 'wb') as file_pi:
          pickle.dump(hist.history, file_pi)
        loss, acc = model.evaluate(X_test, y_test)

        loss1, acc1 = print_test(model,
                                 get_adversarial_examples(model, X_test, y_test,
                                                          epsilon_list[0]), X_test, y_test,
                                 epsilon_list[0])
        loss2, acc2 = print_test(model,
                                 get_adversarial_examples(model, X_test, y_test,
                                                          epsilon_list[1]), X_test, y_test,
                                 epsilon_list[1])
        loss3, acc3 = print_test(model,
                                 get_adversarial_examples(model, X_test, y_test,
                                                          epsilon_list[2]), X_test, y_test,
                                 epsilon_list[2])
        loss4, acc4 = print_test(model,
                                 get_adversarial_examples(model, X_test, y_test,
                                                          epsilon_list[3]), X_test, y_test,
                                 epsilon_list[3])
            # store the loss and accuracy
            
        row = {"loss_clean":loss,
               "acc_clean":acc,
               "0.003_loss":loss1,
               "0.003_acc":acc1,
               "0.005_loss":loss2,
               "0.005_acc":acc2,
               "0.02_acc":acc3,
               "0.02_loss":loss3,
               "0.01_acc":acc4,
               "0.01_loss":acc4
               }
        res_df = res_df.append(row, ignore_index=True)

    return res_df