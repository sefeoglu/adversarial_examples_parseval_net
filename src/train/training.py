import numpy as np
from cleverhans.future.tf2.attacks import fast_gradient_method
import pandas as pd
from sklearn.model_selection import KFold
import sys
import tensorflow
import tensorflow as tf
sys.path.insert(1,'/home/sefika/AE_Parseval_Network/src/models')


def train(instance, X_train, Y_train, X_test, y_test, epochs,
              BS, sgd, generator, callbacks_list):
        # init dimensions
        init = (32, 32, 1)
        # rotate the images to improve the acc.

        res_df = pd.DataFrame(columns=['loss_clean', 'acc_clean'])
        # Ten fold cross validation

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
                epochs= 50, callbacks = callbacks_list,
                validation_data=(x_val, y_val),
                validation_steps=x_val.shape[0] // BS,
            )
            ## write the history

            loss, acc = model.evaluate(X_test, y_test)
           
            row = {'loss_clean': loss, 'acc_clean': acc,}
            res_df = res_df.append(row, ignore_index=True)

        return res_df