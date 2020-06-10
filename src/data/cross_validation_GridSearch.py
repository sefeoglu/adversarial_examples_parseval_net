from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from itertools import product
import pickle

import sys
sys.path.insert(1,'/home/sefika/AE_Parseval_Network/src')
from models.wideresnet.wresnet import WideResidualNetwork
from data.preprocessing import preprocessing

class ModelSelection(object):
    def __init__(self):
        pass
    def KFold_GridSearchCV(self, input_dim, X, Y, X_test, y_test, combinations):
        """Custom Grid SearchCV algorithm with KFold Cross Validation"""
        model_instance = WideResidualNetwork()
        for i, combination in enumerate(combinations):
            kf = KFold(n_splits=3, random_state=42, shuffle=False)
            for j, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_val = X[train_index], X[test_index]
                y_train, y_val = Y[train_index], Y[test_index]
                
                model = model_instance.create_wide_residual_network(combination[2], combination[0], in_dim, nb_classes=4, N=2, k=2, dropout=0.3)
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=80, min_delta=0.1)
                hist = model.fit(X_train,y_train,batch_size = combination[1], steps_per_epoch=len(X_train[0])//combination[1], epochs=combination[3] ,validation_data=(X_val,y_val), validation_steps =len(X_val[0])//combination[1])
                ### store combination and data###
                loss_test, acc_test = model.evaluate(X_test, y_test)
                self.store_data_split(j, X_train,y_train, X_val, y_val)
                self.store_hist(j, hist, es.stopped_epoch, loss_test, acc_test)

            if j == 6:

                break
    def store_data_split(self,name,X_train, y_train, X_validation, y_validation):
        file_name = name+"_train_data.pkl"
        my_data = {'X_train': X_train,
                'Y_train': y_train,
                'Y_validation' : y_test,
                'X_validation' :X_validation
                }
        output = open(file_name, 'wb')
        pickle.dump(my_data, output)
        output.close()
    def store_hist(self,name, hist,epoch, loss_test, acc_test):
        file_name = name+"_hist_data.pkl"
        my_data = {'hist' : hist,
        'epochs' : epoch,
        'loss_test' : loss_test,
        'acc_test' : acc_test
        }
        output = open(file_name, 'wb')
        pickle.dump(my_data, output)
        output.close()

if __name__ == "__main__":
    learning_rate = [0.1, 0.01]
    batch_size = [32,64,128]
    penalty_term = [0.01, 0.001, 0.0001, 0.0005]
    max_epochs = [5000]
    transfer = [True, False]
    add_layers = [True, False]
    in_dim = (100,68,1)
    # create list of all different parameter combinations
    X, Y = preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.1, Shuffle = False)

    param_grid = dict(learning_rate = learning_rate, batch_size = batch_size,penalty_term = penalty_term, epochs = max_epochs)
    combinations = list(product(*param_grid.values()))
    print(combinations)
    instance  = ModelSelection()
    instance.KFold_GridSearchCV(in_dim,X_train,y_train,X_test, y_test, combinations)


