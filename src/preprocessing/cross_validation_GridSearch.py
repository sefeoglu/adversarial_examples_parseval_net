from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from itertools import product
import pickle

import sys
sys.path.insert(1,'/home/sefika/AE_Parseval_Network/src')
from models.wideresnet.wresnet import WideResidualNetwork
from data.preprocessing import preprocessing
import tensorflow
class ModelSelection(object):
    """[summary]

    Args:
        object ([type]): [description]
    """    
    def __init__(self):
        """[summary]
        """        
        pass
    def KFold_GridSearchCV(self, input_dim, X, Y, X_test, y_test, combinations, filename="log.csv"):
         # create containers for resulting data
        """[summary]

        Args:
            input_dim ([type]): [description]
            X ([type]): [description]
            Y ([type]): [description]
            X_test ([type]): [description]
            y_test ([type]): [description]
            combinations ([type]): [description]
            filename (str, optional): [description]. Defaults to "log.csv".
        """         
        wresnet_ins = WideResidualNetwork()
        res_df = pd.DataFrame(columns=['momentum','learning rate','batch size',
                                      'loss1', 'acc1','loss2', 'acc2','loss3', 'acc3'])
        generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,)
        hist_dict_global = {}
        for i, combination in enumerate(combinations):
            kf = KFold(n_splits=3, random_state=42, shuffle=False)
            metrics_dict = {}
  
            for j, (train_index, test_index) in enumerate(kf.split(X)):
               if i >95:
                X_train, X_val = X[train_index], X[test_index]
                y_train, y_val = Y[train_index], Y[test_index]
                model = wresnet_ins.create_wide_residual_network(combination[2], combination[0], in_dim,combination[4], nb_classes=4, N=2, k=2, dropout=0.0)
                model.fit_generator(generator.flow(X_train, y_train, batch_size=combination[1]), steps_per_epoch=len(X_train) // combination[1], epochs=combination[3],
                                    validation_data=(X_val, y_val),
                                    validation_steps=len(X_val) // combination[1],)
                loss, acc = model.evaluate(X_test, y_test)
                metrics_dict[j+1] = {"loss": loss, "acc": acc, "epoch_stopped": combination[3]}
            if i > 95:
              row = {'momentum': combination[4],'learning rate': combination[0],
                        'batch size': combination[1], 'reg_penalty': combination[2],
                        'epoch_stopped1': metrics_dict[1]["epoch_stopped"], 
                        'loss1': metrics_dict[1]["loss"],'acc1': metrics_dict[1]["acc"],
                        'acc1': metrics_dict[1]["acc"],'epoch_stopped2': metrics_dict[2]["epoch_stopped"],
                        'loss2': metrics_dict[2]["loss"],'acc2': metrics_dict[2]["acc"],
                        'epoch_stopped3': metrics_dict[3]["epoch_stopped"],
                        'loss3': metrics_dict[3]["loss"], 'acc3': metrics_dict[3]["acc"]}
              res_df = res_df.append(row , ignore_index=True)
              res_df.to_csv(filename, sep=";")

if __name__ == "__main__":
    learning_rate = [0.1, 0.01]
    batch_size = [64,128,256]
    reg_penalty = [0.01, 0.001, 0.0001]
    epochs = [50,100,150]
    momentum = [0.7,0.6]
    in_dim = (32,32,1)
    # create list of all different parameter combinations
    param_grid = dict(learning_rate = learning_rate, batch_size = batch_size, 
                      reg_penalty = reg_penalty, epochs = epochs, momentum=momentum)
    combinations = list(product(*param_grid.values()))
    X, Y = preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.05, shuffle=True)
    print(combinations)
    instance  = ModelSelection()
    instance.KFold_GridSearchCV(in_dim,X_train,y_train,X_test, y_test, combinations, "grid_16.csv")
