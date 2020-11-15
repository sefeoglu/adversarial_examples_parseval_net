import numpy as np
from cleverhans.future.tf2.attacks import fast_gradient_method
import pandas as pd
from sklearn.model_selection import KFold
import sys
import tensorflow
import tensorflow as tf
#sys.path.insert(1, '/home/sefika/AE_Parseval_Network/src/models')


from _utility import print_test, get_adversarial_examples

import pickle

class AdversarialTraining(object):
    """[summary]

    Args:
        object ([type]): [description]
    """
    def __init__(self, parameter):
        self.epochs = parameter['epochs']
        self.batch_size = parameter['batch_size']
        self.optimizer = parameter['optimizer']

        self.generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=5. / 32,
            height_shift_range=5. / 32,
        )

    def train(self,
              instance,premodel,
              X_train,
              Y_train,
              X_test,
              y_test,
              epsilon_list,
              callbacks_list,
              model_name="ResNet"):
        """[summary]

        Args:
            instance ([type]): [description]
            X_train ([type]): [description]
            Y_train ([type]): [description]
            X_test ([type]): [description]
            y_test ([type]): [description]
            epsilon_list ([type]): [description]
            callbacks_list ([type]): [description]
            model_name (str, optional): [description]. Defaults to "ResNet".

        Returns:
            [type]: [description]
        """

        res_df = pd.DataFrame(columns=["loss_clean","acc_clean","0.003_loss",
                                       "0.003_acc","0.005_loss","0.005_acc",
                                       "0.02_acc","0.02_loss", "0.01_acc","0.01_loss"])
        # Ten fold cross validation

        kfold = KFold(n_splits=10, random_state=42, shuffle=False)

        for j, (train, val) in enumerate(kfold.split(X_train)):

            model = instance.create_wide_residual_network()
            model.compile(loss="categorical_crossentropy",
                          optimizer=self.optimizer,
                          metrics=["acc"])
            print("Finished compiling")
            print(epsilon_list)
            x_train, y_train = X_train[train],Y_train[train]
            x_val, y_val = X_train[val], Y_train[val]
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_dataset = train_dataset.batch(self.batch_size)
            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            val_dataset = val_dataset.batch(self.batch_size)

            for epoch in range(self.epochs):

                for step, (x_train, y_train) in enumerate(train_dataset):
                    x_train = self.data_augmentation(x_train, y_train, model, epsilon_list)
                    hst = model.fit(self.generator.flow(x_train, y_train, self.batch_size), batch_size=self.batch_size)
                
                for step, (x_val, y_val) in enumerate(val_dataset):
                    x_val = self.data_augmentation(x_val, y_val, model, epsilon_list)
                    hst = model.evaluate(x_val, y_val)
            with open(
                    'history_adv_'
                    + model_name + str(j), 'wb') as file_pi:
                pickle.dump(hist.history, file_pi)

            loss, acc = model.evaluate(X_test, y_test)

            loss1, acc1 = print_test(premodel,get_adversarial_examples(model, X_test, y_test,
                                                                    epsilon_list[0]),
                                     X_test, y_test, epsilon_list[0])
            loss2, acc2 = print_test(premodel, get_adversarial_examples(model, X_test, y_test,
                                                                     epsilon_list[1]),
                                     X_test, y_test, epsilon_list[1])
            loss3, acc3 = print_test(premodel,
                                     get_adversarial_examples(model, X_test, y_test,
                                                              epsilon_list[2]),
                                     X_test, y_test, epsilon_list[2])
            loss4, acc4 = print_test(premodel, get_adversarial_examples(model, X_test, y_test,
                                                                     epsilon_list[3]),
                                     X_test, y_test,epsilon_list[3])
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
               "0.01_loss":loss4
               }
            res_df = res_df.append(row, ignore_index=True)

        return res_df

    def data_augmentation(self, X_train, Y_train, pretrained_model,
                          epsilon_list):
        """[summary]

        Args:
            X_train ([type]): [description]
            Y_train ([type]): [description]
            pretrained_model ([type]): [description]
            epsilon_list ([type]): [description]

        Returns:
            [type]: [description]
        """
        first_half_end = int(len(X_train) / 2)
        second_half_end = int(len(X_train))
        x_clean = X_train[0:first_half_end, :, :, :]
        x_adv = self.get_adversarial(
            pretrained_model, X_train[first_half_end:second_half_end, :, :, :],
            Y_train[first_half_end:second_half_end], epsilon_list)
        x_mix = self.merge_data(x_clean, x_adv)
        y_mix = Y_train[0:second_half_end]

        return x_mix, y_mix

    def merge_data(self, x_clean, x_adv):
        """[summary]

        Args:
            x_clean ([type]): [description]
            x_adv ([type]): [description]

        Returns:
            [type]: [description]
        """
        x_mix = []
        for i in range(len(x_clean)):
            x_mix.append(x_clean[i])
        for j in range(len(x_adv)):
            x_mix.append(x_adv[j])
        x_mix = np.array(x_mix)

        return x_mix

    def get_adversarial(self, logits_model, X_true, y_true, epsilon_list):
        return self.adversarial_example(logits_model, X_true, y_true,
                                        epsilon_list)

    def adversarial_example(self, logits_model, X_true, y_true, epsilon_list):
        X_adv = []

        for index, x_true  in enumerate(X_true):
            epsilon = epsilon_list[index]
          
            original_image = x_true
            original_image = tf.reshape(original_image, (1, 32, 32))
            original_label = y_true[index]
            original_label = np.reshape(np.argmax(original_label),
                                      (1, )).astype('int64')
            adv_example_targeted_label = fast_gradient_method(
                logits_model,
                original_image,
                epsilon,
                np.inf,
                y=original_label,
                targeted=False)
            X_adv.append(
                np.array(adv_example_targeted_label).reshape(32, 32, 1))
        X_adv = np.array(X_adv)
        return X_adv