""" Adversarial Training """

import numpy as np
from cleverhans.future.tf2.attacks import fast_gradient_method
import pandas as pd
from sklearn.model_selection import KFold
import sys
import tensorflow
import tensorflow as tf
sys.path.insert(1,'/home/sefika/AE_Parseval_Network/src/models')
from wideresnet.wresnet import WideResidualNetwork

from wresnet import WideResidualNetwork


class AdversarialTraining(object):
    """Adversarial Training  """
    def __init__(self):
        """
        """        
        self.batch_size = 64

        self.generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(
            rotation_range= 10,
            width_shift_range= 5. / 32,
            height_shift_range= 5. / 32,
        )


    def train(self, X_train, Y_train, X_test, y_test, epochs,
              BS, epsilon_list, sgd, callbacks_list):
        # init dimensions
        init = (32, 32, 1)
        # rotate the images to improve the acc.

        res_df = pd.DataFrame(columns=[
            'loss_clean', 'acc_clean', 'loss1', 'acc1', 'loss2', 'acc2',
            'loss3', 'acc3', 'loss4', 'acc4'
        ])
        # Ten fold cross validation

        kfold = KFold(n_splits=10, random_state=42, shuffle=False)
        wresnet_ins = WideResidualNetwork(0.0001, init,0.9, nb_classes=4, N=2, k=1, dropout=0.0)
        for j, (train, val) in enumerate(kfold.split(X_train)):
            
            wrn_16_1 = wresnet_ins.create_wide_residual_network()
            wrn_16_1.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["acc"])
            print("Finished compiling")
            x_train, y_train = self.data_augmentation(X_train[train],
                                                      Y_train[train], BS,
                                                      wrn_16_1,
                                                      epsilon_list)
            x_val, y_val = self.data_augmentation(X_train[val], Y_train[val],
                                                  BS, wrn_16_1,
                                                  epsilon_list)

            hist = wrn_16_1.fit(
                self.generator.flow(x_train, y_train, batch_size=BS),
                steps_per_epoch=len(x_train) // BS,
                epochs=50, callbacks = callbacks_list,
                validation_data=(x_val, y_val),
                validation_steps=x_val.shape[0] // BS,
            )
            loss, acc = wrn_16_1.evaluate(X_test, y_test)

            loss1, acc1 = self.print_test(
                wrn_16_1,
                self.get_adversarial_examples(wrn_16_1, X_test, y_test,
                                              epsilon_list[0]), X_test, y_test,
                epsilon_list[0])
            loss2, acc2 = self.print_test(
                wrn_16_1,
                self.get_adversarial_examples(wrn_16_1, X_test, y_test,
                                              epsilon_list[1]), X_test, y_test,
                epsilon_list[1])
            loss3, acc3 = self.print_test(
                wrn_16_1,
                self.get_adversarial_examples(wrn_16_1, X_test, y_test,
                                              epsilon_list[2]), X_test, y_test,
                epsilon_list[2])
            loss4, acc4 = self.print_test(
                wrn_16_1,
                self.get_adversarial_examples(wrn_16_1, X_test, y_test,
                                              epsilon_list[3]), X_test, y_test,
                epsilon_list[3])
            # store the loss and accuracy
            row = {
                'loss_clean': loss,
                'acc_clean': acc,
                'loss1': loss1,
                'acc1': acc1,
                'loss2': loss2,
                'acc2': acc2,
                'loss3': loss3,
                'acc3': acc3,
                'loss4': loss4,
                'acc4': acc4
            }
            res_df = res_df.append(row, ignore_index=True)

        return res_df

    def data_augmentation(self, X_train, Y_train, batch_size, pretrained_model,
                          epsilon_list):
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

    def print_test(self, model, X_adv, X_test, y_test, epsilon):

        loss, acc = model.evaluate(X_adv, y_test)
        print("epsilon: {} and test evaluation : {}, {}".format(
            epsilon, loss, acc))

        SNR = 20 * np.log10(
            np.linalg.norm(X_test) / np.linalg.norm(X_test - X_adv))
        print("Signal to noise ratio (SNR): {}".format(SNR))

        return loss, acc

    def get_adversarial_examples(self, pretrained_model, X_true, y_true,
                                 epsilon):
        #The attack requires the model to ouput the logits
        logits_model = tf.keras.Model(pretrained_model.input,
                                      pretrained_model.layers[-1].output)
        X_adv = []

        for i in range(len(X_true)):
            random_index = i
            original_image = X_true[random_index]
            original_image = tf.convert_to_tensor(
                original_image.reshape((1, 32, 32))
            )  #The .reshape just gives it the proper form to input into the model, a batch of 1 a.k.a a tensor
            original_label = y_true[random_index]
            original_label = np.reshape(np.argmax(original_label),
                                        (1, )).astype('int64')
            adv_example_targeted_label = fast_gradient_method(logits_model,
                                                              original_image,
                                                              epsilon,
                                                              np.inf,
                                                              y=original_label,
                                                              targeted=False)
            X_adv.append(
                np.array(adv_example_targeted_label).reshape(32, 32, 1))

        X_adv = np.array(X_adv)
        return X_adv

    def adversarial_example(self, logits_model, X_true, Y_true, epsilon_list):
        size = len(X_true)
        X_adv = []
        interval = int(size / 4)
        index_list = [0, interval, interval * 2, interval * 3, size]
        index = 0

        for epsilon in epsilon_list:

            if index == 4:
                break

            x_true = X_true[index_list[index]:index_list[index + 1], :, :, :]
            y_true = Y_true[index_list[index]:index_list[index + 1]]

            index = index + 1

            for i in range(len(x_true)):

                random_index = i
                original_image = x_true[random_index]
                original_image = tf.convert_to_tensor(
                    original_image.reshape((1, 32, 32))
                )  #The .reshape just gives it the proper form to input into the model, a batch of 1 a.k.a a tensor
                original_label = y_true[random_index]
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
