""" Adversarial Training """

import numpy as np

class AdversarialTraining(object):
    """Adversarial Training  """
    def __init__(self):
        pass
    def train(self, pretrained_model, X_train, Y_train, X_test, y_test, epochs, BS, epsilon_list, sgd,  model_type = "Parseval"):
        init = (32, 32,1)
        res_df = pd.DataFrame(columns=['loss_clean','acc_clean',
                                 'loss1', 'acc1','loss2', 'acc2','loss3',
                                  'acc3','loss4', 'acc4'])

        kfold = KFold(n_splits = 10, random_state = 42, shuffle=False)
        for j, (train, val) in enumerate(kfold.split(X_train)):
          x_train, y_train = self.data_augmentation(X_train[train], Y_train[train], BS, pretrained_model, epsilon_list)
          x_val, y_val = self.data_augmentation(X_train[val], Y_train[val], BS, pretrained_model, epsilon_list)
          model = None
          if model_type != "Parseval":
              model = create_wide_residual_network(init, nb_classes=4, N=2, k=2, dropout=0.5)
          else:
              # change here for Parseval
              model = create_wide_residual_network(init, nb_classes=4, N=2, k=2, dropout=0.5)
          model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["acc"])
          hist = model.fit(generator.flow(x_train, y_train, batch_size=BS), steps_per_epoch=len(x_train) // BS, epochs=epochs,
                          callbacks = [lr_scheduler],
                          validation_data=(x_val, y_val),
                          validation_steps=x_val.shape[0] // BS,)
          loss, acc = model.evaluate(X_test, y_test)
          loss1, acc1 = print_test(model, get_adversarial_examples(pretrained_model, X_test, y_test, epsilon_list[0]),X_test, y_test, epsilon_list[0])
          loss2, acc2 = print_test(model, get_adversarial_examples(pretrained_model, X_test, y_test, epsilon_list[1]),X_test, y_test, epsilon_list[1])
          loss3, acc3 = print_test(model, get_adversarial_examples(pretrained_model, X_test, y_test, epsilon_list[2]),X_test, y_test, epsilon_list[2])
          loss4, acc4 = print_test(model, get_adversarial_examples(pretrained_model, X_test, y_test, epsilon_list[3]),X_test, y_test, epsilon_list[3])
          row = {'loss_clean':loss,'acc_clean':acc, 'loss1':loss1, 'acc1':acc1, 'loss2':loss2,
                  'acc2':acc2, 'loss3':loss3, 'acc3':acc3, 'loss4':loss4, 'acc4':acc4}
          res_df = res_df.append(row, ignore_index=True)
          
        return res_df
    def mini_batch_train(self, model, X_train, y_train, x_val, y_val, BS, pretrained_model, epsilon):


        hist = model.fit(generator.flow(X_train, y_train, batch_size=BS), steps_per_epoch=len(X_train) // BS, epochs=1,
                   validation_data=(x_val, y_val),
                   validation_steps=x_val.shape[0] // BS, shuffle = True)
        
        ### TODO ###
        ## Save hist on file.###


    def data_augmentation(self, X_train, Y_train, batch_size, pretrained_model, epsilon_list):
      ### divide data 16,16,16,16 for 4 different epsilons and 64 is true image.### 
        #start_index = self.data_iteration(X_train, batch_size)
        first_half_end = int(len(X_train)/2)
        second_half_end = int(len(X_train))
        x_clean = X_train[0: first_half_end, :, :, :]
        x_adv = self.get_adversarial(pretrained_model, X_train[first_half_end:second_half_end, :, :, :], Y_train[first_half_end:second_half_end], epsilon_list)
        x_mix = self.merge_data(x_clean, x_adv)
        y_mix = Y_train[0:second_half_end]
        ### TODO###
        # Mixture data for 4 epsilon values

        return x_mix, y_mix

    def data_iteration(self, X_train, batch_size):
        N = X_train.shape[0]
        start = np.random.randint(0, N-batch_size)
        return start

    def merge_data(self, x_clean, x_adv):
        x_mix = []
        for i in range(len(x_clean)):
          x_mix.append(x_clean[i])
        for j in range(len(x_adv)):
          x_mix.append(x_adv[j])
        x_mix = np.array(x_mix)

        return x_mix


    def get_adversarial(self, logits_model, X_true, y_true, epsilon_list):

        return self.adversarial_example(logits_model, X_true, y_true, epsilon_list)

    def adversarial_example(self, logits_model, X_true, Y_true, epsilon_list):
        size = len(X_true)
        X_adv = []
        interval = int(size/4)
        index_list = [0, interval, interval*2, interval*3, size]
        index = 0
        for epsilon in epsilon_list:
          if index == 4:
            break
          x_true = X_true[index_list[index]:index_list[index+1], :, :, :]
          y_true = Y_true[index_list[index]:index_list[index+1]]

          index = index + 1

          for i in range(len(x_true)):
            random_index = i
            original_image = x_true[random_index]
            original_image = tf.convert_to_tensor(original_image.reshape((1, 32, 32))) #The .reshape just gives it the proper form to input into the model, a batch of 1 a.k.a a tensor
            original_label = y_true[random_index]
            original_label = np.reshape(np.argmax(original_label), (1,)).astype('int64')
            adv_example_targeted_label = fast_gradient_method(logits_model, original_image, epsilon, np.inf,y=original_label, targeted=False)
            X_adv.append(np.array(adv_example_targeted_label).reshape(32, 32, 1))
          
        X_adv = np.array(X_adv)
        return X_adv
