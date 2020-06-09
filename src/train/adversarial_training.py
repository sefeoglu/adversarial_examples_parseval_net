""" Adversarial Training """

import numpy as np
class AdversarialTraining(object):
    """Adversarial Training  """
    def __init__(self):
        pass
    def train(self, model_type= "Parseval", model, X_train, Y_train, epochs, batch_size, epsilon):
        ### TODO ### 
        # Read Perturbation #
        self.X_adv = self.get_pertubation(model_type, X_train, epsilon)
        ### Function shold be written ###
        self.Y_adv = Y_train
        step_on_epoch =len(X_train[0]/batch_size)
        for epoch in range(0,epochs):
            for j in range(0,step_on_epoch):
                self.mini_batch_train(model,X_train, Y_train, batch_size)

    def mini_batch_train(self, model, X_train,Y_train, batch_size):

        x_train, y_train = self.data_augmentation(X_train, Y_train, batch_size, start_index)
        hist = model.fit(x_train, y_train, batch_size=batch_size, epochs = 1,steps_per_epoch=1)
        ### TODO ###
        ## Save hist on file.###


    def data_augmentation(self,X_train, Y_train, batch_size):
        start_index = data_iteration(X_train, batch_size)
        first_half_end = start_index+batch_size/2
        second_half_end = start_index+batch_size
        x_clean, y_clean = X_train[start_index:first_half_end,:,:,:], Y_train[start_index:first_half_end]
        x_adv, y_adv = self.get_adversarial(first_half_end,second_half_end)
        ### TODO###
        # Mixture data
        return x_mix, y_mix

    def data_iteration(self, X_train, batch_size):
        N = x_train.shape[0]
        start = np.random.randint(0, N-batch_size)
        return start

    def get_adversarial(self, first_half_end, second_half_end)
        return self.X_adv[first_half_end:second_half_end,:,:,:], self.Y_adv[first_half_end:second_half_end]

    def get_pertubation(self, model_type, X_train, epsilon):
        perturbation ="read from file"
        X_adv = X_train-epsilon*pertubation
        return X_adv