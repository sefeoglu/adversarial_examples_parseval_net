    
    def get_pertubation( model_type, X_train, epsilon):
        perturbation ="read from file"
        X_adv = X_train-epsilon*pertubation
        return X_adv
    def compute_perturbation(model, X, y):
        pass