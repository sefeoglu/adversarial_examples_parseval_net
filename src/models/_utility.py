from tensorflow.keras.callbacks import LearningRateScheduler

def exp_decay(epoch):
    """[summary]

    Args:
        epoch (int): epoch number

    Returns:
        float: new learning rate
    """
   initial_lrate = 0.1
   k = 0.1
   lrate = initial_lrate * exp(-k*t)
   return lrate


lr_scheduler = LearningRateScheduler(exp_decay)
