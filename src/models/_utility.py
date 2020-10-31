from tensorflow.keras.callbacks import LearningRateScheduler
# Define configuration parameters
import math
def step_decay(epoch):
    """[summary]

    Args:
        epoch (int): epoch number

    Returns:
        float: new learning rate
    """
    initial_lrate = 0.1
    factor = 0.1
    if epoch < 10:
      lrate = initial_lrate
    elif epoch < 20:
      lrate = initial_lrate*math.pow(factor, 1)
    elif epoch < 30:
      lrate = initial_lrate*math.pow(factor, 2)
    elif epoch < 40 :
      lrate = initial_lrate*math.pow(factor, 3)
    else:
      lrate = initial_lrate*math.pow(factor, 4)
    return lrate



lrate = LearningRateScheduler(step_decay)