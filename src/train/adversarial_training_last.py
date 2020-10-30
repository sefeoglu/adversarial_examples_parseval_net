pip install -qq -e git+http://github.com/tensorflow/cleverhans.git#egg=cleverhans
import sys
sys.path.append('/content/src/cleverhans')
import cleverhans

import numpy as np
import math
import tensorflow as tf
from cleverhans.future.tf2.attacks import fast_gradient_method
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_epochs', 200, 'Number of epochs.')
flags.DEFINE_float('eps', 0.05, 'Total epsilon for FGM and PGD attacks.')
def adversarial_train(base_model, X_train, y_train, X_test, y_test):
    loss_object = tf.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.SGD(learning_rate=0.01)


    # Metrics to track the different accuracies
    train_loss = tf.metrics.Mean(name="train_loss")
    test_acc_clean = tf.metrics.CategoricalAccuracy()
    test_acc_fgsm = tf.metrics.CategoricalAccuracy()

    @tf.function
    def train_step(x,y):
      with tf.GradientTape() as tape:
        predictions = base_model(x)
        loss = loss_object(y, predictions)
      gradients = tape.gradient(loss, base_model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, base_model.trainable_variables))
      train_loss(loss)
    

    #Train model with adversarial training
    for epoch in range(FLAGS.nb_epochs):
      # keras like display of process
      progress_bar_train = tf.keras.utils.Progbar(50000)
      for (x,y) in data.train:
        x = fast_gradient_method(base_model, x, FLAGS.eps,np.inf)
        train_step(x,y)
        progress_bar_train.add(x.shape[0], values=[('loss', train_loss.result())])
    #Evaluate on clean and adversarial data
    progress_bar_test = tf.keras.utils.Progbar(10000)
    for x, y in data_test:
      y_pred = base_model(x)
      test_acc_clean(y, y_pred)

      x_fgsm = fast_gradient_method(base_mode(l, x, eps=FLAGS.eps, np.inf)
      y_pred_fgsm = base_model(x_fgsm)
      test_acc_fgsm(y, y_pred_fgsm)

      progress_bar_test.add(x.shape[0])
    
    print("test acc on the clean examples(%):{:3f}".format(test_acc_clean.result()*100))
    print("test acc FGSM adversarial examples(%):{:3f}".format(test_acc_fgsm.result()*100))


