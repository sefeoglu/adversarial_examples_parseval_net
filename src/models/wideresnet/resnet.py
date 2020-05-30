
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

from collections import namedtuple


import tensorflow as tf
import _utils
import numpy as np
HParams = namedtuple('HParams',
                    'batch_size, num_classes, num_residual_units, k, '
                    'weight_decay, initial_lr, decay_step, lr_decay, '
                    'momentum')
class WideResNet(object):
    """ Wide Residual Network"""
    def __init__(self, hp, images,labels, global_steps):
        """init the wide resnet"""
        self._hp = hp # hyperparameters
        self._images = images #input images
        self._labels = labels #output labels
        self._global_step = global_steps
        self.is_train = tf.placeholder(tf.bool)

    def build_model(self):
        """build resnet"""
        print("Building model....")
        print('\n Building unit init_conv')
        x = _utils._conv(self._images, 68, 100, 1, name="init_conv")
       
        #Residual Blocks
        filters = [16,16*self._hp.k, 32*self._hp.k, 64*self._hp.k]
        strides = [1, 2, 2]
        
        for i in range(1, 4):
            # First residual unit
            with tf.variable_scope('unit_%d_0' % i) as scope:
                print('\tBuilding residual unit: %s' % scope.name)
                x = _utils._bn(x, self.is_train, self._global_step, name='bn_1')
                x = _utils._relu(x, name='relu_1')

                # Shortcut
                if filters[i-1] == filters[i]:
                    if strides[i-1] == 1:
                        shortcut = tf.identity(x)
                    else:
                        shortcut = tf.nn.max_pool(x, [1, strides[i-1], strides[i-1], 1],
                                                  [1, strides[i-1], strides[i-1], 1], 'VALID')
                else:
                    shortcut = _utils._conv(x, 1, filters[i], strides[i-1], name='shortcut')

                # Residual
                x = _utils._conv(x, 3, filters[i], strides[i-1], name='conv_1')
                x = _utils._bn(x, self.is_train, self._global_step, name='bn_2')
                x = _utils._relu(x, name='relu_2')
                x = _utils._conv(x, 3, filters[i], 1, name='conv_2')

                # Merge
                x = x + shortcut
            # Other residual units
            for j in range(1, self._hp.num_residual_units):
                with tf.variable_scope('unit_%d_%d' % (i, j)) as scope:
                    print('\tBuilding residual unit: %s' % scope.name)
                    # Shortcut
                    shortcut = x

                    # Residual
                    x = _utils._bn(x, self.is_train, self._global_step, name='bn_1')
                    x = _utils._relu(x, name='relu_1')
                    x = _utils._conv(x, 3, filters[i], 1, name='conv_1')
                    x = _utils._bn(x, self.is_train, self._global_step, name='bn_2')
                    x = _utils._relu(x, name='relu_2')
                    x = _utils._conv(x, 3, filters[i], 1, name='conv_2')

                    # Merge
                    x = x + shortcut

        # Last unit
        with tf.variable_scope('unit_last') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = _utils._bn(x, self.is_train, self._global_step)
            x = _utils._relu(x)
            x = tf.reduce_mean(x, [1, 2])

        # Logit
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x_shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, x_shape[1]])
            x = _utils._fc(x, self._hp.num_classes)

        self._logits = x

        # Probs & preds & acc
        self.probs = tf.nn.softmax(x, name='probs')
        self.preds = tf.to_int32(tf.argmax(self._logits, 1, name='preds'))
        ones = tf.constant(np.ones([self._hp.batch_size]), dtype=tf.float32)
        zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
        correct = tf.where(tf.equal(self.preds, self._labels), ones, zeros)
        self.acc = tf.reduce_mean(correct, name='acc')
        tf.summary.scalar('accuracy', self.acc)

        # Loss & acc
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=self._labels)
        self.loss = tf.reduce_mean(loss, name='cross_entropy')
        tf.summary.scalar('cross_entropy', self.loss)


    def build_train_op(self):
        # Add l2 loss
        with tf.variable_scope('l2_loss'):
            costs = [tf.nn.l2_loss(var) for var in tf.get_collection(_utils.WEIGHT_DECAY_KEY)]
            # for var in tf.get_collection(utils.WEIGHT_DECAY_KEY):
                # tf.summary.histogram(var.op.name, var)
            l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
        self._total_loss = self.loss + l2_loss

        # Learning rate
        self.lr = tf.train.exponential_decay(self._hp.initial_lr, self._global_step,
                                        self._hp.decay_step, self._hp.lr_decay, staircase=True)
        tf.summary.scalar('learing_rate', self.lr)

        # Gradient descent step
        opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        grads_and_vars = opt.compute_gradients(self._total_loss, tf.trainable_variables())
        # print('\n'.join([t.name for t in tf.trainable_variables()]))
        apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

        # Batch normalization moving average update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            with tf.control_dependencies(update_ops+[apply_grad_op]):
                self.train_op = tf.no_op()
        else:
            self.train_op = apply_grad_op
if __name__ == "__main__":
    init_step = 0
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Build a Graph that computes the predictions from the inference model.
    images = tf.placeholder(tf.float32, [128, 100, 68, 1])
    labels = tf.placeholder(tf.int32, [128])
    hp = None
    network = WideResNet(hp, images, labels, global_step)
    
    network.build_model()
#    network.build_train_op()
#
#    # Summaries(training)
#    train_summary_op = tf.summary.merge_all()