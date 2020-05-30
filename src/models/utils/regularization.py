import tensorflow as tf


class regularization(object):
    def __init__(self):
        """ İnit"""
        pass
    def l2_regularization(self):
        """L2 Regularization"""
        pass
    def orthoganality_penalty(self,weight_vars, ord="fro" ):
        """orthogonality penalty"""
        def get_loss(self,w):
            """Loss calculation"""
            I = tf.eye(w.shape[-1].value)
            m = tf.reshape(w, (-1, w.shape[-1].value))
            d = None
            d = tf.norm(tf.matmul(m, m, transpose_a=True) - I, ord)
            if ord in ['fro', 2]:
                d=d**2
            return tf.reduce_sum(d)
        
        return tf.add_n(list(map(get_loss, weight_vars)))