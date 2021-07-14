from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.framework import dtypes
import numpy as _np


def convex_add(input_layer, layer_3, initial_convex_par=0.5, trainable=False):
    """
    Do a convex combination of input_layer and layer_3. That is, return the output of

        lamda* input_layer + (1 - lamda) * layer_3


    Args:
            input_layer (tf.Tensor):	Input to take convex combinatio of
            layer_3 (tf.Tensor):		Input to take convex combinatio of
            initial_convex_par (float):	Initial value for convex parameter. Must be
                                                                    in [0, 1].
            trainable (bool):			Whether convex parameter should be trainable
                                                                    or not.

    Returns:
            tf.Tensor: Result of convex combination
    """
    # Will implement this as sigmoid(p)*input_layer + (1-sigmoid(p))*layer_3 to ensure
    # convex parameter to be in the unit interval without constraints during
    # optimization

    # Find value for p, also check for legal initial_convex_par
    if initial_convex_par < 0:
        raise ValueError("Convex parameter must be >=0")

    elif initial_convex_par == 0:
        # sigmoid(-16) is approximately a 32bit roundoff error, practically 0
        initial_p_value = -16

    elif initial_convex_par < 1:
        # Compute inverse of sigmoid to find initial p value
        initial_p_value = -_np.log(1 / initial_convex_par - 1)

    elif initial_convex_par == 1:
        # Same argument as for 0
        initial_p_value = 16

    else:
        raise ValueError("Convex parameter must be <=1")

    p = variables.Variable(
        initial_value=initial_p_value, dtype=dtypes.float32, trainable=trainable
    )

    lam = math_ops.sigmoid(p)
    return input_layer * lam + (1 - lam) * layer_3
