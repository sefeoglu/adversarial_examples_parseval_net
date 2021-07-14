import numpy as np


def noise(x, eps=0.3, order=np.inf, clip_min=None, clip_max=None):
    """
    A weak attack that just picks a random point in the attacker's action
    space. When combined with an attack bundling function, this can be used to
    implement random search.
    References:
    https://arxiv.org/abs/1802.00420 recommends random search to help identify
        gradient masking
    https://openreview.net/forum?id=H1g0piA9tQ recommends using noise as part
        of an attack building recipe combining many different optimizers to
        yield a strong optimizer.
    Arguments
    ---------
    x : torch.Tensor
        The input image.
    """

    if order != np.inf:
        raise NotImplementedError(ord)

    eta = np.random.uniform(low=-eps, high=eps, size=x.shape)
    adv_x = x + eta

    return adv_x
