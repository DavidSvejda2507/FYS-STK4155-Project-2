import numpy as np

def Linear(x):
    "Returns given value."
    return x

def Sigmoid(x):
    """
    Sigmoid activation function.

    Args: x (float) : initial output from a node
    Returns: a (float)
    """
    xexp = np.exp(x)
    out = xexp/(1+xexp)
    return out


def ReLU(x):
    """
    Rectified Linear Unit activation function.
    Args:
        x (float) : initial output from a node
    Returns: a (float)
    """
    return max(0, x)

def LeakyReLU(x, alpha):
    """
    Leaky Rectified Linear Unit activation function.
    For fixing the Dying ReLU problem.
    Args:
        x (float) : initial output from a node
        alpha (float) : parameter for ouput for x < 0
    Returns: a (float)
    """
    return max(alpha*x, x)
