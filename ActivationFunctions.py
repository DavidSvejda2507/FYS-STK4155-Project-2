import numpy as np
from numba import jit

def Linear():

    @jit(nopython=True)
    def Linear_(x: np.array) -> np.array:
        "Returns given value."
        return x

    @jit(nopython=True)
    def Derivative(x: np.array, Fx: np.array) -> np.array:
        return 1

    return Linear_, Derivative

def Sigmoid():

    @jit(nopython=True)
    def Sigmoid_(x: np.array) -> np.array:
        """
        Sigmoid activation function.

        Args: x (float) : initial output from a node
        Returns: a (float)
        """
        xexp = np.exp(x)
        out = xexp/(1+xexp)
        return out

    @jit(nopython=True)
    def Derivative(x: np.array, Fx: np.array) -> np.array:
        return Fx * (1-Fx)

    return Sigmoid_, Derivative

def ReLU():

    @jit(nopython=True)
    def ReLU_(x: np.array) -> np.array:
        """
        Rectified Linear Unit activation function.
        Args:
            x (float) : initial output from a node
        Returns: a (float)
        """
        return np.where(x>0, x, 0)

    @jit(nopython=True)
    def Derivative(x: np.array, Fx: np.array) -> np.array:
        return np.where(x>0, 1, 0)

    return ReLU_, Derivative

def LeakyReLU(alpha):

    @jit(nopython=True)
    def LeakyReLU_(x: np.array) -> np.array:
        """
        Leaky Rectified Linear Unit activation function.
        For fixing the Dying ReLU problem.
        Args:
            x (float) : initial output from a node
            alpha (float) : parameter for ouput for x < 0
        Returns: a (float)
        """
        return np.where(x>0, x, alpha*x)

    @jit(nopython=True)
    def Derivative(x: np.array, Fx: np.array) -> np.array:
        return np.where(x>0, 1, alpha)

    return LeakyReLU_, Derivative

def SoftMax():
    """
    Returns the Sortmax activation function and it's derivative
    The shape of the input of the softmax function is (a, b)
    With a the number of categories, and b the number of samples.
    The softmax is applied to each sample.
    """    
    @jit(nopython=True)
    def SoftMax_(x: np.array) -> np.array:
        exp = np.exp(x)
        Z = np.sum(exp, axis = 0)
        return exp/Z

    # @jit(nopython=True)
    def Derivative(x: np.array, Fx: np.array) -> np.array:
        out = -Fx[:, np.newaxis] * Fx[np.newaxis, :]
        diag = np.arange(Fx.shape[0])
        out[diag, diag, :] += Fx
        return out
        # out = np.zeros((Fx.shape[0], Fx.shape[0], Fx.shape[1]), dtype = float)
        # for i in range(out.shape[2]):
        #     out[:,:,i] = np.outer(Fx[:,i], Fx[:,i])
        # for i in range(out.shape[2]):
        #     out[:,:,i] += np.diag(Fx[:,i])
        # return out

    return SoftMax_, Derivative
