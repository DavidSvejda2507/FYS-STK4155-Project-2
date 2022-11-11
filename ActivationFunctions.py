import numpy as np

def Linear():
    
    def Linear_(x):
        "Returns given value."
        return x
    
    def Derivative(x, Fx):
        return 1
    
    return Linear_, Derivative

def Sigmoid():
    
    def Sigmoid_(x):
        """
        Sigmoid activation function.

        Args: x (float) : initial output from a node
        Returns: a (float)
        """
        xexp = np.exp(x)
        out = xexp/(1+xexp)
        return out
    
    def Derivative(x, Fx):
        return Fx * (1-Fx)
    
    return Sigmoid_, Derivative

def ReLU():
    
    def ReLU_(x):
        """
        Rectified Linear Unit activation function.
        Args:
            x (float) : initial output from a node
        Returns: a (float)
        """
        return np.where(x>0, x, 0)
    
    def Derivative(x, Fx):
        return np.where(x>0, 1, 0)
    
    return ReLU_, Derivative

def LeakyReLU(alpha):
    
    def LeakyReLU_(x):
        """
        Leaky Rectified Linear Unit activation function.
        For fixing the Dying ReLU problem.
        Args:
            x (float) : initial output from a node
            alpha (float) : parameter for ouput for x < 0
        Returns: a (float)
        """
        return np.where(x>0, x, alpha*x)
    
    def Derivative(x, Fx):
        return np.where(x>0, 1, alpha)
    
    return LeakyReLU_, Derivative

def SoftMax():
    """
    Returns the Sortmax activation function and it's derivative
    The shape of the input of the softmax function is (a, b)
    With a the number of categories, and b the number of samples.
    The softmax is applied to each sample.
    """        
    def SoftMax_(x):
        exp = np.exp(x)
        Z = np.sum(exp, axis = 0)[np.newaxis, :]
        return exp/Z
    
    def Derivative(x, Fx):
        out = -Fx[:, np.newaxis] @ Fx[np.newaxis, :]
        out += np.diag(Fx)
        return Fx
    
    return SoftMax_, Derivative