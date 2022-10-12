import numpy as np

class Node():
    """
    Creates a node.

    Args:
        w (np.arraylike) : weights
        b (float) : bias
        af (function) : activation function
    """
    def __init__(self, w, b, af):
        self.w = w
        self.b = b
        self.af = af

    def output(self, input):
        """
        Returns weighted sum of inputs pluss bias.

        Args:

        input (np.arraylike) : array of input with length len(w)
        """
        sum = np.dot(self.w, input) + self.b
        self.out = self.af(sum)
        return self.out
