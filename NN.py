import numpy as np

class Model():
    """
    Creates a model with layers of nodes.

    Args:
        W (list of np.arraylike) : weights for each node in each layer
        B (list of np.arraylike) : Bias for each noede in each layer
        AF (list of tuples of functions(np.array) and their derivatives) : Activation function for each layer
    """
    def __init__(self, W, B, AF):
        self.layers = np.array(\
        [Layer(W_i, B_i, AF_i) for W_i, B_i, AF_i in zip(W, B, AF)])

    def Feed_forward(self, input_layer):
        """
        Sends input through the model and returns the final output layer

        Args:
            input_layer ((n_input, )) : initial inputs
        """
        for layer in self.layers:
            input_layer = layer.output(input_layer)
        return input_layer

class Layer():
    """
    Creates an array of nodes in Layer object.

    Args:
        w (np.arraylike (n_nodes, n_weights)) : weights for each node
        b (np.arraylike (n_nodes, )) : bias for each node
        af (function) : activation function for each node in the layer

    """
    def __init__(self, w, b, af):
        self.nodes = np.array([Node(w_i, b_i, af) for w_i, b_i in zip(w, b)])
        self.n_nodes = len(self.nodes)
        self.w = w
        self.b = b
        self.af = af

    def output(self, input):
        """
        Returns the output of the whole layer

        Args:
            input (np.arraylike (n_nodes_prev, )) : output from previous layer
        """
        out = self.af(np.dot(self.w, input) + self.b)
        return out

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
