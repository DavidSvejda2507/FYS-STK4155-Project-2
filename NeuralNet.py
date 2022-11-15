import numpy as np
# from numba import jit

class Model():
    def __init__(self, shapes, AFs, optimiser, lamda=0.1):
        """Make a neural network with randomised initial weights and biases.

        Args:
            shapes ([int]): Shapes of the input, each intermediate layer, and the output
            AFs [[(func(np.array -> np.array)), derivative of f],...]: Activation functions for each layer of the model
        """
        if not len(shapes) == len(AFs) + 1:
            raise ValueError("Length of shapes should be one more than the length of AFs")
        self.layers = []
        for i in range(len(shapes)-1):
            self.layers.append(Layer(shapes[i], shapes[i+1], AFs[i]))
        self.optimiser = optimiser
        self.optimiser.set_model(self)
        self.lamda = lamda

    def feed_forward(self, in_data):
        """Feed the input through the network

        Args:
            in_data (np.array): Input data

        Returns:
            np.array: Output data (output of last layer)
        """
        for layer in self.layers:
            in_data = layer.feed_forward(in_data)
        return in_data

    def back_propagate(self, in_data, target, costFunc):
        """Performs one back propagation step on the network

        Args:
            in_data (np.array): training data to be passed into the network, shape should be (a,b) where a is the siz of
                                one input and b is the number of samples
            target (np.array): Target output values for each of the inputs, the shape should be compatible with confFunc
            costFunc ((predictions, targets) -> (cost, derivatives)): Function which computes the cost function
                                and its derivatives with respect to the activations in the last layer.

        Returns:
            _type_: _description_
        """
        _, derivatives, Cost = self.layers[0].back_propagation(in_data, self.layers[1:], target, costFunc, self.lamda)
        derivatives.reverse()
        self.optimiser.update_model(derivatives)
        return Cost.mean()

# @jitclass
class Layer():

    def __init__(self, in_data: int, output: int, AF):
        """Make a Neural Network layer, initializing weights and bias from random uniform distribution.

        Args:
            in_data (int): size of the input
            output (int): size of the layer
            AF ((func(np.array -> np.array)), derivative): Activation function
        """
        self.W = np.random.normal(0,1,(output, in_data))
        self.B = np.random.normal(0,1,(output, 1))
        #set the activation function and its derivative
        self.AF = AF[0]
        self.dF = AF[1]

    def feed_forward(self, in_data):
        """Feed the input through the layer

        Args:
            in_data (np.array): Input data with shape (size of one input, number of inputs)

        Returns:
            np.array: Output data (activations of all neurons in this layer)
        """
        z_data = (self.W @ in_data) + self.B
        return self.AF(z_data)

    def back_propagation(self, in_data, layers, target, costFunc, lamda):
        """Feed the input forward and then propagate the derivative backward

        Args:
            in_data (np.array): Input of the layer, should have shape (size of one input, number of training examples)
            layers (List of layers): List of layers that come after this one in the network
            target (np.array): Array of the desired outputs that costFunc can use to calculate the cost
            costFunc ((predictions, targets) -> (cost, derivatives)): Function which computes the cost function
            and its derivatives with respect to the activations in the last layer.

        Returns:
            dCdIN: Derivatives of the cost function with respect to each of the inputs of the layer
            list_: List of the derivatives of all of the variables in this layer and the following layers
            Cost: Value of the cost function
        """

        #to understand the following line of code better, recall:
            #a row of self.W contains the weights of a specific neuron in this layer
            #a column of in_data contains the input for this layer, corresponding to a specific training example
            #self.W @ in_data is the row-column product of matrices. In the resulting matrix the following holds:
            #different rows <-> different neurons; different columns <-> different training examples.

            #self.B is a column, which means that it is added to all columns of the matrix gotten in this way.
        intermediate_sum = (self.W @ in_data) + self.B
        #intermediate_sum is a matrix, whose columns are the vectors z^l for this layer, referred to different
        #training examples. 


        #compute activation of layer, again output is a matrix.
        output = self.AF(intermediate_sum)

        if len(layers) > 0:
            dCdOUT, list_, Cost = layers[0].back_propagation(output, layers[1:], target, costFunc, lamda)
        else:
            #when the layer is the last, compute the cost function and its derivative with respect to the output
            #costFunc acts element-wise, so Cost and dCdOut are matrices
            list_ = []
            Cost, dCdOUT = costFunc(output, target)

        #dCdB is the vector {\Delta_j^l} for layer l
        dOUTdB = self.dF(intermediate_sum, output)
        if len(np.shape(dOUTdB)) == 2:
            dCdB = dOUTdB*dCdOUT
        else:

            dCdB = dOUTdB * dCdOUT[np.newaxis, :, :]
            dCdB = np.sum(dCdB, axis = 1)
        #use dCdB immediately to compute dC/dW
        dCdW = in_data[np.newaxis,:,:] * dCdB[:, np.newaxis, :]

        #dCdIn contains the derivative of the cost function with respect to the activations of the previous layer
        dCdIN = np.tensordot(self.W, dCdB, ((0,), (0,)))
        
        #add regulrisation
        dCdW += lamda*self.W[:,:,np.newaxis]
        dCdB += lamda*self.B
        list_.append([dCdW, dCdB])
        return dCdIN, list_, Cost
