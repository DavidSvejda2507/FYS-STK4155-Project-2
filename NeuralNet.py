import numpy as np



class Model():
    
    def __init__(self, shapes, AFs):
        """Make a neural network with randomised initial 

        Args:
            shapes ([int]): Shapes of the input, each intermediate layer, and the output
            AFs ([(func(np.array -> np.array)), derivative of f]): Activation functions for each layer of the model
        """
        if not len(shapes) == len(AFs) + 1:
            raise ValueError("Length of shapes should be one more than the length of AFs")
        self.layers = []
        for i in range(len(shapes)-1):
            self.layers.append(Layer(shapes[i], shapes[i+1], AFs[i]))
    
    def feed_forward(self, input):
        """Feed the input through the network

        Args:
            input (np.array): Input data

        Returns:
            np.array: Output data
        """        
        for layer in self.layers:
            input = layer.feed_forward(input)
        return input

class Layer():
    
    def __init__(self, input, output, AF):
        """Make a Neural Network layer

        Args:
            input (int): size of the input
            output (int): size of the layer
            AF ((func(np.array -> np.array)), derivative): Activation function
        """        
        self.W = np.random((output, input))
        self.B = np.random((output))
        self.AF = AF[0]
        self.dF = AF[1]
        
    def feed_forward(self, input):
        """Feed the input through the layer

        Args:
            input (np.array): Input data
        
        Returns:
            np.array: Output data
        """
        input = (self.W @ input) + self.B
        return self.AF(input)
    
    def back_propagation(self, input, layers, target, costFunc):
        """Feed the input forward and then propagate the result backward

        Args:
            output (np.array): Array with the shape of self.B

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """        
        intermediate_sum = (self.W @ input) + self.B
        output = self.AF(intermediate_sum)
        if len(layers) > 0:
            dCdOUT, list_, result = layers[0].back_propagation(output, layers[1:], target, costFunc)
        else:
            result = output
            list_ = []
            dCdOUT = costFunc(output, target)
        raise NotImplementedError()
        return dCdOUT, [(dW, dB), ...], result