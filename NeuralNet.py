import numpy as np



class Model():
    
    def __init__(self, shapes, AFs, optimiser):
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
        self.optimiser = optimiser
        self.optimiser.set_model(self)
    
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
    
    def back_propagate(self, input, target, costFunc):
        _, derivatives, Cost = self.layers[0].back_propagation(input, self.layers[1:], target, costFunc)
        derivatives.reverse()
        self.optimiser.update(derivatives)
        return Cost.mean()

class Layer():
    
    def __init__(self, input, output, AF):
        """Make a Neural Network layer

        Args:
            input (int): size of the input
            output (int): size of the layer
            AF ((func(np.array -> np.array)), derivative): Activation function
        """        
        self.W = np.random.rand(output, input)
        self.B = np.random.rand(output, 1)
        self.AF = AF[0]
        self.dF = AF[1]
        
    def feed_forward(self, input):
        """Feed the input through the layer

        Args:
            input (np.array): Input data with shape (size of one input, number of inputs)
        
        Returns:
            np.array: Output data
        """
        input = (self.W @ input) + self.B
        return self.AF(input)
    
    def back_propagation(self, input, layers, target, costFunc):
        """Feed the input forward and then propagate the derivative backward

        Args:
            input (np.array): Input of the layer, should have shape (size of one input, number of inputs)
            layers (List of layers): List of layers that come after this one in the network
            target (np.array): Array of the desired outputs that costFunc can use to calculate the cost
            costFunc (func(predictions, targets -> cost, derivatives)): Function to calculate the cost and the derivatives

        Returns:
            dCdIN: Derivatives of the cost function with respect to each of the inputs of the layer
            list_: List of the derivatives of all of the variables in this layer and the following layers
            Cost: Value of the cost function
        """   
        intermediate_sum = (self.W @ input) + self.B
        output = self.AF(intermediate_sum)
        
        if len(layers) > 0:
            dCdOUT, list_, Cost = layers[0].back_propagation(output, layers[1:], target, costFunc)
        else:
            list_ = []
            Cost, dCdOUT = costFunc(output, target)
        
        dCdB = self.dF(intermediate_sum, output)*dCdOUT
        dCdW = input[np.newaxis,:,:] * dCdB[:, np.newaxis, :]
        dCdIN = np.tensordot(self.W, dCdB, ((0,), (0,)))
        list_.append([dCdW, dCdB])
        return dCdIN, list_, Cost