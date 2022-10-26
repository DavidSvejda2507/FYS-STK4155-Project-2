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
    
    def feed_forward(self, in_data):
        """Feed the input through the network

        Args:
            in_data (np.array): Input data

        Returns:
            np.array: Output data
        """        
        for layer in self.layers:
            in_data = layer.feed_forward(in_data)
        return in_data
    
    def back_propagate(self, in_data, target, costFunc):
        _, derivatives, Cost = self.layers[0].back_propagation(in_data, self.layers[1:], target, costFunc)
        derivatives.reverse()
        self.optimiser.update(derivatives)
        return Cost.mean()

class Layer():
    
    def __init__(self, in_data, output, AF):
        """Make a Neural Network layer

        Args:
            in_data (int): size of the input
            output (int): size of the layer
            AF ((func(np.array -> np.array)), derivative): Activation function
        """        
        self.W = np.random.rand(output, in_data)
        self.B = np.random.rand(output, 1)
        self.AF = AF[0]
        self.dF = AF[1]
        
    def feed_forward(self, in_data):
        """Feed the input through the layer

        Args:
            in_data (np.array): Input data with shape (size of one input, number of inputs)
        
        Returns:
            np.array: Output data
        """
        in_data = (self.W @ in_data) + self.B
        return self.AF(in_data)
    
    def back_propagation(self, in_data, layers, target, costFunc):
        """Feed the input forward and then propagate the derivative backward

        Args:
            in_data (np.array): Input of the layer, should have shape (size of one input, number of inputs)
            layers (List of layers): List of layers that come after this one in the network
            target (np.array): Array of the desired outputs that costFunc can use to calculate the cost
            costFunc (func(predictions, targets -> cost, derivatives)): Function to calculate the cost and the derivatives

        Returns:
            dCdIN: Derivatives of the cost function with respect to each of the inputs of the layer
            list_: List of the derivatives of all of the variables in this layer and the following layers
            Cost: Value of the cost function
        """   
        intermediate_sum = (self.W @ in_data) + self.B
        output = self.AF(intermediate_sum)
        
        if len(layers) > 0:
            dCdOUT, list_, Cost = layers[0].back_propagation(output, layers[1:], target, costFunc)
        else:
            list_ = []
            Cost, dCdOUT = costFunc(output, target)
        
        dCdB = self.dF(intermediate_sum, output)*dCdOUT
        dCdW = in_data[np.newaxis,:,:] * dCdB[:, np.newaxis, :]
        dCdIN = np.tensordot(self.W, dCdB, ((0,), (0,)))
        list_.append([dCdW, dCdB])
        return dCdIN, list_, Cost