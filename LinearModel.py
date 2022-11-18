import numpy as np
###NB: We are aware that this whole file is not strictly necessary, since we could build such a linear regression
# model using our NN class, with 1 layer and 1 node, the activation function beign the linear af, and the cost function being
# the mean square error. This code was written for didactic purposes.

class fakeLayer():
    '''mimics a layer attribute of a neural network'''
    def __init__(self, n_params):
        '''args:
            n_params: number of parameters of the model which is optimized.
        '''
        #store matrix of zeros (for imitation purpose)
        self.W = np.zeros((1,1))
        #store actual parameters which are optimized, initialize them at random.
        self.B = np.zeros((n_params,1))

class Model():
    ''' a class for a linear model, which mimics the neural network class to exploit its optimisers.
    '''
    def __init__(self, a_fake_layer, optimiser ):
        '''args:
            a_fake_layer: instance of the class fakeLayer, to mimic model of NN (this is purposefully done to use the
            optimizers which we wrote for the NN class.)
            optimiser: instance of an optimiser defined in optimisers.py
            derivative (func(beta, **kwargs)): derivative with respect to the parameters
        '''
        #create a layers attribute, in the same format of the model.layers attribute of our NeuralNet model. 
        self.layers = [ a_fake_layer, ]
        self.optimiser = optimiser
        self.optimiser.set_model(self)
            
    def MSEcost(self, beta, batchdata):
        '''
        returns the derivative of the MSE in OLS, and the MSE.
            args: 
                batchdata: batch of training examples in a dict format {X: design_matrix, y: target}
                    design_matrix: np.ndarray with 2 dimensions
                    y: np.array with 1 dimension
                betas: parameters of the linear model, in np.array (1-dim) format
            returns:
                (gradient, cost),
                gradient is inside a nested list and  
                where gradient[i,k] contains the derivative of the cost with respect to beta_k, for the i-th training example. 
        '''
        # print("\n\n\nenter MSEcost")
        X, y = batchdata.values()
        # print(f"X: {X}")##, shape y: {y.shape}, shape beta: {beta.shape}")

        #analytical expression for derivatives and cost
        diff = X @ beta - y 
        # print(f"diff : {diff}")

        gradient = X*diff
        # print(f" gradient : {gradient}")
        cost = diff*diff
        cost = np.mean(cost, axis=0)

        #want different training examples over different columns, so transpose:
        gradient = gradient.T

        #the first element in the list which is returned is a 3d np.ndarray, which mimics the format of dCdW
        #which is returned by the neuralnet's model.layer.back_propagation
        #print("exit MSE cost")
        return [[np.zeros((1,1,1)), gradient]], cost

    def gd_step(self, batchdata):
        '''perform a gradient descent step. 
            args: batchdata, in the same format as described in the docstring of MSEcost

            outputs: 
                (cost, beta): float, np.array()'''

        #compute derivatives
        betaarray = self.layers[0].B
        derivatives, cost = self.MSEcost(betaarray, batchdata)
        #print(cost)
        #at this point derivatives is:
        #a list containing one list: [[array_of_zeros, dCdB ],]
        #dCdB being a np.ndarray such that
        #dCdB[ii, jj] contains the derivative of the cost function with respect to the ii-th parameter,
        #computed at the jj-th training example. 
        self.optimiser.update_model(derivatives)
        beta = self.layers[0].B

        return cost, beta


