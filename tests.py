from math import inf
import numpy as np
import pytest
import warnings
import matplotlib.pyplot as plt

import NeuralNet as NN
import optimisers as op
import ActivationFunctions as AF
import lrSchedules as lr




def test_gradient_descent():
    from solvers import gradient_descent
    
    """
    C(beta) = 3 (beta0 - 2)**2 + (beta1 - 3)**2
    beta should converge to (2, 3)
    """
    def derivative(beta):
        return np.array((6*(beta[0]-2), 2*(beta[1]-3)))
    
    beta0 = np.array((0.,0.))
    
    
    #################### BASIC ######################
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Cause all warnings to always be triggered.
        
        
        beta, iters = gradient_descent(derivative, beta0, {}, 1)
        assert len(w) == 1
        assert "converge" in str(w[-1].message)
        beta, iters = gradient_descent(derivative, beta0, {}, 0.2)
        assert len(w) == 1
        assert np.allclose(beta, np.array((2., 3.)), 1e-5, 1e-5)
        beta, iters = gradient_descent(derivative, beta0, {}, 0.1)
        assert len(w) == 2
        assert "converge" in str(w[-1].message)
        beta, iters = gradient_descent(derivative, beta0, {}, 0.1, max_iter=100)
        assert len(w) == 2
        assert np.allclose(beta, np.array((2., 3.)), 1e-5, 1e-5)
    
    
    #################### MOMENTUM ######################
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Cause all warnings to always be triggered.
    
        beta, iters, log = gradient_descent(derivative, beta0, {}, 0.08, momentum = 1.8, log = True)
        assert len(w) == 0
        assert np.allclose(beta, np.array((2., 3.)), 1e-5, 1e-5)
        # print(beta, iters)
        # log = np.array(log)
        # plt.plot(log[:,0], log[:,1])
        # plt.savefig("test.pdf")


def test_neural_net(optimiser):
    
    
    def lr_func(count):
        return 100/(10000+count)
    
    # model = NN.Model((2, 20, 1), [AF.ReLU()]*2, op.MomentumOptimiser(0.005, 3))
    model = NN.Model((2, 20, 1), [AF.Sigmoid(), AF.Linear()], optimiser=optimiser)
    # model = NN.Model((2, 20, 2), [AF.Sigmoid(), AF.SoftMax()], optimiser=optimiser)
    
    def func(x, y):
        return np.exp(-(x*x+y*y))
        # return np.stack([np.exp(-(x*x+y*y)), 1-np.exp(-(x*x+y*y))], axis = 0)
    
    def costFunc(pred, correct):
        diff = (pred-correct)
        return diff**2, diff*2
    
    for i in range(5000):
            inputs = np.random.rand(2,1000)
            targets = func(inputs[0], inputs[1]) + np.random.normal(scale = 0.1, size = 1000)
            new_cost = model.back_propagate(inputs, targets, costFunc)
            print(new_cost)
            # print(dcost)
    # print("\n")
    # print(optimiser.count)
    # print("\n")
        
    inputs = np.random.rand(2,200)
    targets = func(inputs[0], inputs[1])
    predictions = model.feed_forward(inputs)
    print(np.sort((predictions-targets)))
    print(optimiser.__class__)
    assert np.allclose(predictions, targets, atol=0.2)
    
    
    
    
    
    
    

if __name__ == "__main__":
    test_gradient_descent()
    
    optimisers = [
        # Sigmoid Tests
        op.Optimiser(0.01, lamda = 1e-4),
        op.AdaGradOptimiser(0.05, 1e-8, lamda = 1e-4),
        op.MomentumOptimiser(0.01, momentum = 2, lamda = 1e-4),
        op.RMSPropOptimiser(0.01, 1e-8, 0.9),
        op.AdamOptimiser(0.01, 1e-8, 0.9, 0.999, lamda = 1e-4)
        
        # Softmax Tests
        # op.Optimiser(0.1, lamda = 2e-4),
        
                  ]
    for optimiser in optimisers:
        test_neural_net(optimiser)