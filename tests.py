from math import inf
import numpy as np
import pytest
import warnings
import matplotlib.pyplot as plt




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

def test_neural_net():
    import NeuralNet as NN
    import optimisers as op
    import ActivationFunctions as AF
    
    # model = NN.Model((2, 20, 1), [AF.ReLU()]*2, op.MomentumOptimiser(0.005, 3))
    model = NN.Model((2, 20, 1), [AF.ReLU()]*2, op.Optimiser(0.01))
    
    def func(x, y):
        return np.exp(x*x+y*y)
    
    def costFunc(pred, correct):
        diff = (pred-correct)
        return diff**2, diff*2
    
    inputs = np.random.rand(2,200)
    targets = func(inputs[0], inputs[1])
    
    cost = 1e3
    dcost = 1
    while(dcost>1e-10):
        new_cost = model.back_propagate(inputs, targets, costFunc)
        dcost = 0.5*dcost + cost-new_cost
        cost = new_cost
        # print("#", end = "", flush=True)
        print(cost)
        print(dcost)
    print("\n")
        
    predictions = model.feed_forward(inputs)
    print(predictions-targets)
    assert np.allclose(predictions, targets, atol=0.1)
    
    
    
    
    
    
    

if __name__ == "__main__":
    test_gradient_descent()
    test_neural_net()