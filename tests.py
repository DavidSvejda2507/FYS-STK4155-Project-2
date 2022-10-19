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


if __name__ == "__main__":
    test_gradient_descent()