import numpy as np
from warnings import warn



def convergence_warning(grad, eta):
    """Issue a convergence warning

    Args:
        grad (float): Gradient that was achieved.
        eta (float): Convergence cut off.
    """    
    warn(f"Gradient descent didn't converge, stepsize was {grad} , limit is set to {eta}.", stacklevel=2)

def gradient_descent(derivative, beta0, kwargs, lr, eta = 1e-8, epochs = 50, momentum = 1, log = False):
    """Perform gradient descent to optimise beta according to the derivatives of the cost function as defined by derivative

    Args:
        derivative (func(beta, **kwargs)): derivative of beta.
        beta0 (np.arraylike): initial value of the parameters to be optimised.
        kwargs (dict): kwargs for derivative function, such as data.
        lr (float): learning rate.
        eta (float, optional): Max error. Defaults to 1e-8.
        epochs (int, optional): Max number of steps. Defaults to 50.
        momentum (float, optional): Size of the momentum effect. Defaults to 1.
        log (Bool, optional): Set to True to keep track of a log of the beta values. Defaults to False.
    """    
    
    beta = np.array(beta0)
    cary = 1 - 1./momentum
    d_beta = 0
    if log: log_list = [beta.copy()]
    iters = 0
    while True:
        d_beta = lr * derivative(beta, **kwargs) + cary * d_beta
        beta -= d_beta
        iters += 1
        if log: log_list.append(beta.copy())
        if (np.linalg.norm(d_beta) < eta): break
        if (iters >= epochs):
            convergence_warning(np.linalg.norm(d_beta), eta)
            break
            
    # print(beta, iters)
    if log: return beta, iters, log_list
    return beta, iters


#### In Development #####

def stochastic_gradient_descent(derivative, beta0, kwargs, kwarg_splitting_func, lr, eta = 1e-8, epochs = 50, momentum = 1, log = False):
    """Perform gradient descent to optimise beta according to the derivatives of the cost function as defined by derivative

    Args:
        derivative (func(beta, **kwargs)): derivative of beta.
        beta0 (np.arraylike): initial value of the parameters to be optimised.
        kwargs (dict): kwargs for derivative function, such as data.
        lr (float): learning rate.
        eta (float, optional): Max error. Defaults to 1e-8.
        epochs (int, optional): Max number of steps. Defaults to 50.
        momentum (float, optional): Size of the momentum effect. Defaults to 1.
        log (Bool, optional): Set to True to keep track of a log of the beta values. Defaults to False.
    """    
    
    beta = np.array(beta0)
    cary = 1 - 1./momentum
    d_beta = 0
    if log: log_list = [beta.copy()]
    iters = 0
    while True:
        for kwarg in kwarg_splitting_func(kwargs):
            d_beta = lr * derivative(beta, **kwarg) + cary * d_beta
            beta -= d_beta
        iters += 1
        if log: log_list.append(beta.copy())
        if (np.linalg.norm(d_beta) < eta): break
        if (iters >= epochs):
            convergence_warning(np.linalg.norm(d_beta), eta)
            break
            
    # print(beta, iters)
    if log: return beta, iters, log_list
    return beta, iters
    