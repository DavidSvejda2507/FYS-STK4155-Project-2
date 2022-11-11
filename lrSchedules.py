import numpy as np
from numba import jit

def hyperbolic_lr(lr: float, t1: int):
    """Makes a hyperbolic learning rate schedule

    Args:
        lr (float): initial learning rate
        t1 (int): Number of steps until the learning rate is halved
    """    
    @jit
    def lr_schedule(count: int):
        return (lr*t1)/(t1+count)
    return lr_schedule


def linear_lr(lr: float, t1: int):
    """Makes a linear learning rate schedule. Beware, if count ever exceeds t1 the learningrate becomes negative

    Args:
        lr (float): initial learning rate
        t1 (int): number of learning steps to be done
    """    
    @jit
    def lr_schedule(count: int):
        return lr*(t1-count+1)/t1
    return lr_schedule


def exponential_lr(lr, t1):
    @jit
    def lr_schedule(count):
        return lr * np.exp(-count/t1)
    return lr_schedule

