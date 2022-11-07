import numpy as np
import matplotlib.pyplot as plt
import NeuralNet as NN
import optimisers as op
import ActivationFunctions as AF

def func(x, y):
    return np.exp(x*x+y*y)

def costFunc(pred, correct):
    diff = (pred-correct)
    return diff**2, diff*2

def lr_ep_error(n_epochs, inputs, targets, costFunc, model):
    epochs = 0
    while epochs <= n_epochs:
        model.back_propagate(inputs, targets, costFunc)
        epochs += 1
    predictions = model.feed_forward(inputs)
    error = costFunc(predictions, targets)[0].mean()
    return error

def FindOpt(l_range, lr_range, ep_range, costFunc, af, opt):
    min = 1e8
    for i, L in enumerate(l_range):
        print('\n')
        print('Completion:', i/len(l_range))
        for lr in lr_range:
            print('.', end='', flush=True)
            for n_epoch in ep_range:
                print(',', end='', flush=True)
                model = NN.Model(shapes, [af]*2, opt(lr, L))

                inputs = np.random.rand(2,200)
                targets = func(inputs[0], inputs[1])

                error = lr_ep_error(n_epoch, inputs, targets, costFunc, model)

                if error<min:
                    min = error
                    ep_min = n_epoch
                    lr_min = lr
                    L_min = L
        print('\n')
    return L_min, lr_min, ep_min, min

def FindPlateau(cutoff, l_range, lr_range, ep_range, costFunc, af, opt):
    dE = 1e8
    L_min, lr_min, ep_min, min = FindOpt(l_range, lr_range, ep_range, costFunc, af, opt)
    while dE > cutoff:
            l_range = np.linspace(0.8*L_min, 1.2*L_min, len(l_range))
            lr_range = np.linspace(0.8*lr_min, 1.2*lr_min, len(lr_range))
            ep_range = np.linspace(0.8*ep_range, 1.2*ep_range, len(ep_range))
            L_min, lr_min, ep_min, min_new = FindOpt(l_range, lr_range, ep_range, costFunc, af, opt)
            dE = abs(min-min_new)
            min = min_new
    return L_min, lr_min, ep_min
[l_range, lr_range, ep_range] = [np.linspace(1, 10, 10), np.linspace(0, 10, 10), np.logspace(1, 3, 10)]
shapes = (2, 20, 1)
print(FindOpt(l_range, lr_range, ep_range, costFunc, AF.ReLU(), op.MomentumOptimiser))
