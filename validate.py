import Hyper as hp
import numpy as np
import matplotlib.pyplot as plt
import NeuralNet as NN
import optimisers as op
import ActivationFunctions as AF
import pandas as pd
import Data
import lrSchedules as lrs
import os
import copy

def validate(n_epochs, optimisers, opt_names, Lmds, afs, shape, name):
    """

    Trains each optimiser 5 times with given parameters. Takes the average of
    the accuracies and cross entropies and the standard deviation for each of them,
    then plots both the accuracy and cross entropy in a errorbar plot.

    Args:

    optimisers: initialized optimisers [opt1, opt2, ...]
    opt_names: optimiser names used for x_ticks in plot
    Lmds: lambda, regularisation parameter for each optimiser
    afs: activation function for layer other then the output layer in the form: [af]
    name: filename for savefig

    """
    times = 5 #Number of times initializing and training the same model
    AccEntMeanErr = np.zeros((2, len(optimisers), 2))
    for k, optimiser in enumerate(optimisers):
        #create accuracy and validation array
        ValAccEnt = np.zeros((times, 2))
        #create and train model 5 times
        for i in range(times):
            opt_copy = copy.deepcopy(optimiser)
            model = NN.Model(shape, afs+[AF.SoftMax()], opt_copy, lamda=Lmds[k])
            train, test, val, train_tar, test_tar, val_tar = Data.load_data()
            batches = np.array_split(train, 22, axis=1)
            batches_targets = np.array_split(train_tar, 22, axis=0)

            epochs = 0
            while epochs <= n_epochs[k]:
                for _ in range(22):
                    rand_n = np.random.choice(range(22))
                    model.back_propagate(batches[rand_n], batches_targets[rand_n], hp.Cross_Entropy)
                epochs += 1

            #validate for each run
            predictions = model.feed_forward(val)
            ValAccEnt[i, 0] = hp.Accuracy(predictions.T, val_tar)[0].mean()
            ValAccEnt[i, 1] = hp.Cross_Entropy(predictions, val_tar)[0].mean()

        AccEntMeanErr[0, k, :] = np.mean(ValAccEnt[:, 0]), np.std(ValAccEnt[:, 0])
        AccEntMeanErr[1, k, :] = np.mean(ValAccEnt[:, 1]), np.std(ValAccEnt[:, 1])

    x_axis = np.arange(0, len(optimisers), 1)
    for i, title in enumerate(['Accuracy', 'Cross Entropy']):
        fig = plt.figure(figsize=(3.313, 3))
        plt.xticks(ticks=x_axis, labels=opt_names, rotation=-60)
        plt.title(title)
        y = AccEntMeanErr[i, :, 0]
        yerr = AccEntMeanErr[i, :, 1]
        plt.errorbar(x_axis, y, yerr, marker='s', mfc='red', linestyle='None')
        plt.tight_layout()
        plt.savefig(f'Data/Plots/{name}{title}.png')
        plt.show()

if __name__ == '__main__':
    #optimiser(Lr, momentum*)
    opt1 = op.LrScheduleOptimiser(lrs.hyperbolic_lr(1e-3, 215), op.Optimiser(1e-3))
    opt2 = op.LrScheduleOptimiser(lrs.hyperbolic_lr(1e-3, 464), op.MomentumOptimiser(1e-3, momentum=1.5))
    optimisers = [opt1, opt2, op.AdaGradOptimiser(1e-3), op.AdamOptimiser(1e-3), op.RMSPropOptimiser(1e-3)]
    opt_names = ['Optimiser', 'Momentum', 'AdaGrad', 'Adam', 'RMSProp']
    n_epochs = 1*np.ones(len(optimisers))
    Lmds = 1e-4*np.ones(len(optimisers))
    validate(n_epochs, optimisers, opt_names, Lmds, [], (64, 10), 'validationtest')
