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


def weights_heat(Lr, n_epochs, opt, Lmd):

    optimiser = opt(Lr)
    model = NN.Model((64, 10), [AF.SoftMax()], optimiser, lamda=Lmd)
    train, test, _, train_tar, test_tar, _ = Data.load_data()
    batches = np.array_split(train, 22, axis=1)
    batches_targets = np.array_split(train_tar, 22, axis=0)

    epochs = 0
    while epochs <= n_epochs:
        for _ in range(22):
            rand_n = np.random.choice(range(22))
            model.back_propagate(batches[rand_n], batches_targets[rand_n], hp.Cross_Entropy)
        epochs += 1

    fig, axs = plt.subplots(2, 5)
    for i, node in enumerate(model.layers[-1].W):
        deflatten = np.reshape(node, (8, 8))
        ax = axs[int(i/5), i%5]
        ax.imshow(deflatten, cmap='gray')
        ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        ax.set_title(str(i))
    fig.tight_layout()
    plt.savefig(f'Data/Plots/{opt.__name__}Weights.pdf')
