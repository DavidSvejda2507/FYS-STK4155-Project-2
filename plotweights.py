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


def weights_heat(n_epochs, optimiser, Lmd, afs, shape, name, grid_shape):
    """
    Plots a heatmap of the weights for each node in the first layer after the input layer.
    """
    model = NN.Model(shape, afs, optimiser, lamda=Lmd)
    train, test, _, train_tar, test_tar, _ = Data.load_data()
    batches = np.array_split(train, 22, axis=1)
    batches_targets = np.array_split(train_tar, 22, axis=0)

    epochs = 0
    while epochs <= n_epochs:
        for _ in range(22):
            rand_n = np.random.choice(range(22))
            model.back_propagate(batches[rand_n], batches_targets[rand_n], hp.Cross_Entropy)
        epochs += 1
    fig, axs = plt.subplots(grid_shape[0], grid_shape[1])
    for i, node in enumerate(model.layers[0].W):
        deflatten = np.reshape(node, (8, 8))
        ax = axs[int(i/grid_shape[1]), i%grid_shape[1]]
        ax.imshow(deflatten, cmap='gray')
        ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        ax.set_title(str(i))
    fig.tight_layout()
    plt.savefig(f'Data/Plots/{name}.pdf')
