import numpy as np
import matplotlib.pyplot as plt
import NeuralNet as NN
import optimisers as op
import ActivationFunctions as AF
import pandas as pd
import Data

ln10 = np.log(10)

def Accuracy(predictions, targets):
    numbers = np.arange(0, 10, 1)
    predicted_values = numbers[np.argmax(predictions, axis=1)]
    diff = np.equal(predicted_values, targets)
    return diff**2, diff*2

def Cross_Entropy(predictions, targets):
    predictions = predictions.T
    expanded = np.expand_dims(targets, axis=1)
    predicted_values = np.reshape(np.take_along_axis(predictions, expanded, axis=1), (len(predictions), ))
    error = -np.log10(abs(predicted_values)+1e-8)
    derror = -1/(abs(predicted_values*ln10)+1e-8)
    zeros = np.zeros((10, len(predicted_values)))
    i = 0
    for k, j in zip(derror, targets):
        zeros[j, i] = k
        i += 1
    return error, zeros

def lr_ep_error(n_epochs, inputs, targets, test_data, test_targets, costFunc, model):
    epochs = 0
    while epochs <= n_epochs:
        model.back_propagate(inputs, targets, costFunc)
        epochs += 1
    predictions = model.feed_forward(test_data)
    Acc = Accuracy(predictions.T, test_targets)[0].mean()
    Cross_Ent = Cross_Entropy(predictions, test_targets)[0].mean()
    return Acc, Cross_Ent

def FixedLambda(L, lr_range, ep_range, data, targets, test_data, test_targets, costFunc, shapes, af, opt):
    min = 1e8
    max = 0

    lr_ep_Cross_ent = np.zeros((len(lr_range), len(ep_range)))
    lr_ep_Acc = np.zeros((len(lr_range), len(ep_range)))

    for k, lr in enumerate(lr_range):
        print('.', end='', flush=True)
        for j, n_epoch in enumerate(ep_range):
            print(',', end='', flush=True)
            model = NN.Model(shapes, [af]*(len(shapes)-1), opt(lr, L))

            inputs = data

            Acc, Cross_Ent = lr_ep_error(n_epoch, inputs, targets, test_data, test_targets, costFunc, model)

            lr_ep_Cross_ent[k, j] = Cross_Ent
            lr_ep_Acc[k, j] = Acc

            if Acc>max:
                max = Acc
                acc_entropy = Cross_Ent
                k_acc_min = k
                j_acc_min = j

            if Cross_Ent<min:
                min = Cross_Ent
                Cross_acc = Acc
                k_min = k
                j_min = j

    print('\n')
    #k = lr, j = epochs
    return [k_min, j_min, min, Cross_acc, k_acc_min, j_acc_min, max, acc_entropy, lr_ep_Cross_ent, lr_ep_Cross_ent]

def FixedLrEpoch(lr, n_epoch, l_range, data, targets, test_data, test_targets, costFunc, shapes, af, opt):
    min = 1e4
    for i, L in enumerate(l_range):
        print('.', end='', flush=True)
        model = NN.Model(shapes, [af]*(len(shapes)-1), opt(lr, L))

        inputs = data

        Acc, Cross_Ent = lr_ep_error(n_epoch, inputs, targets, test_data, test_targets, costFunc, model)

        if Cross_Ent<min:
            min = Cross_Ent
            i_min = i
    print('\n')

    return i_min



def Run(L, lr_range, ep_range, data, targets, test_data, test_targets, costFunc, shapes, af, opt, name):
    RL = FixedLambda(L, lr_range, ep_range, data, targets, test_data, test_targets, costFunc, shapes, af, opt)
    print(f'Lr: {lr_range[RL[0]]} , Epochs: {ep_range[RL[1]]}')
    Acc_Image = RL[-2]
    Ent_Image = RL[-1]
    Cols = [f'{num}' for num in ep_range]
    Rows = [f'{num}' for num in lr_range]
    Acc_df = pd.DataFrame(Acc_Image, columns=Cols, index=Rows)
    Ent_df = pd.DataFrame(Ent_Image, columns=Cols, index=Rows)
    Acc_df.to_csv(f'./Data/NrHidden{len(shapes)-2}/{opt.__name__}/LrEpoch/Acc_{name}.txt')
    Ent_df.to_csv(f'./Data/NrHidden{len(shapes)-2}/{opt.__name__}/LrEpoch/Ent_{name}.txt')

L = 5
[lr_range, ep_range] = [np.logspace(-4, 0, 10), np.logspace(2, 3, 10)]
shapes = (64, 10)
train, test, val, train_tar, test_tar, val_tar = Data.load_data()
Run(L, lr_range, ep_range, train, train_tar, test, test_tar, Cross_Entropy, shapes, AF.SoftMax(), op.MomentumOptimiser, f'L_{L}')
