import numpy as np
import matplotlib.pyplot as plt
import NeuralNet as NN
import optimisers as op
import ActivationFunctions as AF
import pandas as pd
import Data
import lrSchedules as lrs
import os

ln10 = np.log(10)

def Accuracy(predictions, targets):
    """
    Args

    predictions : (10, n_images)
    targets : (n_images, )

    Returns: (n_images, ) with 1 for correct result and 0 for false, (n_images, ) with 2 for correct result and 0 for false (not used)
    """
    numbers = np.arange(0, 10, 1)
    predicted_values = numbers[np.argmax(predictions, axis=1)]
    diff = np.equal(predicted_values, targets)
    return diff**2, diff*2

def Cross_Entropy(predictions, targets):
    """
    Args

    predictions : (10, n_images)
    targets : (n_images, )

    Returns: (n_images, ) with a values for how certain it is, (n_images, ) with derivative
    """
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

def lr_ep_error(n_epochs, nr_batches, inputs, targets, test_data, test_targets, costFunc, model):
    """
    Trains the model for a given number of epochs
    and returns the accuracy and cross entropy values

    """
    epochs = 0
    while epochs <= n_epochs:
        batches = np.array_split(inputs, nr_batches, axis=1)
        batches_targets = np.array_split(targets, nr_batches, axis=0)
        for batch_nr in range(nr_batches):
            rand_n = np.random.choice(range(nr_batches))
            model.back_propagate(batches[rand_n], batches_targets[rand_n], costFunc)
        epochs += 1
    predictions = model.feed_forward(test_data)
    Acc = Accuracy(predictions.T, test_targets)[0].mean()
    # print(predictions)
    Cross_Ent = Cross_Entropy(predictions, test_targets)[0].mean()
    return Acc, Cross_Ent

def FixedLambda(L, lr_range, ep_range, nr_batches, data, targets, test_data, test_targets, costFunc, shapes, afs, opt, Lmd, schedule, t):
    """
    For a given lambda it iterates over the learning rates and
    number of epochs and calculates the accuracy and cross entropy.

    Args:

    L: Momentum of the model (0 or False for no momentum)
    lr_range: The different learning rates to use
    ep_range: The different number og epochs to use
    ...

    Return:

    k_min: the lr_range index for lowest cross entropy
    j_min: the ep_range index for lowest cross entropy
    min: Minumum cross entropy found
    Cross_acc: The accuracy at the lowest corss entropy
    k_acc_min: the lr_range index for highest accuracy
    j_acc_min: the ep_range index for highest accuracy
    max: The highest accuracy
    acc_entropy: The cross entropy at the highest accuracy

    lr_ep_Acc: 2D array of all accuracies for the different learning rates and number of epochs
    lr_ep_Cross_ent: 2D array of all cross entropies for the different learning rates and number of epochs
    """
    min = 1e8
    max = 0

    lr_ep_Cross_ent = np.zeros((len(lr_range)+1, len(ep_range)+1))
    lr_ep_Acc = np.zeros((len(lr_range)+1, len(ep_range)+1))
    lr_ep_Cross_ent[1:, 0] = lr_range
    lr_ep_Cross_ent[0, 1:] = ep_range
    lr_ep_Acc[1:, 0] = lr_range
    lr_ep_Acc[0, 1:] = ep_range

    batches = np.array_split(data, nr_batches, axis=1)
    batches_targets = np.array_split(targets, nr_batches, axis=0)

    for k, lr in enumerate(lr_range):
        print('.', end='', flush=True)

        if L:
            optimiser = opt(lr, momentum=L)
        else:
            optimiser = opt(lr)

        if schedule:
            optimiser = op.LrScheduleOptimiser(schedule(lr, t), optimiser)

        model = NN.Model(shapes, afs, optimiser, lamda=Lmd)
        epochs = 0

        for j, n_epochs in enumerate(ep_range):
            print(',', end='', flush=True)

            while epochs <= n_epochs:
                for _ in range(nr_batches):
                    rand_n = np.random.choice(range(nr_batches))
                    model.back_propagate(batches[rand_n], batches_targets[rand_n], costFunc)
                epochs += 1
            predictions = model.feed_forward(test_data)
            Acc = Accuracy(predictions.T, test_targets)[0].mean()
            Cross_Ent = Cross_Entropy(predictions, test_targets)[0].mean()

            lr_ep_Cross_ent[k+1, j+1] = Cross_Ent
            lr_ep_Acc[k+1, j+1] = Acc

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
    return [k_min, j_min, min, Cross_acc, k_acc_min, j_acc_min, max, acc_entropy, lr_ep_Acc, lr_ep_Cross_ent]

def FixedLrEpoch(L, lr, ep, nr_batches, data, targets, test_data, test_targets, costFunc, shapes, afs, opt, Lmd_range, schedule, t):

    """
    For a given lambda it iterates over the learning rates and
    number of epochs and calculates the accuracy and cross entropy.

    Args:

    L: Momentum of the model (0 or False for no momentum)
    lr: the learning rate
    ep: the number of epochs
    ...
    Lmd_range: the different lambdas to use
    Return:

    i_min: the Lmd_range index for lowest cross entropy
    min: Minumum cross entropy found
    Cross_acc: The accuracy at the lowest cross entropy
    i_acc_min: the ep_range index for highest accuracy
    max: The highest accuracy
    acc_entropy: The cross entropy at the highest accuracy

    Lmd_Acc_Cross_ent: array with [[Lmd_range], [accuracies], [cross entropies]]
    """

    min = 1e8
    max = 0

    Lmd_Acc_Cross_ent = np.zeros((len(Lmd_range), 3))
    Lmd_Acc_Cross_ent[:, 0] = Lmd_range

    for i, Lmd in enumerate(Lmd_range):
        print('.', end='', flush=True)
        if L:
            optimiser = opt(lr, momentum=L)
        else:
            optimiser = opt(lr)

        if schedule:
            model = NN.Model(shapes, afs, op.LrScheduleOptimiser(schedule(lr, t), optimiser), lamda=Lmd)
        else:
            model = NN.Model(shapes, afs, optimiser, lamda=Lmd)
        inputs = data

        Acc, Cross_Ent = lr_ep_error(ep, nr_batches, inputs, targets, test_data, test_targets, costFunc, model)

        Lmd_Acc_Cross_ent[i, 1:3] = [Acc, Cross_Ent]

        if Acc>max:
            max = Acc
            acc_entropy = Cross_Ent
            i_acc_min = i

        if Cross_Ent<min:
            min = Cross_Ent
            Cross_acc = Acc
            i_min = i

    print('\n')
    #k = lr, j = epochs
    return [i_min, min, Cross_acc, i_acc_min, max, acc_entropy, Lmd_Acc_Cross_ent]



def Run(L, lr_range, ep_range, nr_batches, data, targets, test_data, test_targets, costFunc, shapes, afs, opt, name, Lmd, t, schedule):
    """
    Takes the parameters for FixedLambda, sends them to the function, and saves the arrays to a file each.
    """
    RL = FixedLambda(L, lr_range, ep_range, nr_batches, data, targets, test_data, test_targets, costFunc, shapes, afs, opt, Lmd, schedule, t)
    Acc_Image = RL[-2]
    Ent_Image = RL[-1]
    min = RL[2]
    max = RL[6]
    print(f'Lr: {lr_range[RL[0]]} , Epochs: {ep_range[RL[1]]}, Best Acc: {max}, Best CE: {min}')
    if name:
        os.makedirs(f'./Data/NrHidden{len(shapes)-2}/{opt.__name__}/LrEpoch', exist_ok = True)
        np.save(f'./Data/NrHidden{len(shapes)-2}/{opt.__name__}/LrEpoch/Acc_{name}', Acc_Image)
        np.save(f'./Data/NrHidden{len(shapes)-2}/{opt.__name__}/LrEpoch/Ent_{name}', Ent_Image)

def RunLambda(L, lr, ep, nr_batches, data, targets, test_data, test_targets, costFunc, shapes, afs, opt, Lmd_range, t, schedule):
    """
    Takes the parameters for FixedLrEpoch, sends them to the function, and saves the array to a file each.
    """
    RL = FixedLrEpoch(L, lr, ep, nr_batches, data, targets, test_data, test_targets, costFunc, shapes, afs, opt, Lmd_range, schedule, t)
    Acc_Ent_Image = RL[-1]
    min = RL[2]
    max = RL[4]
    print(f'Lmd: {Lmd_range[RL[0]]}, Best Acc: {max}, Best CE: {min}')
    os.makedirs(f'./Data/NrHidden{len(shapes)-2}/{opt.__name__}/Lambda', exist_ok = True)
    np.save(f'./Data/NrHidden{len(shapes)-2}/{opt.__name__}/Lambda/Acc_Ent', Acc_Ent_Image)

def SendToLrEpoch(L, t1, schedule, opt, lr_range, ep_range, Lmd, name):
    """
    Collection of often used variables to send to Run, which sends it to FixedLambda
    """
    shapes = (64, 10)
    train, test, _, train_tar, test_tar, _ = Data.load_data()
    Run(L, lr_range, ep_range, 22, train, train_tar, test, test_tar, Cross_Entropy,
        shapes, [AF.SoftMax()], opt, name, Lmd, t1, schedule)

def SendToLambda(L, Lr, ep, t1, opt, schedule):
    """
    Collection of often used variables to send to RunLambda, which sends it to FixedLrEpoch
    """
    Lmd_range = np.logspace(-5, -3, 50)
    shapes = (64, 10)
    train, test, _, train_tar, test_tar, _ = Data.load_data()
    RunLambda(L, Lr, ep, 22, train, train_tar, test, test_tar, Cross_Entropy, shapes,
              [AF.SoftMax()], opt, Lmd_range, t1, schedule)

def NetworkToLrEpoch(L, t1, schedule, opt, lr_range, ep_range, Lmd, name, af, shapes):
    train, test, _, train_tar, test_tar, _ = Data.load_data()
    Run(L, lr_range, ep_range, 22, train, train_tar, test, test_tar, Cross_Entropy,
        shapes, [af]*(len(shapes)-2) + [AF.SoftMax()], opt, name, Lmd, t1, schedule)

def NetworkToLambda(L, Lr, ep, t1, opt, schedule, af, shapes):
    Lmd_range = np.logspace(-5, -3, 50)
    train, test, _, train_tar, test_tar, _ = Data.load_data()
    RunLambda(L, Lr, ep, 22, train, train_tar, test, test_tar, Cross_Entropy, shapes,
              [af]*(len(shapes)-2)+[AF.SoftMax()], opt, Lmd_range, t1, schedule)
