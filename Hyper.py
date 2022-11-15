import numpy as np
import matplotlib.pyplot as plt
import NeuralNet as NN
import optimisers as op
import ActivationFunctions as AF
import pandas as pd
import Data
import lrSchedules as lrs

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

def lr_ep_error(n_epochs, nr_batches, inputs, targets, test_data, test_targets, costFunc, model):
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
    Cross_Ent = Cross_Entropy(predictions, test_targets)[0].mean()
    return Acc, Cross_Ent

def FixedLambda(L, lr_range, ep_range, nr_batches, data, targets, test_data, test_targets, costFunc, shapes, af, opt, Lmd, schedule, t):
    min = 1e8
    max = 0

    lr_ep_Cross_ent = np.zeros((len(lr_range)+1, len(ep_range)+1))
    lr_ep_Acc = np.zeros((len(lr_range)+1, len(ep_range)+1))
    lr_ep_Cross_ent[1:, 0] = lr_range
    lr_ep_Cross_ent[0, 1:] = ep_range
    lr_ep_Acc[1:, 0] = lr_range
    lr_ep_Acc[0, 1:] = ep_range

    for k, lr in enumerate(lr_range):
        print('.', end='', flush=True)
        for j, n_epoch in enumerate(ep_range):
            print(',', end='', flush=True)
            if L:
                optimiser = opt(lr, momentum=L)
            else:
                optimiser = opt(lr)

            if schedule:
                model = NN.Model(shapes, [af]*(len(shapes)-1), op.LrScheduleOptimiser(schedule(lr, t), optimiser), lamda=Lmd)
            else:
                model = NN.Model(shapes, [af]*(len(shapes)-1), optimiser, lamda=Lmd)
            inputs = data

            Acc, Cross_Ent = lr_ep_error(n_epoch, nr_batches, inputs, targets, test_data, test_targets, costFunc, model)

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
    #k = lr, j = epochs
    return [k_min, j_min, min, Cross_acc, k_acc_min, j_acc_min, max, acc_entropy, lr_ep_Acc, lr_ep_Cross_ent]

def FixedLrEpoch(L, lr, ep, nr_batches, data, targets, test_data, test_targets, costFunc, shapes, af, opt, Lmd_range, schedule, t):
    min = 1e8
    max = 0

    Lmd_Cross_ent = np.zeros((len(Lmd_range)+1, 2))
    Lmd_Acc = np.zeros((len(Lmd_range)+1, 2))
    Lmd_Cross_ent[1:, 0] = Lmd_range
    Lmd_Acc[1:, 0] = Lmd_range

    for i, Lmd in enumerate(Lmd_range):
        print('.', end='', flush=True)
        if L:
            optimiser = opt(lr, momentum=L)
        else:
            optimiser = opt(lr)

        if schedule:
            model = NN.Model(shapes, [af]*(len(shapes)-1), op.LrScheduleOptimiser(schedule(lr, t), optimiser), lamda=Lmd)
        else:
            model = NN.Model(shapes, [af]*(len(shapes)-1), optimiser, lamda=Lmd)
        inputs = data

        Acc, Cross_Ent = lr_ep_error(ep, nr_batches, inputs, targets, test_data, test_targets, costFunc, model)

        Lmd_Cross_ent[i, 1] = Cross_Ent
        Lmd_Acc[i, 1] = Acc

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
    return [i_min, min, Cross_acc, i_acc_min, max, acc_entropy, Lmd_Acc, Lmd_Cross_ent]



def Run(L, lr_range, ep_range, nr_batches, data, targets, test_data, test_targets, costFunc, shapes, af, opt, name, Lmd, t, schedule):
    RL = FixedLambda(L, lr_range, ep_range, nr_batches, data, targets, test_data, test_targets, costFunc, shapes, af, opt, Lmd, schedule, t)
    Acc_Image = RL[-2]
    Ent_Image = RL[-1]
    min = RL[2]
    max = RL[6]
    print(f't1: {t}, Lr: {lr_range[RL[0]]} , Epochs: {ep_range[RL[1]]}, Best Acc: {max}, Best CE: {min}')
    np.save(f'./Data/NrHidden{len(shapes)-2}/{opt.__name__}/LrEpoch/Acc_{name}', Acc_Image)
    np.save(f'./Data/NrHidden{len(shapes)-2}/{opt.__name__}/LrEpoch/Ent_{name}', Ent_Image)

def RunLambda(L, lr, ep, nr_batches, data, targets, test_data, test_targets, costFunc, shapes, af, opt, name, Lmd_range, t, schedule):
    RL = FixedLrEpoch(L, lr, ep, nr_batches, data, targets, test_data, test_targets, costFunc, shapes, af, opt, Lmd_range, schedule, t)
    Acc_Image = RL[-2]
    Ent_Image = RL[-1]
    min = RL[2]
    max = RL[4]
    print(f't1: {t}, Lmd: {Lmd_range[RL[0]]}, Best Acc: {max}, Best CE: {min}')
    np.save(f'./Data/NrHidden{len(shapes)-2}/{opt.__name__}/Lambda/Acc_{name}', Acc_Image)
    np.save(f'./Data/NrHidden{len(shapes)-2}/{opt.__name__}/Lambda/Ent_{name}', Ent_Image)

def SendToLrEpoch():
    Llist = [0, 1.5, 0, 0, 0]
    Lmd = 1e-4
    [lr_range, ep_range] = [np.logspace(-4, 0, 10), np.logspace(2, 3, 10)]
    shapes = (64, 10)
    train, test, val, train_tar, test_tar, val_tar = Data.load_data()
    schedules = [lrs.hyperbolic_lr, lrs.hyperbolic_lr, None, None, None]
    opts = [op.Optimiser, op.MomentumOptimiser, op.AdaGradOptimiser, op.AdamOptimiser, op.RMSPropOptimiser]
    t1 = [215, 464, None, None, None]
    for n_opt, opt in enumerate(opts):
        L = Llist[n_opt]
        schedule = schedules[n_opt]
        t = t1[n_opt]
        name = f't{t}hyperbolic'
        Run(L, lr_range, ep_range, 22, train, train_tar, test, test_tar, Cross_Entropy, shapes, AF.SoftMax(), opt, name, Lmd, t, schedule)

def SendToLambda(L, Lr, ep, t1, opt, schedule):
    Lmd_range = np.linspace(1e-5, 1e-3, 50)
    shapes = (64, 10)
    train, test, val, train_tar, test_tar, val_tar = Data.load_data()
    name = f't{t1}hyperbolic'
    RunLambda(L, Lr, ep, 22, train, train_tar, test, test_tar, Cross_Entropy, shapes, AF.SoftMax(), opt, name, Lmd_range, t1, schedule)
