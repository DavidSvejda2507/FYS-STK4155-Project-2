import numpy as np
import matplotlib.pyplot as plt
import NeuralNet as NN
import optimisers as op
import ActivationFunctions as AF
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
    for k, lr in enumerate(lr_range):
        print('.', end='', flush=True)
        for j, n_epoch in enumerate(ep_range):
            print(',', end='', flush=True)
            model = NN.Model(shapes, [af]*(len(shapes)-1), opt(lr, L))

            inputs = data

            Acc, Cross_Ent = lr_ep_error(n_epoch, inputs, targets, test_data, test_targets, costFunc, model)

            if Cross_Ent<min:
                min = Cross_Ent
                final_acc = Acc
                k_min = k
                j_min = j
    print('\n')
    #k = lr, j = epochs
    return k_min, j_min, min, final_acc

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

def FindOpt(data, targets, test_data, test_targets, l_range, lr_range, ep_range, costFunc, af, opt):
    min = 1e8
    print('Fixed Lambda')
    L = l_range[int(len(l_range)/2)]
    k_min, j_min, min, final_acc = FixedLambda(L, lr_range, ep_range, data, targets, test_data, test_targets, costFunc, shapes, af, opt)
    print('Fixed Lr Epoch')
    i_min = FixedLrEpoch(lr_range[k_min], ep_range[j_min], l_range, data, targets, test_data, test_targets, costFunc, shapes, af, opt)
    print('Fixed Lambda (2nd)')
    k_min, j_min, min, final_acc = FixedLambda(l_range[i_min], lr_range, ep_range, data, targets, test_data, test_targets, costFunc, shapes, af, opt)
    return i_min, k_min, j_min, min, final_acc

def ZoomArray(array, index):
    diff = abs(array[index]- array[index-1])
    ret_array = np.arange(array[index]-diff, array[index]+diff, diff/len(array))
    return ret_array

def FindPlateau(data, targets, test_data, test_targets, cutoff, l_range, lr_range, ep_range, costFunc, af, opt):
    dE = 1e8
    # i = L, k = lr, j = epoch
    i_min, k_min, j_min, min, final_acc = FindOpt(data, targets, test_data, test_targets, l_range, lr_range, ep_range, costFunc, af, opt)
    while dE > cutoff:
            NewRun = 'New Run'
            print(f'{NewRun:-^20}')
            l_range = ZoomArray(l_range, i_min)
            lr_range = ZoomArray(lr_range, k_min)
            ep_range = ZoomArray(ep_range, j_min)
            i_min, k_min, j_min, min_new, acc_new = FindOpt(data, targets, test_data, test_targets, l_range, lr_range, ep_range, costFunc, af, opt)
            dE = abs(min-min_new)
            min = min_new

    # Found minimum, extracting values
    L_min = l_range[i_min]
    lr_min = lr_range[k_min]
    ep_min = ep_range[j_min]

    return L_min, lr_min, ep_min, min, acc_new

[l_range, lr_range, ep_range] = [np.linspace(1, 10, 10), np.linspace(0, 10, 5), np.logspace(1, 3, 4)]
shapes = (64, 10)
train, test, val, train_tar, test_tar, val_tar = Data.load_data()
cutoff = 1e-2
print(FindPlateau(train, train_tar, test, test_tar, cutoff, l_range, lr_range, ep_range, Cross_Entropy, AF.SoftMax(), op.MomentumOptimiser))
