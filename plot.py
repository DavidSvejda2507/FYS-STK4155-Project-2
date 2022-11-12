import numpy as np
import matplotlib.pyplot as plt

figsize = (3.313, 2)


def plot_lr_epochs(filename: str, fig_name: str, fig_title: str):
    data = np.load(filename, allow_pickle=True)
    
    plt.figure(figsize=figsize)
    plt.imshow(data[1:,1:])
    
    plt.xticks(data[1:, 0])
    plt.yticks(data[0, 1:])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
    
    plt.title(fig_title)
    plt.savefig(fig_name)
    plt.show()
    
    
    
def plot_lamda(filename: str, fig_name: str, fig_title: str):
    data = pd.read_csv(filename)
    
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    ax1.plot(data, "r--")
    ax2.plot(data, "b--")
    
    ax1.set_xscale("log")

    ax1.set_xlabel("lambda")
    ax1.set_ylabel("Cross Entropy")
    ax2.set_ylabel("Accuracy")
    
    fig.title(fig_title)
    fig.savefig(fig_name)
    fig.show()
    
if __name__ == "__main__":
    plot_lr_epochs("Data/NrHidden0/MomentumOptimiser/LrEpoch/Acc_L_5.txt", "test.pdf", "title")