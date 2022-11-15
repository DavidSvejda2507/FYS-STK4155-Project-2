import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import os

figsize = (3.313, 3)


def plot_lr_epochs(filename: str, fig_name: str, fig_title: str, savepath: str):
    data = np.load(filename, allow_pickle=True)

    plt.figure(figsize=figsize)
    plt.imshow(data[1:,1:])

    plt.yticks(ticks = np.arange(0, 10, 1), labels = np.round(data[1:, 0], 4))
    plt.xticks(ticks = np.arange(0, 10, 1), labels = map((int), data[0, 1:]), rotation=-40)
    plt.tick_params(axis='y', length=3)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.ylim(1e-4, 1e2)
    # plt.xlim(1e-4, 1e2)
    plt.xlabel("epochs")
    plt.ylabel("learning rate")

    plt.colorbar()
    plt.title(fig_title)
    plt.tight_layout()
    plt.savefig(savepath)
    # plt.show()



def plot_lamda(filename: str, fig_name: str, fig_title: str, savepath: str):
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
    fig.savefig(savepath)
    #fig.show()

if __name__ == "__main__":
    base = "Data/NrHidden0/"
    for dir in os.listdir(base):
        for dir2 in os.listdir(base + dir):
            for name in os.listdir(base + dir + '/' + dir2):
                plot_lr_epochs(f'{base}{dir}/{dir2}/{name}', dir2 + name + ".png", f"{dir}{name[:3]}", f"Data/Plots/{dir}{name[:3]}.png")
