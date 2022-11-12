import numpy as np

def load_data(train=0.6, test=0.2, val=0.2):

    """
    Returns train test and val data set in the form

    [(target, [flattened image (64,)]), ...]
    """

    from sklearn.datasets import load_digits

    mnist = load_digits()
    n_images = len(mnist.images)
    n_train = int(train*n_images)
    n_test = n_train + int(test*n_images)
    n_val = n_test + int(val*n_images)

    Flat = np.array([Im.flatten() for Im in mnist.images])
    Tar = mnist.target

    train = Flat[:n_train]
    test = Flat[n_train:n_test]
    val = Flat[n_test:n_val]

    train_tar = Tar[:n_train]
    test_tar = Tar[n_train:n_test]
    val_tar = Tar[n_test:n_val]

    return train.T, test.T, val.T, train_tar.T, test_tar.T, val_tar.T
