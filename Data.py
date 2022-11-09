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
    Tar_Im = [(Tar, Im.flatten()) for Tar, Im in zip(mnist.target, mnist.images)]
    Tar_Im = np.array(Tar_Im, dtype=object)
    train_set = Tar_Im[:n_train]
    test_set = Tar_Im[n_train:n_test]
    val_set = Tar_Im[n_test:]
    
    return train_set, test_set, val_set
