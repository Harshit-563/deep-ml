import numpy as np

def shuffle_data(X, y, seed=None):
   if seed is not None:
    a = np.random.seed(seed)
    n = X.shape[0]
    b = np.random.permutation(n)
    return X[b],y[b]
