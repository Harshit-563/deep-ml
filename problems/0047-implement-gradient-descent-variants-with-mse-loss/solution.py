import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.

    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent
        method: 'batch', 'stochastic', or 'mini_batch'

    Returns:
        Optimized weights
    """

    m = X.shape[0]
    weights = weights.astype(float).copy()

    for _ in range(n_epochs):

        if method == 'batch':
            predictions = X @ weights
            gradient = (2 / m) * X.T @ (predictions - y)
            weights -= learning_rate * gradient

        elif method == 'stochastic':
            for i in range(m):
                xi = X[i:i+1]
                yi = y[i:i+1]

                prediction = xi @ weights
                gradient = 2 * xi.T @ (prediction - yi)
                weights -= learning_rate * gradient.flatten()

        elif method == 'mini_batch':
            for start in range(0, m, batch_size):
                end = start + batch_size

                X_batch = X[start:end]
                y_batch = y[start:end]

                batch_m = len(X_batch)

                predictions = X_batch @ weights
                gradient = (2 / batch_m) * X_batch.T @ (predictions - y_batch)
                weights -= learning_rate * gradient

    return weights