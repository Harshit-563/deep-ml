import numpy as np

def posterior(X, y, m0, S0, sigma2):
    sn = np.linalg.inv(
        np.linalg.inv(S0) + (1 / sigma2) * X.T @ X
    )

    mn = sn @ (
        np.linalg.solve(S0, m0) +
        (1 / sigma2) * X.T @ y
    )

    return mn, sn


def sequential_posterior(X, y, m0, S0, sigma2):
    means = []

    m = m0
    S = S0

    for i in range(len(X)):
        m, S = posterior(
            X[i:i+1],
            y[i:i+1],
            m,
            S,
            sigma2
        )

        means.append(m)

    return m, S, np.array(means)


def predictive(X_test, mN, SN, sigma2):
    mean = X_test @ mN
    var = sigma2 + np.sum((X_test @ SN) * X_test, axis=1)

    return mean, var