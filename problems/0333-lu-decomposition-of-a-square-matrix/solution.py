import numpy as np
from scipy.linalg import lu, solve

def lu_decomposition(A: list) -> tuple:
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        # Compute U
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(
                L[i][k] * U[k][j] for k in range(i)
            )

        # Compute L
        for j in range(i + 1, n):
            L[j][i] = (
                A[j][i]
                - sum(L[j][k] * U[k][i] for k in range(i))
            ) / U[i][i]

    return (L, U)