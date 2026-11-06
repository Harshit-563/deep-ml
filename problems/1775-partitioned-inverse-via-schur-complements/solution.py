import numpy as np


def partitioned_inverse(M, k):
    """
    Compute the inverse of

        M = [[E, F],
             [G, H]]

    using the Schur complement S = E - F H^{-1} G.
    """
    M = np.asarray(M, dtype=float)

    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]

    H_inv = np.linalg.inv(H)

    S = E - F @ H_inv @ G
    S_inv = np.linalg.inv(S)

    top_left = S_inv
    top_right = -S_inv @ F @ H_inv
    bottom_left = -H_inv @ G @ S_inv
    bottom_right = H_inv + H_inv @ G @ S_inv @ F @ H_inv

    return np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])


def block_slogdet(M, k):
    """
    Compute (sign, logabsdet) using

        det(M) = det(E - F H^{-1} G) * det(H)
    """
    M = np.asarray(M, dtype=float)

    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]

    H_inv = np.linalg.inv(H)
    S = E - F @ H_inv @ G

    sign_s, logabs_s = np.linalg.slogdet(S)
    sign_h, logabs_h = np.linalg.slogdet(H)

    return [float(sign_s * sign_h),
            float(logabs_s + logabs_h)]


def bordered_inverse(A_inv, b, c):
    """
    Given A^{-1}, compute the inverse of

        [[A, b],
         [b^T, c]]

    in O(n^2) time.
    """
    A_inv = np.asarray(A_inv, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    Ab = A_inv @ b
    s = c - b @ Ab

    if abs(s) < 1e-12:
        raise ValueError("Schur complement is singular")

    s_inv = 1.0 / s

    top_left = A_inv + s_inv * np.outer(Ab, Ab)
    top_right = (-s_inv * Ab).reshape(-1, 1)
    bottom_left = (-s_inv * Ab).reshape(1, -1)
    bottom_right = np.array([[s_inv]])

    return np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])