import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    B = np.asarray(B)
    C = np.asarray(C)
    d = np.linalg.inv(C)
    P = d @ B
    return P