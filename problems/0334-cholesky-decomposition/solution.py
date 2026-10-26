import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    c = np.linalg.det(A)
    if(len(A)!=len(A[0]) or c<=0 ): return -1
    b = np.linalg.cholesky(A)
    return b.tolist()