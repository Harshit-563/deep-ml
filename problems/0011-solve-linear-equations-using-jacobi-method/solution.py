import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    # Ensure float type for division accuracy
    A = A.astype(float)
    b = b.astype(float)
    
    # Initialize the solution vector with zeros
    x = np.zeros_like(b)
    
    # Extract the diagonal elements and the remainder matrix
    D = np.diag(A)
    R = A - np.diag(D)
    
    # Iterate n times
    for _ in range(n):
        # x_next = (b - R * x) / D
        x = (b - np.dot(R, x)) / D
        
    return x
