import numpy as np

def bincount(a: np.ndarray, n_bins: int) -> np.ndarray:
    """Count occurrences of 0..n_bins-1 in integer array a."""
    d = {}
    for x in range(n_bins):
        d[x]=0
    for x in a:
        d[x]+=1
    b = []    
    for v in d.values():
        b.append(v)
    return np.asarray(b)    

