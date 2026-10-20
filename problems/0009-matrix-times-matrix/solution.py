import numpy as np
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
                    d = len(a[0])
                    e = len(b)
                    if (d!=e): return -1
                    a = np.asarray(a,dtype=float)
                    b = np.asarray(b,dtype=float)
                    c = a @ b
                    
                    return c