import numpy as np
def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	a = np.asarray(a,dtype=float)
	b = np.asarray(b,dtype=float)
	if (a.shape!= b.shape): return -1
	c = a + b 
	return c 


