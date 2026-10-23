import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	if (len(a[0])!=len(a)): return -1
	a = np.asarray(a,dtype=float)
	b = np.asarray(b,dtype=float)
	c = a @ b 
	return c

	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	