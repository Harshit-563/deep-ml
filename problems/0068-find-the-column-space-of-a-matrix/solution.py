
import numpy as np

def matrix_image(A):
	# Write your code here
	a = np.linalg.matrix_rank(A)
	b = [[0 for _ in range(a)] for _ in range(len(A))]
	for i in range(len(A)):
		for j in range(a):
			b[i][j]=A[i][j]
	return b		
			
		
