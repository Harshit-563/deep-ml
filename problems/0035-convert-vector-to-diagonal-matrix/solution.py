import numpy as np

def make_diagonal(x):
	A = [[0 for _ in range(len(x))] for _ in range(len(x))]
	for i in range(len(A)):
		for j in range(len(A)):
			if(i==j):
				A[i][j]=x[i] 
	return A			
