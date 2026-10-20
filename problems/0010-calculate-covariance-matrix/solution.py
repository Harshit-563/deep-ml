import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	vectors = np.asarray(vectors,dtype=float)
	d = np.cov(vectors)
	return d
	