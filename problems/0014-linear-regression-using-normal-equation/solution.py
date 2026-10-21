import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X = np.asarray(X,dtype=float)
	y = np.asarray(y,dtype=float)
	theta = np.linalg.inv(X.T @ X) @ X.T @ y

	return theta