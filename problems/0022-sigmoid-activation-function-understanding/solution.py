import math

def sigmoid(z: float) -> float:
    #Your code here
    result = 1/(1+math.exp(-1*z))
    return result