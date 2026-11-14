import numpy as np

def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exp_scores = [np.exp(x - max_score) for x in scores]
    total = sum(exp_scores)
    return [x / total for x in exp_scores]