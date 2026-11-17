import math

def single_neuron_model(features: list[list[float]], labels: list[int],
                        weights: list[float], bias: float) -> tuple[list[float], float]:
    
    probabilities = []

    for x in features:
        # Linear combination
        z = sum(w * f for w, f in zip(weights, x)) + bias
        
        # Sigmoid activation
        prob = 1 / (1 + math.exp(-z))
        probabilities.append(round(prob, 4))

    # Mean Squared Error
    mse = sum((y - p) ** 2 for y, p in zip(labels, probabilities)) / len(labels)
    mse = round(mse, 4)

    return probabilities, mse
    