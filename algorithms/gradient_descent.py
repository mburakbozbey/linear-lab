import numpy as np


class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, gradients):
        updated_weights = weights - self.learning_rate * gradients
        return updated_weights
