import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
            X (array-like): The input data.
            y (array-like): The target values.

        Returns:
            None
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)

        for _ in range(self.num_iterations):
            y_pred = np.dot(X, self.weights)
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            self.weights -= self.learning_rate * dw
            cost = (1 / (2 * num_samples)) * np.sum(np.square(y_pred - y))
            self.cost_history.append(cost)

    def predict(self, X):
        return np.dot(X, self.weights)
