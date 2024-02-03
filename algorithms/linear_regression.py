"""Module providing a function efficiently calculate math operations"""

import numpy as np


class LinearRegression:
    """
    Linear regression model.

    Parameters:
        learning_rate (float): The learning rate for gradient descent. Default is 0.01.
        num_iterations (int): The number of iterations for gradient descent. Default is 1000.

    Attributes:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        weights (ndarray): The learned weights of the linear regression model.
        cost_history (list): The history of cost values during training.

    Methods:
        fit(X, y): Fit the model to the training data.
        predict(X): Predict the target values for the input data.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.cost_history = []

    def fit(self, x, y):
        """
        Fit the model to the training data.

        Parameters:
            x (array-like): The input data.
            y (array-like): The target values.

        Returns:
            None
        """
        num_samples, num_features = x.shape
        self.weights = np.zeros(num_features)

        for _ in range(self.num_iterations):
            y_pred = np.dot(x, self.weights)
            dw = (1 / num_samples) * np.dot(x.T, (y_pred - y))
            self.weights -= self.learning_rate * dw
            cost = (1 / (2 * num_samples)) * np.sum(np.square(y_pred - y))
            self.cost_history.append(cost)

    def predict(self, x):
        """
        Predict the target values for the input data.

        Parameters:
            x (array-like): The input data.

        Returns:
            ndarray: The predicted target values.
        """
        return np.dot(x, self.weights)
