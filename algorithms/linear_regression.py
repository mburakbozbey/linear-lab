import time

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils.preprocess import preprocess_data


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


import plotly.graph_objects as go


def plot_data_and_regression(X, y, predictions):
    """Plot data and regression line using Plotly.

    Parameters:
        X (ndarray): Input features.
        y (ndarray): Target values.
        predictions (ndarray): Predicted values.
    """

    fig = go.Figure()

    # Sort the data by X for plotting the actual data points and regression line
    sorted_indices = X[:, 1].argsort()
    X = X[sorted_indices]
    y = y[sorted_indices]
    predictions = predictions[sorted_indices]

    # Add actual data points
    fig.add_trace(go.Scatter(x=X[:, 1], y=y, mode="markers", name="Actuals"))

    # Add regression line
    fig.add_trace(
        go.Scatter(x=X[:, 1], y=predictions, mode="markers", name="Predictions")
    )

    fig.update_layout(
        title="Linear Regression",
        xaxis_title="X",
        yaxis_title="y",
        legend=dict(x=0, y=1, bordercolor="Black", borderwidth=2),
    )

    fig.show()


def run_linear_regression_california():
    """
    Run linear regression on the California housing dataset.

    This function loads the California housing dataset, preprocesses the data, adds a bias term, splits the data into training and testing sets, creates and trains a linear regression model with bias, makes predictions on the test set, calculates evaluation metrics, and plots the linear regression line and data points.
    """
    # Load the California housing dataset
    california_housing = fetch_california_housing()
    X, y = california_housing.data, california_housing.target

    # Preprocess the data using the imported function
    X_scaled = preprocess_data(X)

    # Add bias term
    X_with_bias = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_with_bias, y, test_size=test_size, random_state=42
    )

    # Create and train the linear regression model with bias
    model = LinearRegression(learning_rate=learning_rate, num_iterations=num_iterations)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Plot the linear regression line and data points
    plot_data_and_regression(X_test, y_test, predictions, model)

    print("Mean Squared Error: {:.2f}".format(mse))
    print("R-squared: {:.2f}".format(r2))


if __name__ == "__main__":
    start_time = time.time()
    run_linear_regression_california()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time: {:.2f} seconds".format(execution_time))
