""" Main script to run the linear regression experiment. """

import time

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from algorithms.linear_regression import LinearRegression
from config.config import Config
from utils.preprocess import plot_data_and_regression, preprocess_data

config = Config("config/params.yaml")
learning_rate = config.get_model_learning_rate()
num_iterations = config.get_model_num_iterations()
test_size = config.get_data_split_test_size()


def run_experiment():
    """
    Run the linear regression experiment.
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
    plot_data_and_regression(X_test, y_test, predictions)

    print(f"Mean Squared Error: {mse:.2f}")  # Use f-string for string formatting
    print(f"R-squared: {r2:.2f}")  # Use f-string for string formatting


if __name__ == "__main__":
    start_time = time.time()
    run_experiment()
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
