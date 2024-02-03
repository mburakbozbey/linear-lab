""" Tests for the main module. """

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from algorithms.linear_regression import LinearRegression
from utils.preprocess import preprocess_data


@pytest.fixture
def california_housing_data():
    """
    Fixture to load and preprocess the California housing dataset.
    """
    # Load the California housing dataset
    california_housing = fetch_california_housing()
    x, target = california_housing.data, california_housing.target
    x = pd.DataFrame(x)

    # Convert all columns to numeric, replacing non-numeric values with NaN
    x = x.apply(pd.to_numeric, errors="coerce")

    # Now you can call preprocess_data
    x_scaled = preprocess_data(x.values)
    # Preprocess the data using the imported function

    # Add bias term
    x_with_bias = np.c_[np.ones((x_scaled.shape[0], 1)), x_scaled]

    return x_with_bias, target


def test_model_training(california_housing_data):
    """
    Test the model training process.
    """
    x_with_bias, y = california_housing_data

    # Define the test size
    test_size = 0.2

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_with_bias, y, test_size=test_size, random_state=42
    )

    # Define the learning rate and number of iterations
    learning_rate = 0.01
    num_iterations = 1000

    # Create and train the linear regression model with bias
    model = LinearRegression(learning_rate=learning_rate, num_iterations=num_iterations)
    model.fit(x_train, y_train)

    return model, x_test, y_test


def test_model_prediction(california_housing_data):
    """
    Test the model prediction process.
    """
    model, X_test, y_test = test_model_training(california_housing_data)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Assert that the mean squared error is within a certain range
    assert 0 <= mse <= 100

    # Assert that the R-squared value is within a certain range
    assert 0 <= r2 <= 1
