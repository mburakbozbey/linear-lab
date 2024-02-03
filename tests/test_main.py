""" Tests for the main module. """

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from algorithms.linear_regression import LinearRegression
from main import preprocess_data


@pytest.fixture
def california_housing_data():
    """
    Fixture to load and preprocess the California housing dataset.
    """
    # Load the California housing dataset
    california_housing = fetch_california_housing()
    data, target = california_housing.data, california_housing.target

    # Preprocess the data using the imported function
    X_scaled = preprocess_data(data)

    # Add bias term
    X_with_bias = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

    return X_with_bias, target


def test_preprocess_data():
    """
    Test the preprocess_data function
    """
    X = np.array([[1, 2, 3], [4, 5, 6]])
    expected_result = np.array(
        [[-1.22474487, -0.81649658, -0.40824829], [0.40824829, 1.22474487, 2.04124145]]
    )
    assert np.allclose(preprocess_data(X), expected_result)


def test_model_training():
    """
    Test the model training process.
    """
    X_with_bias, y = california_housing_data

    # Define the test size
    test_size = 0.2

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_with_bias, y, test_size=test_size, random_state=42
    )

    # Define the learning rate and number of iterations
    learning_rate = 0.01
    num_iterations = 1000

    # Create and train the linear regression model with bias
    model = LinearRegression(learning_rate=learning_rate, num_iterations=num_iterations)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_model_prediction():
    """
    Test the model prediction process.
    """
    model, X_test, y_test = test_model_training()

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Assert that the mean squared error is within a certain range
    assert 0 <= mse <= 100

    # Assert that the R-squared value is within a certain range
    assert 0 <= r2 <= 1
