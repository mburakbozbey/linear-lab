""" Tests for the LinearRegression class. """

import numpy as np

from algorithms.linear_regression import LinearRegression


def test_linear_regression_fit():
    """
    Test case for the fit method of LinearRegression class.
    """
    # Create a LinearRegression instance
    lr = LinearRegression(learning_rate=0.01, num_iterations=1000)

    # Generate some dummy data
    x = np.array([[1, 2], [3, 4]])
    y = np.array([0.5, 1.0])

    # Fit the model to the data
    lr.fit(x, y)

    # Assert that the weights have been learned
    assert lr.weights is not None


def test_linear_regression_predict():
    """
    Test case for the predict method of the LinearRegression class.

    This test case verifies that the predict method of the LinearRegression class
    correctly predicts the target values for a given input.

    Steps:
    1. Create a LinearRegression instance with a specified learning rate and number of iterations.
    2. Set the weights of the LinearRegression instance manually.
    3. Generate some dummy input data.
    4. Predict the target values using the predict method of the LinearRegression instance.
    5. Assert that the predicted values are equal to the expected values.

    Expected behavior:
    - The predicted values should be equal to the expected values.

    """

    # Create a LinearRegression instance
    lr = LinearRegression(learning_rate=0.01, num_iterations=1000)

    # Set the weights manually
    lr.weights = np.array([0.5, 0.5])

    # Generate some dummy data
    x = np.array([[1, 2], [3, 4]])

    # Predict the target values
    y_pred = lr.predict(x)

    # Assert that the predicted values are correct
    assert np.array_equal(y_pred, np.array([1.5, 3.5]))


def test_linear_regression_cost_history():
    """
    Test case for checking the cost history of LinearRegression.

    This function creates a LinearRegression instance, generates some dummy data,
    fits the model to the data, and asserts that the cost history is not empty.

    Returns:
        None
    """
    # Create a LinearRegression instance
    lr = LinearRegression(learning_rate=0.01, num_iterations=1000)

    # Generate some dummy data
    x = np.array([[1, 2], [3, 4]])
    y = np.array([0.5, 1.0])

    # Fit the model to the data
    lr.fit(x, y)

    # Assert that the cost history is not empty
    assert len(lr.cost_history) > 0
