""" Tests for the preprocess module. """

import numpy as np
import pytest

from utils.preprocess import plot_data_and_regression, preprocess_data


@pytest.fixture
def preprocessed_data_fixture():
    """
    Fixture for preprocessed_data.
    """
    # Create dummy data
    x = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])

    # Call the function
    preprocessed_data = preprocess_data(x)

    return preprocessed_data


def test_plot_data_and_regression_without_predictions():
    """
    Test the plot_data_and_regression function without predictions.
    """
    # Create dummy data
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([10, 11, 12])

    # Call the function
    plot_data_and_regression(x, y)


def test_plot_data_and_regression_with_predictions():
    """
    Test the plot_data_and_regression function with predictions.
    """
    # Create dummy data
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([10, 11, 12])
    predictions = np.array([9, 10, 11])

    # Call the function
    plot_data_and_regression(x, y, predictions)


def test_preprocess_data():
    """
    Test the preprocess_data function.
    """
    # Assert the output
    expected_output = np.array([[1, 2, 0], [4, 0, 6], [7, 8, 9]])
    assert np.array_equal(preprocessed_data_fixture, expected_output)
