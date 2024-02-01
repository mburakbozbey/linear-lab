import numpy as np
import plotly.graph_objects as go
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from algorithms.linear_regression import LinearRegression, plot_data_and_regression
from config.config import Config  # Import the configuration
from utils.preprocess import preprocess_data

config = Config()
LEARNING_RATE = config.LEARNING_RATE
NUM_ITERATIONS = config.NUM_ITERATIONS
TEST_SIZE = config.TEST_SIZE


@pytest.fixture
def california_housing_dataset():
    """
    Fixture for the California housing dataset.
    """
    return fetch_california_housing()


@pytest.fixture
def scaled_features(california_housing_dataset):
    """
    A fixture that scales the features of the California housing dataset.
    """
    X, y = california_housing_dataset.data, california_housing_dataset.target
    scaled_features = StandardScaler().fit_transform(X)
    return scaled_features


@pytest.fixture
def features_with_bias(scaled_features):
    features_with_bias = np.c_[np.ones((scaled_features.shape[0], 1)), scaled_features]
    return features_with_bias


@pytest.fixture
def train_test_data(features_with_bias, california_housing_dataset):
    """
    Fixture for generating train and test data from features and target dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features_with_bias,
        california_housing_dataset.target,
        test_size=TEST_SIZE,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def trained_model(train_test_data):
    """
    Fixture for a trained model using the provided train test data.
    """
    X_train, X_test, y_train, y_test = train_test_data
    model = LinearRegression(learning_rate=LEARNING_RATE, num_iterations=NUM_ITERATIONS)
    model.fit(X_train, y_train)
    return model


def test_model_fit(trained_model):
    assert trained_model.weights is not None
    assert trained_model.cost_history is not None
    """
    A function to test the fitting of the model.

    Args:
        trained_model: The trained model to be tested.

    Returns:
        None
    """
    assert len(trained_model.cost_history) == NUM_ITERATIONS


def test_model_predict(trained_model, train_test_data):
    """
    Function to make predictions using a trained model and assert the shape of the predictions.

    Parameters:
    - trained_model: the trained model for making predictions
    - train_test_data: a tuple containing the training and testing data (X_train, X_test, y_train, y_test)

    Returns:
    None
    """
    X_train, X_test, y_train, y_test = train_test_data
    predictions = trained_model.predict(X_test)
    assert predictions.shape == y_test.shape


def test_plot_data_and_regression():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    predictions = np.array([1.5, 2.5])
    plot_data_and_regression(X, y, predictions)
    # Add assertions for the expected behavior of the function
