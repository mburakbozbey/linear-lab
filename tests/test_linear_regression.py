from algorithms.linear_regression import LinearRegression, plot_data_and_regression
from utils.preprocess import preprocess_data
import pytest
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from config.config import Config  # Import the configuration
import plotly.graph_objects as go

config = Config()
LEARNING_RATE = config.LEARNING_RATE
NUM_ITERATIONS = config.NUM_ITERATIONS
TEST_SIZE = config.TEST_SIZE

@pytest.fixture
def california_housing_dataset():
    return fetch_california_housing()

@pytest.fixture
def scaled_features(california_housing_dataset):
    X, y = california_housing_dataset.data, california_housing_dataset.target
    scaled_features = StandardScaler().fit_transform(X)
    return scaled_features

@pytest.fixture
def features_with_bias(scaled_features):
    features_with_bias = np.c_[np.ones((scaled_features.shape[0], 1)), scaled_features]
    return features_with_bias

@pytest.fixture
def train_test_data(features_with_bias, california_housing_dataset):
    X_train, X_test, y_train, y_test = train_test_split(features_with_bias, california_housing_dataset.target, test_size=TEST_SIZE, random_state=42)
    return X_train, X_test, y_train, y_test

@pytest.fixture
def trained_model(train_test_data):
    X_train, X_test, y_train, y_test = train_test_data
    model = LinearRegression(learning_rate=LEARNING_RATE, num_iterations=NUM_ITERATIONS)
    model.fit(X_train, y_train)
    return model

def test_model_fit(trained_model):
    assert trained_model.weights is not None
    assert trained_model.cost_history is not None
    assert len(trained_model.cost_history) == NUM_ITERATIONS

def test_model_predict(trained_model, train_test_data):
    X_train, X_test, y_train, y_test = train_test_data
    predictions = trained_model.predict(X_test)
    assert predictions.shape == y_test.shape

def test_plot_data_and_regression(trained_model, train_test_data):
    X_train, X_test, y_train, y_test = train_test_data
    predictions = trained_model.predict(X_test)

    # Plot the data and regression line using Plotly
    fig = go.Figure()

    # Add actual data points
    fig.add_trace(go.Scatter(x=X_test[:, 1], y=y_test, mode='markers', name='Actual Data'))

    # Add regression line
    fig.add_trace(go.Scatter(x=X_test[:, 1], y=predictions, mode='lines', name='Linear Regression Line'))

    fig.update_layout(title='Linear Regression',
                      xaxis_title='X',
                      yaxis_title='y',
                      legend=dict(x=0, y=1, bordercolor="Black", borderwidth=2))

    fig.show()