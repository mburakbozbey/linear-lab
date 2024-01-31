
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from main import run_experiment

# Pytest fixture to mock the configuration file content
@pytest.fixture
def mock_config_yaml(monkeypatch):
    config = {
        'data': {'test_size': 0.2, 'random_state': 42},
        'preprocessing': {'apply_standard_scaler': True},
        'model': {'learning_rate': 0.01, 'num_iterations': 1000},
        'plot': {'title': 'Linear Regression Analysis', 'x_axis_title': 'Median Income', 'y_axis_title': 'Median House Value'}
    }
    monkeypatch.setattr('yaml.safe_load', lambda x: config)

# Fixture to mock the fetch_california_housing function
@pytest.fixture
def mock_fetch_data(monkeypatch):
    monkeypatch.setattr('main.fetch_california_housing', lambda: (np.array([[1, 2], [3, 4]]), np.array([0.5, 1.0])))

# Fixture to mock the train_test_split function
@pytest.fixture
def mock_split(monkeypatch):
    monkeypatch.setattr('main.train_test_split', lambda *args, **kwargs: (np.array([[1, 2]]), np.array([[3, 4]]), np.array([0.5]), np.array([1.0])))

# Fixture to mock the LinearRegressionWithBias class
@pytest.fixture
def mock_lr(monkeypatch):
    mock_model_instance = MagicMock()
    monkeypatch.setattr('main.LinearRegressionWithBias', MagicMock(return_value=mock_model_instance))
    mock_model_instance.predict.return_value = np.array([1.0])
    return mock_model_instance

# Fixture to mock the plot_data_and_regression function
@pytest.fixture
def mock_plot(monkeypatch):
    monkeypatch.setattr('main.plot_data_and_regression', MagicMock())

def test_run_experiment(mock_config_yaml, mock_fetch_data, mock_split, mock_lr, mock_plot):
    # Run the experiment
    run_experiment('test_config.yaml')

    # Assert the LinearRegressionWithBias was initialized correctly
    mock_lr.assert_called_with(learning_rate=0.01, num_iterations=1000)

    # Assert the model was fit with the correct data
    mock_lr.fit.assert_called_with(np.array([[1, 2]]), np.array([0.5]))

    # Assert predict was called on the model
    mock_lr.predict.assert_called_with(np.array([[3, 4]]))

    # Assert the plotting function was called
    mock_plot.assert_called_once()
