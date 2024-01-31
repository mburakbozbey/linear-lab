import pytest
from config.config import Config
import os

print(os.getcwd())
# Create a fixture to initialize the Config object
@pytest.fixture
def config():
    return Config('config/params.yaml')

# Test the get_model_params method
def test_get_model_params(config):
    model_params = config.get_model_params()
    assert model_params == {'learning_rate': 0.01, 'num_iterations': 1000}

# Test the get_scaler_params method
def test_get_scaler_params(config):
    scaler_params = config.get_scaler_params()
    assert scaler_params == {'method': 'StandardScaler', 'with_bias': True}

# Test the get_data_split_params method
def test_get_data_split_params(config):
    data_split_params = config.get_data_split_params()
    assert data_split_params == {'test_size': 0.2, 'random_state': 42}

# Test the get_metrics method
def test_get_metrics(config):
    metrics = config.get_metrics()
    assert metrics == ['mean_squared_error', 'r2_score']

if __name__ == '__main__':
    pytest.main(['-v', 'test_config.py'])