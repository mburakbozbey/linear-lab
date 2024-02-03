"""Tests for the config module."""

import pytest

from config.config import Config


@pytest.fixture(name="config")
def config_fixture():
    """
    Returns a Config object initialized with the parameters from "config/params.yaml".
    """
    return Config("config/params.yaml")


def test_get_model_learning_rate(config):
    """
    Test case for the get_model_learning_rate method of the Config class.

    This test verifies that the get_model_learning_rate method returns
    the correct learning rate value.s

    Args:
        config: An instance of the Config class.

    Returns:
        None
    """
    learning_rate = config.get_model_learning_rate()
    assert learning_rate == 0.01


def test_get_model_learning_rate_with_default(config):
    """
    Test case to verify the behavior of the `get_model_learning_rate` method
    when a default value is provided.

    Args:
        config: An instance of the Config class.

    Returns:
        None
    """
    learning_rate = config.get_model_learning_rate(default=0.1)
    assert learning_rate == 0.01


def test_get_model_num_iterations(config):
    """
    Test case for the `get_model_num_iterations` method of the `config` object.

    This test verifies that the `get_model_num_iterations` method returns the
    expected value of 1000.

    Args:
        config: An instance of the Config class.

    Returns:
        None
    """
    num_iterations = config.get_model_num_iterations()
    assert num_iterations == 1000


def test_get_model_num_iterations_with_default(config):
    """
    Test the 'get_model_num_iterations' method with the default value.

    Args:
        config: The configuration object.

    Returns:
        None
    """
    num_iterations = config.get_model_num_iterations(default=2000)
    assert num_iterations == 1000


def test_get_data_split_test_size(config):
    """
    Test case to verify the correctness of the `get_data_split_test_size`
    method in the `config` object.

    This test checks if the returned test size is equal to 0.2.
    """
    test_size = config.get_data_split_test_size()
    assert test_size == 0.2


def test_get_data_split_test_size_with_default(config):
    """
    Test case to verify the behavior of the `get_data_split_test_size` method
    when using the default value.

    Args:
        config: An instance of the Config class.

    Returns:
        None
    """
    test_size = config.get_data_split_test_size(default=0.3)
    assert test_size == 0.2
