""" This module defines a class to load and access parameters from a YAML file. """

import yaml


class Config:
    """
    This class represents a configuration object that loads and accesses parameters
    from a YAML file.
    """

    def __init__(self, config_file):
        self.params = self.load_config(config_file)

    @staticmethod
    def load_config(config_file):
        """
        Load the configuration from a YAML file.

        Args:
            config_file (str): The path to the configuration file.

        Returns:
            dict: The loaded configuration parameters.
        """
        with open(config_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def get_model_learning_rate(self, default=0.01):
        """
        Get the learning rate for the model.

        Args:
            default (float, optional): The default learning rate if not specified
                                     in the configuration.
                                      Defaults to 0.01.

        Returns:
            float: The learning rate.
        """
        return self.params.get("model", {}).get("learning_rate", default)

    def get_model_num_iterations(self, default=1000):
        """
        Get the number of iterations for the model.

        Args:
            default (int, optional): The default number of iterations if not specified
            in the configuration.
                                     Defaults to 1000.

        Returns:
            int: The number of iterations.
        """
        return self.params.get("model", {}).get("num_iterations", default)

    def get_data_split_test_size(self, default=0.2):
        """
        Get the test size for data split.

        Args:
            default (float, optional): The default test size if not specified in the
            configuration.
                                       Defaults to 0.2.

        Returns:
            float: The test size.
        """
        return self.params.get("data_split", {}).get("test_size", default)

    # Add other methods to access additional parameters as needed


# Usage example
config = Config("config/params.yaml")
learning_rate = config.get_model_learning_rate()
num_iterations = config.get_model_num_iterations()
test_size = config.get_data_split_test_size()
