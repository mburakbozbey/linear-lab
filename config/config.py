import yaml

class Config:
    def __init__(self, config_file='config/params.yaml'):
        with open(config_file, 'r') as file:
            self.params = yaml.safe_load(file)

        # Access the model parameters
        self.LEARNING_RATE = self.params['model'].get('learning_rate', 0.01)
        self.NUM_ITERATIONS = self.params['model'].get('num_iterations', 1000)

        # Access the data split parameters
        self.TEST_SIZE = self.params['data_split'].get('test_size', 0.2)

    def get_model_params(self):
        return self.params.get('model', {})

    def get_scaler_params(self):
        return self.params.get('scaler', {})

    def get_data_split_params(self):
        return self.params.get('data_split', {})

    def get_metrics(self):
        return self.params.get('metrics', [])
