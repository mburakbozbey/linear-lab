import time
import numpy as np
import plotly.express as px
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from utils.preprocess import preprocess_data
from config.config import Config  # Import the configuration

# Constants from config.py
config = Config()
LEARNING_RATE = config.LEARNING_RATE
NUM_ITERATIONS = config.NUM_ITERATIONS
TEST_SIZE = config.TEST_SIZE


# Updated LinearRegression class with bias term
class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.cost_history = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)

        for _ in range(self.num_iterations):
            y_pred = np.dot(X, self.weights)
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            self.weights -= self.learning_rate * dw
            cost = (1 / (2 * num_samples)) * np.sum(np.square(y_pred - y))
            self.cost_history.append(cost)

    def predict(self, X):
        return np.dot(X, self.weights)

def plot_data_and_regression(X, y, predictions, model):
    """Plot data and regression line using Plotly.

    Parameters:
        X (ndarray): Input features.
        y (ndarray): Target values.
        predictions (ndarray): Predicted values.
        model (LinearRegressionWithBias): Trained model for accessing cost history.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add actual data points
    fig.add_trace(go.Scatter(x=X[:, 1], y=y, mode='markers', name='Actual Data'))

    # Sort the data by X[:, 1] for plotting the regression line
    sorted_indices = X[:, 1].argsort()
    sorted_x = X[sorted_indices]
    sorted_predictions = predictions[sorted_indices]

    # Add regression line
    fig.add_trace(go.Scatter(x=sorted_x[:, 1], y=sorted_predictions, mode='lines', name='Linear Regression Line'))

    fig.update_layout(title='Linear Regression',
                      xaxis_title='X',
                      yaxis_title='y',
                      legend=dict(x=0, y=1, bordercolor="Black", borderwidth=2))

    fig.show()


def run_linear_regression_california():
    # Load the California housing dataset
    california_housing = fetch_california_housing()
    X, y = california_housing.data, california_housing.target

    # Preprocess the data using the imported function
    X_scaled = preprocess_data(X)

    # Add bias term
    X_with_bias = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_with_bias, y, test_size=TEST_SIZE, random_state=42)

    # Create and train the linear regression model with bias
    model = LinearRegression(learning_rate=LEARNING_RATE, num_iterations=NUM_ITERATIONS)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Plot the linear regression line and data points
    plot_data_and_regression(X_test, y_test, predictions, model)

    print("Mean Squared Error: {:.2f}".format(mse))
    print("R-squared: {:.2f}".format(r2))

if __name__ == "__main__":
    start_time = time.time()
    run_linear_regression_california()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time: {:.2f} seconds".format(execution_time))
