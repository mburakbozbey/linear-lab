# %%

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource

# Constants
LEARNING_RATE = 0.01
NUM_ITERATIONS = 1000
DATA_SIZE = 100
DATA_SCALE = 10
TEST_SIZE = 0.2

class LinearRegression:
    """Linear regression model.

    Parameters:
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros((num_features, 1))
        self.bias = 0

        for _ in range(self.num_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            cost = (1 / (2 * num_samples)) * np.sum(np.square(y_pred - y))
            self.cost_history.append(cost)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def generate_data(size, scale):
    """Generate realistic data.

    Parameters:
        size (int): Size of the dataset.
        scale (float): Scaling factor for data.

    Returns:
        X (ndarray): Input features.
        y (ndarray): Target values.
    """
    np.random.seed(0)
    X = np.random.rand(size, 1) * scale
    y = 2 * X + 1 + np.random.randn(size, 1)
    return X, y

def plot_data_and_regression(X, y, predictions):
    """Plot data and regression line using Bokeh.

    Parameters:
        X (ndarray): Input features.
        y (ndarray): Target values.
        predictions (ndarray): Predicted values.
    """
    source_actual = ColumnDataSource(data=dict(x=X.flatten(), y=y.flatten()))
    source_predicted = ColumnDataSource(data=dict(x=X.flatten(), y=predictions.flatten()))

    p = figure(title="Linear Regression", x_axis_label="X", y_axis_label="y")

    p.scatter("x", "y", source=source_actual, size=8, color="black", legend_label="Actual Data")
    p.line("x", "y", source=source_predicted, line_width=2, line_color="navy", legend_label="Linear Regression Line")

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    cost_history_plot = figure(title="Cost Function Over Time", x_axis_label="Iterations", y_axis_label="Cost")
    cost_history_plot.line(range(len(model.cost_history)), model.cost_history, line_width=2, line_color="green")

    show(gridplot([[p], [cost_history_plot]]))

def run_linear_regression():
    # Generate realistic data
    X, y = generate_data(DATA_SIZE, DATA_SCALE)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression(learning_rate=LEARNING_RATE, num_iterations=NUM_ITERATIONS)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Plot the linear regression line and data points using Bokeh
    plot_data_and_regression(X_test, y_test, predictions)

    print("Mean Squared Error: {:.2f}".format(mse))
    print("R-squared: {:.2f}".format(r2))

if __name__ == "__main__":
    start_time = time.time()
    run_linear_regression()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time: {:.2f} seconds".format(execution_time))

# %%
