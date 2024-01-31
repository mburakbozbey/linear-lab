import time
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource

# Constants
LEARNING_RATE = 0.01
NUM_ITERATIONS = 1000
TEST_SIZE = 0.2

# Updated LinearRegression class with bias term
class LinearRegressionWithBias:
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
    """Plot data and regression line using Bokeh.

    Parameters:
        X (ndarray): Input features.
        y (ndarray): Target values.
        predictions (ndarray): Predicted values.
        model (LinearRegressionWithBias): Trained model for accessing cost history.
    """
    source_actual = ColumnDataSource(data=dict(x=X[:, 1], y=y))
    source_predicted = ColumnDataSource(data=dict(x=X[:, 1], y=predictions))

    p = figure(title="Linear Regression", x_axis_label="X", y_axis_label="y")

    p.scatter("x", "y", source=source_actual, size=8, color="black", legend_label="Actual Data")
    p.line("x", "y", source=source_predicted, line_width=2, line_color="navy", legend_label="Linear Regression Line")

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    cost_history_plot = figure(title="Cost Function Over Time", x_axis_label="Iterations", y_axis_label="Cost")
    cost_history_plot.line(range(len(model.cost_history)), model.cost_history, line_width=2, line_color="green")

    show(gridplot([[p], [cost_history_plot]]))

def run_linear_regression_california():
    # Load and preprocess the California housing dataset
    california_housing = fetch_california_housing()
    X, y = california_housing.data, california_housing.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_with_bias = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_with_bias, y, test_size=TEST_SIZE, random_state=42)

    # Create and train the linear regression model with bias
    model = LinearRegressionWithBias(learning_rate=LEARNING_RATE, num_iterations=NUM_ITERATIONS)
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
