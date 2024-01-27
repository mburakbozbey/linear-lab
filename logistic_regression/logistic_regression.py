# %%
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource

# Constants
LEARNING_RATE = 0.01
NUM_ITERATIONS = 1000
DATA_SIZE = 100
DATA_SCALE = 10
TEST_SIZE = 0.2

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape

        if num_features == 1:
            X = np.column_stack((np.ones((num_samples, 1)), X))  # Add a column of ones for bias

        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            cost = (-1 / num_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            self.cost_history.append(cost)

    def predict(self, X):
        num_samples, num_features = X.shape

        if num_features == 1:
            X = np.column_stack((np.ones((num_samples, 1)), X))  # Add a column of ones for bias

        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class

def generate_data(size, scale):
    # Creating a linearly separable dataset with hardcoded values
    X_class_1 = np.array([[2], [3], [4], [5], [6], [7], [8], [9], [10], [11]])
    X_class_2 = np.array([[12], [13], [14], [15], [16], [17], [18], [19], [20], [21]])
    X = np.vstack((X_class_1, X_class_2))
    y = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
    return X, y

def plot_data_and_decision_boundary(X, y, model):
    source = ColumnDataSource(data=dict(x=X.flatten(), y=y.flatten(), color=['blue' if val == 0 else 'green' for val in y.flatten()]))

    p = figure(title="Logistic Regression", x_axis_label="X", y_axis_label="y")

    p.scatter("x", "y", source=source, size=8, fill_color='color', legend_label="Class", line_color='black')

    # Step 3: Use Sigmoid for Decision Boundary
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.sigmoid(x_range * model.weights[1] + model.bias)

    p.line(x_range, y_range, line_width=2, line_color="navy", legend_label="Decision Boundary")

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    cost_history_plot = figure(title="Cost Function Over Time", x_axis_label="Iterations", y_axis_label="Cost")
    cost_history_plot.line(range(len(model.cost_history)), model.cost_history, line_width=2, line_color="green")

    show(gridplot([[p], [cost_history_plot]]))

def run_logistic_regression():
    X, y = generate_data(DATA_SIZE, DATA_SCALE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    model = LogisticRegression(learning_rate=LEARNING_RATE, num_iterations=NUM_ITERATIONS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if model.weights is not None and len(model.weights) > 0:
        plot_data_and_decision_boundary(X_test, y_test, model)
    else:
        print("Model is not properly trained. Please check your training data.")

    print("Accuracy: {:.2f}%".format(accuracy * 100))

if __name__ == "__main__":
    start_time = time.time()
    run_logistic_regression()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time: {:.2f} seconds".format(execution_time))

# %%
