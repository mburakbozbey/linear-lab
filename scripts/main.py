import time

import numpy as np
import plotly.graph_objects as go
import yaml
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import model and utility functions
from algorithms.linear_regression import LinearRegression
from utils.plotting import plot_data_and_regression
from utils.preprocessing import preprocess_data


def run_experiment(config_path):
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load the dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # Preprocess the data
    X = preprocess_data(X, config["preprocessing"])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # Initialize and train the model
    model = LinearRegression(
        learning_rate=config["model"]["learning_rate"],
        num_iterations=config["model"]["num_iterations"],
    )
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Plotting
    plot_data_and_regression(X_test, y_test, predictions, model, config["plot"])

    # Print metrics
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")


if __name__ == "__main__":
    start_time = time.time()
    run_experiment("config/linear_regression.yaml")
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
