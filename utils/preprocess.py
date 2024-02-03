"""
Preprocesses the input data by identifying columns with NaN values,
handling missing values for numerical columns,
standardizing the data for numerical columns,
and returning the preprocessed data.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_data(x):
    """
    Preprocesses the input data by identifying columns with NaN values,
    handling missing values for numerical columns,
    standardizing the data for numerical columns,
    and returning the preprocessed data.

    Parameters:
    x (array-like): The input data to be preprocessed.

    Returns:
    array-like: The preprocessed input data.
    """
    # Step 1: Identify columns with NaN values
    nan_columns = np.isnan(np.array(x)).any(axis=0)

    # Step 2: Handle missing values for numerical columns
    for col_idx, has_nan in enumerate(nan_columns):
        if has_nan:
            imputer = SimpleImputer(strategy="mean")
            x[:, col_idx] = imputer.fit_transform(
                x[:, col_idx].reshape(-1, 1)
            ).flatten()

    # Step 3: Standardize the data for numerical columns
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    return x


def plot_data_and_regression(x, y, predictions=None):
    """Plot data for all features on a single page using Plotly subplots.

    Parameters:
        x (ndarray): Input features with shape (n_samples, n_features).
        y (ndarray): Target values.
        predictions (ndarray, optional): Predicted values. If None, no predictions are plotted.
    """
    n_features = x.shape[1]

    # Determine the layout of subplots
    rows = int(np.ceil(np.sqrt(n_features)))
    cols = rows

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Feature {i+1}" for i in range(n_features)],
    )

    for i in range(n_features):
        row = (i // cols) + 1
        col = (i % cols) + 1

        # Sort the data by the current feature for plotting
        sorted_indices = x[:, i].argsort()
        x_sorted = x[sorted_indices, i]
        y_sorted = y[sorted_indices]

        # Add actual data points
        fig.add_trace(
            go.Scatter(x=x_sorted, y=y_sorted, mode="markers", name=f"Actuals F{i+1}"),
            row=row,
            col=col,
        )

        # Optionally, add predictions if available
        if predictions is not None:
            predictions_sorted = predictions[sorted_indices]
            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=predictions_sorted,
                    mode="markers",
                    name=f"Predictions F{i+1}",
                ),
                row=row,
                col=col,
            )

    # Update plot layout
    fig.update_layout(
        height=1000, width=1000, title_text="All Features vs Target", showlegend=False
    )
    fig.show()
