import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_data(X):
    """
    Preprocesses the input data by identifying columns with NaN values, separating categorical and non-categorical columns,
    handling missing values for categorical columns, standardizing the data for non-categorical columns, and returning the preprocessed data.

    Parameters:
    X (array-like): The input data to be preprocessed.

    Returns:
    array-like: The preprocessed input data.
    """
    # Step 1: Identify columns with NaN values
    nan_columns = np.isnan(X).any(axis=0)

    # Step 2: Separate categorical and non-categorical columns (replace with actual categorical column info)
    categorical_columns = np.array(
        [False] * X.shape[1]
    )  # Replace with actual categorical columns

    # Step 3: Handle missing values for categorical columns
    for col_idx, is_categorical in enumerate(categorical_columns):
        if nan_columns[col_idx]:
            if is_categorical:
                imputer = SimpleImputer(strategy="most_frequent")
                X[:, col_idx] = imputer.fit_transform(
                    X[:, col_idx].reshape(-1, 1)
                ).flatten()
            else:
                imputer = SimpleImputer(strategy="mean")
                X[:, col_idx] = imputer.fit_transform(
                    X[:, col_idx].reshape(-1, 1)
                ).flatten()

    # Step 4: Standardize the data for non-categorical columns
    non_categorical_columns = ~categorical_columns
    scaler = StandardScaler()
    X[:, non_categorical_columns] = scaler.fit_transform(X[:, non_categorical_columns])

    return X


def plot_data_and_regression(X, y, predictions=None):
    """Plot data for all features on a single page using Plotly subplots.

    Parameters:
        X (ndarray): Input features with shape (n_samples, n_features).
        y (ndarray): Target values.
        predictions (ndarray, optional): Predicted values. If None, no predictions are plotted.
    """
    n_features = X.shape[1]

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
        sorted_indices = X[:, i].argsort()
        X_sorted = X[sorted_indices, i]
        y_sorted = y[sorted_indices]

        # Add actual data points
        fig.add_trace(
            go.Scatter(x=X_sorted, y=y_sorted, mode="markers", name=f"Actuals F{i+1}"),
            row=row,
            col=col,
        )

        # Optionally, add predictions if available
        if predictions is not None:
            predictions_sorted = predictions[sorted_indices]
            fig.add_trace(
                go.Scatter(
                    x=X_sorted,
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
