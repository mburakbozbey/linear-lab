import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_data(X):
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
