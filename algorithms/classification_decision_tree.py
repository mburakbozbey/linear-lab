import csv
import os
import random


# Define a TreeNode class for the decision tree
class TreeNode:
    def __init__(
        self, data_class=None, split_feature=None, threshold=None, left=None, right=None
    ):
        """
        Initializes a node in a decision tree.

        Parameters:
            data_class (optional): The class label if it's a leaf node
            split_feature (optional): The feature used for splitting
            threshold (optional): Threshold value for splitting
            left (optional): Left subtree
            right (optional): Right subtree
        """
        self.data_class = data_class  # The class label if it's a leaf node
        self.split_feature = split_feature  # The feature used for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.left = left  # Left subtree
        self.right = right  # Right subtree


# Define the DecisionTreeClassifier class
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        """
        Initializes the class with an optional maximum depth parameter.

        Parameters:
            max_depth (int, optional): Maximum depth of the tree

        Returns:
            None
        """
        self.max_depth = max_depth  # Maximum depth of the tree
        self.root = None  # The root node of the decision tree

    def fit(self, X, y, depth=0):
        if depth == self.max_depth or len(set(y)) == 1:
            # If the tree reaches the maximum depth or all labels are the same, create a leaf node
            data_class = max(set(y), key=y.count)
            return TreeNode(data_class=data_class)

        num_features = len(X[0])
        best_split_feature = None
        best_threshold = None
        best_gini = float("inf")

        for feature in range(num_features):
            values = list(set(X[i][feature] for i in range(len(X))))
            for val in values:
                left_indices = [i for i in range(len(X)) if X[i][feature] <= val]
                right_indices = [i for i in range(len(X)) if X[i][feature] > val]
                left_labels = [y[i] for i in left_indices]
                right_labels = [y[i] for i in right_indices]

                gini = self.calculate_gini(left_labels, right_labels)
                if gini < best_gini:
                    best_gini = gini
                    best_split_feature = feature
                    best_threshold = val
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        if best_gini == float("inf"):
            # No suitable split found, create a leaf node
            data_class = max(set(y), key=y.count)
            return TreeNode(data_class=data_class)

        # Recursively build left and right subtrees
        left_subtree = self.fit(
            [X[i] for i in best_left_indices],
            [y[i] for i in best_left_indices],
            depth + 1,
        )
        right_subtree = self.fit(
            [X[i] for i in best_right_indices],
            [y[i] for i in best_right_indices],
            depth + 1,
        )

        return TreeNode(
            split_feature=best_split_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def calculate_gini(self, left_labels, right_labels):
        """
        Calculate the Gini impurity for a split given the labels on the left and right nodes.

        Parameters:
            left_labels (list): The labels on the left node.
            right_labels (list): The labels on the right node.

        Returns:
            float: The Gini impurity for the split.
        """
        total_count = len(left_labels) + len(right_labels)
        gini_left = 1.0 - sum(
            [(left_labels.count(c) / len(left_labels)) ** 2 for c in set(left_labels)]
        )
        gini_right = 1.0 - sum(
            [
                (right_labels.count(c) / len(right_labels)) ** 2
                for c in set(right_labels)
            ]
        )
        gini = (len(left_labels) / total_count) * gini_left + (
            len(right_labels) / total_count
        ) * gini_right
        return gini

    def predict(self, X):
        """
        Predicts the output for each sample in the input X using the decision tree model.

        Parameters:
            X (array-like): The input samples.

        Returns:
            list: The predictions for each input sample.
        """
        predictions = []
        for sample in X:
            predictions.append(self.predict_single(sample, self.root))
        return predictions

    def predict_single(self, sample, node):
        """
        Predicts the class for a single sample using the decision tree node.

        Args:
            sample: The input sample to predict the class for.
            node: The decision tree node to start the prediction from.

        Returns:
            The predicted class for the input sample.
        """
        if node.data_class is not None:
            return node.data_class
        if sample[node.split_feature] <= node.threshold:
            return self.predict_single(sample, node.left)
        else:
            return self.predict_single(sample, node.right)


# Load data from the CSV file in the "data" folder
def load_data_from_csv(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing two lists: X (list of lists) and y (list).
    """
    X = []
    y = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            X.append([float(val) for val in row[:-1]])
            y.append(int(row[-1]))
    return X, y


# Define the main function
def main():
    """
    A function to load a dataset from a CSV file, split the data into training and testing sets,
    create and train a decision tree classifier, make predictions, and calculate accuracy.
    """
    # Load the dataset
    data_folder = "data"
    data_file = "dataset.csv"
    file_path = os.path.join(data_folder, data_file)
    X, y = load_data_from_csv(file_path)

    # Split the data into a training set and a testing set (you can use a more advanced method here)
    random.seed(42)
    random.shuffle(X)
    random.seed(42)
    random.shuffle(y)
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Create and train the decision tree classifier
    max_depth = 5  # You can change the max_depth as needed
    classifier = DecisionTreeClassifier(max_depth=max_depth)
    classifier.root = classifier.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = classifier.predict(X_test)

    # Calculate accuracy
    accuracy = sum(
        1 for p, true_label in zip(predictions, y_test) if p == true_label
    ) / len(y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
