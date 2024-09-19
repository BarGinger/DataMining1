"""
File: Tree.py
Student 1: Bar Melinarskiy, student number: 2482975
Student 2: Prathik Alex Matthai, student number: 9020675
Student 3: Mohammed Bashabeeb, student number: 7060424
Date: September 12, 2024
Description: Assignment 1 - Classification Trees, Bagging and Random Forests
Tree class object
"""

import numpy as np
from Node import Node
from tqdm import tqdm
import copy
from collections import Counter


def calc_standard_error(y_pred: np.ndarray, y_true: np.ndarray):
    # Check if the lengths of y_pred and y_true are the same
    if len(y_pred) != len(y_true):
        raise ValueError("y_pred and y_true must have the same length")

    # Count the number of mismatches
    num_mismatches = np.sum(y_pred != y_true)
    Rts = num_mismatches / len(y_true)
    standard_error = np.sqrt(Rts * (1 - Rts) / len(y_true))
    return standard_error


def traverse_the_tree(node: Node, x: np.ndarray) -> Node:
    """
            Traverse the given tree starting from the root node and navigate to the leaf node
             that corresponds to the given data.

            Parameters
            ----------
            @param node - the current node we wish to test the data with
            x : {array-like, sparse matrix} of shape (n_samples, n_features)
                The input sample.

            Returns
            -------
            @return the appropriate leaf node
            """

    if node.get_threshold() == float('-inf'):
        return node

    if node.apply_threshold(x):
        return traverse_the_tree(node.get_right_son(), x)

    return traverse_the_tree(node.get_left_son(), x)


class Tree(object):
    """A classification tree object

    Attributes:
        nmin    - The number of observations that a node must contain at least, for it to be allowed to be split.
        minleaf - The minimum number of observations required for a leaf node.
        nfeat - The number of features that should be considered for each split.
        impurity_func - The impurity function to be used while building the tree.
    """

    def __init__(self, nmin: int, minleaf: int, nfeat: int, impurity_func="gini_index"):
        self.root = None
        self.nmin = nmin
        self.minleaf = minleaf
        self.nfeat = nfeat
        self.impurity_func = impurity_func

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Build a decision tree classifier from the training set (X, y).

        @param X: a data matrix (2-dimensional array) containing the attribute values.
        Each row of x contains the attribute values of one training example.
        @param y: the vector (1-dimensional array) of class labels.
        The class label is binary, with values coded as 0 and 1.
        """
        self.root = Node(x=X, y=y, impurity_func=self.impurity_func)
        node_list = [self.root]
        pbar = tqdm(desc="Building tree", unit="nodes", position=0, leave=True, delay=2)

        while len(node_list) > 0:
            current_node = node_list.pop()
            unique_classes = np.unique(current_node.get_classes())
            # to do a split the current node needs to be impure and must contain at least nmin observations
            # Additionally, check that there is more than one class in the node,
            # if not there is no need to continue splitting
            if (current_node.get_impurity() > 0 and len(current_node.get_data()) >= self.nmin
                    and len(unique_classes) > 1):
                new_node_left, new_node_right = self.find_best_split(current_node)
                if new_node_left and new_node_right:
                    node_list.extend([new_node_left, new_node_right])
                    current_node.set_right_son(new_node_right)
                    current_node.set_left_son(new_node_left)

            pbar.update(1)  # Update progress bar
        pbar.close()

    def find_best_split(self, current_node) -> (Node, Node):
        data = copy.deepcopy(current_node.get_data())
        classes = current_node.get_classes()
        features_count = data.shape[1]
        # check if we are in a random forest settings, meaning we need to randomly select self.nfeat feathers
        indices = range(features_count)
        if self.nfeat < features_count:
            indices = np.random.choice(features_count, size=self.nfeat, replace=False)

        # cols_values, cols_values_counts = current_node.get_unique_values()
        current_impurity = current_node.get_impurity()
        max_impurity = float('-inf')
        new_node_left, new_node_right = None, None

        for col in indices:
            # In class, we discussed that if we calculate the probability (in two classes case)
            # of each distinct value and sort the array accordingly
            # then there is no need to check consecutive values that have the same probability -
            # meaning the optimal solution will not be there
            col_values = data[:, col]
            col_unique_vals = np.unique(col_values, axis=0)
            # probs = np.array([np.mean(classes[col_values == val] == 0) for val in col_unique_vals])
            # # sort in ascending order p(0|x=l1) <= p(0|x=l2) <= .... <= p(0|x=lL)
            # sorted_indices = np.argsort(probs)
            # probs_sorted = probs[sorted_indices]
            # col_unique_vals_sorted = col_unique_vals[sorted_indices]
            #
            # # Leave only first instances of unique probability values
            # unique_probs, unique_indices = np.unique(probs_sorted, return_index=True)
            # unique_probs_values = col_unique_vals_sorted[unique_indices]
            for i, value in enumerate(col_unique_vals):
                new_node_left, new_node_right, max_impurity = self.calc_split_for_column(data, classes, col, value,
                                                                                         current_node, current_impurity,
                                                                            new_node_left, new_node_right, max_impurity)

        return new_node_left, new_node_right

    def calc_split_for_column(self, data, classes, col, value, current_node, current_impurity,
                              new_node_left, new_node_right, max_impurity):
        filter_arr = data[:, col] <= value
        right_data, right_classes = data[filter_arr], classes[filter_arr]
        left_data, left_classes = data[~filter_arr], classes[~filter_arr]

        # Check the condition of minleaf
        # The minimum number of observations required for a leaf node.
        if len(left_data) < self.minleaf or len(right_data) < self.minleaf:
            return new_node_left, new_node_right, max_impurity

        # Create left and right node looking for best split using the impurity_func
        # (gini-index) to determining the quality of a split.
        left_node = Node(x=left_data, y=left_classes, impurity_func=self.impurity_func)
        right_node = Node(x=right_data, y=right_classes, impurity_func=self.impurity_func)
        left_impurity = left_node.get_impurity() * (len(left_data) / len(data))
        right_impurity = right_node.get_impurity() * (len(right_data) / len(data))
        delta_impurity = current_impurity - (left_impurity + right_impurity)
        if max_impurity < delta_impurity:
            max_impurity = delta_impurity
            new_node_left, new_node_right = left_node, right_node
            current_node.set_threshold(threshold=value)
            current_node.set_y_index(index=col)
        return new_node_left, new_node_right, max_impurity


    def predict(self, X: np.ndarray):
        """Predict class for X.

        For a classification model, the predicted class for each sample in X is
        returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """

        if self.root is None:
            raise ValueError("The current tree has not been fitted yet.")

        y_pred = np.array([traverse_the_tree(self.root, x).get_majority_vote() for x in X])
        return y_pred

    def print_tree(self, classes_names: np.ndarray, file_path=None):
        """
        Print the decision tree structure.
        """
        if self.root is not None:
            file = None
            if file_path:
                file = open(file_path, 'w')
            try:
                self._print_tree_structure(self.root, depth=0, classes_names=classes_names, file=file)
            finally:
                if file:
                    file.close()
        else:
            print("The tree has not been fitted yet.")

    def _print_tree_structure(self, node: Node, depth: int, classes_names=None, file=None):
        """
        Recursively print the decision tree structure.
        """
        if node is None:
            return

        indent = "|   " * depth
        if depth > 0:
            indent += "|--- "

        labels = node.get_classes()
        label_counts = Counter(labels)
        label_counts_str = ', '.join(f"{label}: {count}" for label, count in label_counts.items())

        if node.get_left_son() is not None:
            class_name = node.get_y_index()
            if classes_names is not None:
                class_name = classes_names[node.get_y_index()]

            line = f"{indent}{class_name} <= {node.get_threshold()}, # of samples: {len(node.get_data())} ({label_counts_str})"
        else:
            line = f"{indent}class: {node.get_majority_vote()}, # of samples: {len(node.get_data())} ({label_counts_str})"



        if file:
            file.write(line + "\n")
        else:
            print(line)

        if node.get_left_son() is not None:
            self._print_tree_structure(node.get_left_son(), depth + 1, classes_names=classes_names, file=file)
        if node.get_right_son() is not None:
            self._print_tree_structure(node.get_right_son(), depth + 1, classes_names=classes_names, file=file)
