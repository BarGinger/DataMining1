"""
File: main.py
Student 1: Bar Melinarskiy, student number: 2482975
Student 2: Prathik Alex Matthai, student number: 9020675
Student 3: Mohammed Bashabeeb, student number: 7060424
Date: September 12, 2024
Description: Assignment 1 - Classification Trees, Bagging and Random Forests
"""

import numpy as np
from Tree import Tree
from typing import List
from tqdm import tqdm


def tree_grow(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int) -> Tree:
    """
    Create a tree from the given training data and restorations

    @param x: a data matrix (2-dimensional array) containing the attribute values.
    Each row of x contains the attribute values of one training example. You may assume that all attributes are numeric.
    @param y: the vector (1-dimensional array) of class labels. The class label is binary, with values coded as 0 and 1.
    @param nmin: the number of observations that a node must contain at least, for it to be allowed to be split.
     In other words: if a node contains fewer cases than nmin, it becomes a leaf node.
    @param minleaf: the minimum number of observations required for a leaf node; hence a split that creates a node
     with fewer than minleaf observations is not acceptable. If the algorithm performs a split, it should be the best
     split that meets the minleaf constraint.
     If there is no split that meets the minleaf constraint, the node becomes a leaf node.
    @param nfeat: the number of features that should be considered for each split. Every time we compute the best split
     in a particular node, we first draw at random nfeat features from which the best split is to be selected.
     For normal tree growing, nfeat is equal to the total number of predictors (the number of columns of x).
     For random forests, nfeat is smaller than the total number of predictors.
    @return tree object that can be used for predicting new cases
    """
    # Declare the new tree
    tree = Tree(nmin=nmin, minleaf=minleaf, nfeat=nfeat)
    # Fit the tree (build it) on the given training set
    tree.fit(X=x, y=y)
    return tree


def tree_pred(x: np.ndarray, tr: Tree) -> np.ndarray:
    """
    Predict the class of the given input data using the given tree object

    @param x: A data matrix (2-dimensional array) containing the attribute values of the cases for which
     predictions are required,
    @param tr: is a tree object created with the function tree_grow. The function
    @return y:np.ndarray, which is the vector (1-dimensional array) of predicted class labels for the cases in x,
    that is, y[i] contains the predicted class label for row i of x.
    """

    return tr.predict(X=x)


def tree_grow_b(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int, m: int):
    """
    Bagging for creating a tree from the given training data and restorations

    @param x: a data matrix (2-dimensional array) containing the attribute values.
    Each row of x contains the attribute values of one training example. You may assume that all attributes are numeric.
    @param y: the vector (1-dimensional array) of class labels. The class label is binary, with values coded as 0 and 1.
    @param nmin: the number of observations that a node must contain at least, for it to be allowed to be split.
     In other words: if a node contains fewer cases than nmin, it becomes a leaf node.
    @param minleaf: the minimum number of observations required for a leaf node; hence a split that creates a node
     with fewer than minleaf observations is not acceptable. If the algorithm performs a split, it should be the best
     split that meets the minleaf constraint.
     If there is no split that meets the minleaf constraint, the node becomes a leaf node.
    @param nfeat: the number of features that should be considered for each split. Every time we compute the best split
     in a particular node, we first draw at random nfeat features from which the best split is to be selected.
     For normal tree growing, nfeat is equal to the total number of predictors (the number of columns of x).
     For random forests, nfeat is smaller than the total number of predictors.
    @param m: denotes the number of bootstrap samples to be drawn. On each bootstrap sample a tree is grown.
    @return a list containing these m trees.
    """

    # Get the size of the training set
    training_size = len(x)
    trees = []

    # create m trees
    for i in tqdm(range(m), desc=f"Creating {m} trees", unit="trees"):
        # Draw a sample with replacement from the training set.
        # The sample should be of the same size as the training set.
        print(f"\nCreating tree-{i+1}", flush=True)
        indices = np.random.choice(training_size, size=training_size, replace=True)
        # Select corresponding values from x and y arrays
        x_i = x[indices]
        y_i = y[indices]
        # train the i-th tree
        tree = tree_grow(x=x_i, y=y_i, nmin=nmin, minleaf=minleaf, nfeat=nfeat)
        # save the i-th tree to the return array
        trees.append(tree)

    return trees


def tree_pred_b(x: np.ndarray, tr: List[Tree]) -> np.ndarray:
    """
    This function applies tree_pred to x using each tree in the list in turn.
     For each row of x the final prediction is obtained by taking the majority vote of the m predictions.
     The function returns a vector y, where y[i] contains the predicted class label for row i of x.
    Predict the class of the given input data using the given tree object

    @param x: A data matrix (2-dimensional array) containing the attribute values of the cases for which
     predictions are required,
    @param tr: a list of trees created using the tree_grow_b function
    @return y:np.ndarray, which is the vector (1-dimensional array) of predicted class labels for the cases in x,
    that is, y[i] contains the predicted class label for row i of x.
    """

    # predict using each given tree
    predictions = np.array([tree_pred(x=x, tr=tree) for tree in tr])
    # Get the most frequent class for each sample
    # np.bincount - Count number of occurrences of each value in array of non-negative ints
    majority_vote = np.array([np.argmax(np.bincount(column)) for column in predictions.T])
    return majority_vote
