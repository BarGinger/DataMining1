"""
File: Node.py
Student 1: Bar Melinarskiy, student number: 2482975
Student 2: Prathik Alex Matthai, student number: 9020675
Student 3: Mohammed Bashabeeb, student number: 7060424
Date: September 12, 2024
Description: Assignment 1 - Classification Trees, Bagging and Random Forests
Node class object
"""

from typing import Optional
import numpy as np


class Node(object):
    """A classification tree node class

    Attributes:
        x:np.ndarray    - Node's data.
        y:np.ndarray    - Samples classes.
        left_son:Node    - The left son of the current node.
        right_son:Node - The right son of the current node.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, impurity_func: str,
                 left: Optional['Node'] = None, right: Optional['Node'] = None):
        self.data = x
        self.classes = y
        self.classes = y
        self.y_index = -1  # -1 marks as unset
        self.left_son = left
        self.right_son = right
        self.impurity_func = impurity_func
        self.impurity = 0
        self.threshold = float('-inf')  # -inf mark as unset
        if impurity_func == "entropy":
            self.entropy()
        else:
            self.gini_index()

    def get_impurity(self) -> float:
        """
        Get current node's impurity
        @return impurity score
        """
        return self.impurity

    def get_threshold(self) -> float:
        """
        Get current node's threshold
        @return threshold
        """
        return self.threshold

    def get_data(self) -> np.ndarray:
        """
        Get current node's data
        @return data
        """
        return self.data

    # def get_data(self, num_samples: Optional[int]) -> np.ndarray:
    #     """
    #     Get current node's data
    #     @return data
    #     """
    #     if num_samples is None or num_samples >= len(self.data):
    #         return self.data
    #     return np.random.choice(self.data, size=num_samples, replace=False)

    def get_unique_values(self) -> (np.ndarray, np.ndarray):
        """
        Get unique values and counts of the node's data
        @return values, counts
        """
        return np.unique(self.data, axis=0, return_counts=True)

    def get_classes(self) -> np.ndarray:
        """
        Get current node's classes
        @return classes
        """
        return self.classes

    def get_unique_classes(self) -> (np.ndarray, np.ndarray):
        """
        Get unique values and counts of the node's classes
        @return values, counts
        """
        return np.unique(self.classes, return_counts=True)

    def get_left_son(self):
        """
        Get current node's left_son
        @return left_son
        """
        return self.left_son

    def get_right_son(self):
        """
        Get current node's right_son
        @return right_son
        """
        return self.right_son

    def get_majority_vote(self) -> float:
        """
        Get current node's classes majority vote
        @return majority vote
        """
        return np.argmax(np.bincount(self.classes))

    def apply_threshold(self, x: np.ndarray) -> bool:
        """
        Apply the threshold on the given data (x >= threshold)
        @return True if x >= threshold, False otherwise
        """

        return x[self.get_y_index()] <= self.get_threshold()

    def get_indices_ge_threshold(self, X: np.ndarray) -> np.ndarray:
        """
        Get mask for indexes that are >= threshold
        @return  mask for indexes that are >= threshold
        """

        mask = X[:, self.get_y_index()] >= self.get_threshold()
        return mask

    def get_y_index(self) -> int:
        """
        Get current node's y index
        @return y index
        """
        return self.y_index

    def set_y_index(self, index: int):
        """
        Set current node's y index
        @param index: y index
        """
        self.y_index = index

    def set_left_son(self, son=None):
        """
        Set current node's left_son
        @param son:Node left son node
        """
        self.left_son = son

    def set_right_son(self, son=None):
        """
        Set current node's right_son
        @param son:Node right son node
        """
        self.right_son = son

    def set_threshold(self, threshold: float):
        """
        Set current node's threshold
        @param threshold
        """
        self.threshold = threshold

    def entropy(self):
        """
        Calc entropy impurity for current node's data
        """
        values, counts = self.get_unique_classes()
        probs = counts / len(self.data)
        self.impurity = (-1) * np.sum(probs * np.log2(probs))

    def gini_index(self):
        """
        Calc gini-index impurity for current node's data
        """
        values, counts = self.get_unique_classes()
        self.impurity = 1 - np.sum((counts / len(self.data)) ** 2)
