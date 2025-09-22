# src/data.py
import numpy as np
from typing import Tuple
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Digits dataset and return the features and target values.

    Returns:
        X (numpy.ndarray): Features of shape (n_samples, 64) — flattened 8x8 pixels.
        y (numpy.ndarray): Target labels of shape (n_samples,) in {0..9}.
    """
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y

def split_data(X: np.ndarray, y: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets.

    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.

    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12
    )
    return X_train, X_test, y_train, y_test
