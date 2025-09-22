# src/predict.py
from __future__ import annotations
from functools import lru_cache
import numpy as np
import joblib
from .train import MODEL_PATH  

@lru_cache(maxsize=1)
def _load_model():
    """
    Load the trained model once and cache it.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it first:\n  python -m src.train"
        )
    return joblib.load(MODEL_PATH)

def predict_data(X):
    """
    Predict the class labels for the input data.

    Args:
        X: array-like of shape (n_samples, n_features)
           For Digits, n_features = 64 (flattened 8x8).

    Returns:
        np.ndarray: Predicted class labels of shape (n_samples,).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return _load_model().predict(X)