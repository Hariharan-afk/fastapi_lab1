# src/train.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import joblib

from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import load_data, split_data  

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "digits_model.pkl"   

def fit_model(X_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    Train a classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", gamma="scale", C=3.0, probability=True, random_state=42)),
    ])
    clf.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)