# src/main.py
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from src.predict import predict_data          
from src.data import load_data, split_data   

app = FastAPI(title="Digits Classifier")

# ---------- Schemas ----------
class DigitsData(BaseModel):
    # 8x8 image flattened to 64 grayscale values (0..16)
    pixels: List[float] = Field(..., min_length=64, max_length=64)

class PredictionResponse(BaseModel):
    response: int 


# ---------- Health ----------
@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


# ---------- Metrics ----------
@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    try:
        y_pred = predict_data(X_test)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not available or failed to predict: {e}")
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    classes, counts = np.unique(y, return_counts=True)
    class_counts = {int(c): int(cnt) for c, cnt in zip(classes, counts)}
    return {
        "test_accuracy": round(acc, 4),
        "n_test": int(X_test.shape[0]),
        "class_counts": class_counts,
        "confusion_matrix": cm
    }


# ---------- dataset_info ----------
@app.get("/dataset_info")
def dataset_info() -> Dict[str, Any]:
    X, y = load_data()
    return {
        "dataset": "sklearn.load_digits",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_shape": [8, 8],
        "classes": list(range(10)),
    }


# ---------- sample_info ----------
@app.get("/sample/{i}")
def sample(i: int) -> Dict[str, Any]:
    X, y = load_data()
    n = X.shape[0]
    if i < 0 or i >= n:
        raise HTTPException(status_code=404, detail=f"Index {i} out of range [0, {n-1}]")
    flat = X[i].tolist()
    img2d = np.array(flat, dtype=float).reshape(8, 8).tolist()
    return {"index": i, "label": int(y[i]), "pixels_flat": flat, "pixels_8x8": img2d}


# ---------- Predict ----------
@app.post("/predict", response_model=PredictionResponse)
async def predict_digit(payload: DigitsData):
    try:
        y_pred = predict_data([payload.pixels])          # shape: (1, 64)
        return PredictionResponse(response=int(y_pred[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
