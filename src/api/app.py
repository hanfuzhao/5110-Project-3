"""
FastAPI application that serves the trained forecasting model.

Environment variables:
- MODEL_PATH: path to the serialized model (default artifacts/models/linear_regression.joblib)
- SCHEMA_PATH: path to the feature schema json (default artifacts/models/feature_schema.json)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/models/linear_regression.joblib"))
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", "artifacts/models/feature_schema.json"))


class PredictionRequest(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    predictions: Dict[str, float]


def load_schema(schema_path: Path) -> Dict:
    if not schema_path.exists():
        raise RuntimeError(f"Feature schema not found at {schema_path}")
    with schema_path.open() as f:
        return json.load(f)


def load_model(model_path: Path):
    if not model_path.exists():
        raise RuntimeError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


schema = load_schema(SCHEMA_PATH)
model = load_model(MODEL_PATH)
feature_names: List[str] = schema["feature_names"]
target_columns: List[str] = schema["target_columns"]

app = FastAPI(title="E-commerce Forecast API", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    missing = [feat for feat in feature_names if feat not in request.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    row = {feat: request.features[feat] for feat in feature_names}
    df = pd.DataFrame([row])

    preds = model.predict(df)[0]
    predictions = {target: float(value) for target, value in zip(target_columns, preds)}
    return PredictionResponse(predictions=predictions)

