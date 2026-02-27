import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class SoilMetrics(BaseModel):
    nitrogen: float = Field(..., ge=0, description="Nitrogen content in soil")
    phosphorus: float = Field(..., ge=0, description="Phosphorus content in soil")
    potassium: float = Field(..., ge=0, description="Potassium content in soil")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    ph: float = Field(..., ge=0, le=14, description="Soil pH value")
    rainfall: float = Field(..., ge=0, description="Rainfall in mm")


class PredictionResponse(BaseModel):
    recommended_crop: str
    confidence: float = Field(..., ge=0, le=1)


class ModelNotFoundError(Exception):
    pass


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "crop_model.pkl"
CROP_DATASET_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"
LABEL_ENCODER_CANDIDATES = [
    BASE_DIR / "model" / "crop_label_encoder.pkl",
    BASE_DIR / "model" / "label_encoder.pkl",
]

app = FastAPI(
    title="Crop Recommendation API",
    description="Predict recommended crop from soil and climate metrics.",
    version="1.0.0",
)


def load_model(model_path: Path) -> Any:
    if not model_path.exists():
        raise ModelNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)


def _is_numeric_like(value: Any) -> bool:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        try:
            float(text)
            return True
        except ValueError:
            return False
    return False


@lru_cache(maxsize=1)
def load_crop_labels() -> list[str]:
    if not CROP_DATASET_PATH.exists():
        return []

    labels: set[str] = set()
    with CROP_DATASET_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            label = (row.get("label") or "").strip()
            if label:
                labels.add(label)
    return sorted(labels)


@lru_cache(maxsize=1)
def load_label_encoder() -> Any | None:
    for encoder_path in LABEL_ENCODER_CANDIDATES:
        if encoder_path.exists():
            try:
                return joblib.load(encoder_path)
            except Exception:
                return None
    return None


def decode_crop_label(predicted_value: Any) -> str:
    if isinstance(predicted_value, str) and not _is_numeric_like(predicted_value):
        return predicted_value

    encoder = load_label_encoder()
    if encoder is not None:
        try:
            decoded = encoder.inverse_transform([int(float(predicted_value))])[0]
            return str(decoded)
        except Exception:
            pass

    if _is_numeric_like(predicted_value):
        label_index = int(float(predicted_value))
        labels = load_crop_labels()
        if 0 <= label_index < len(labels):
            return labels[label_index]

    return str(predicted_value)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_crop(payload: SoilMetrics) -> PredictionResponse:
    try:
        model = load_model(MODEL_PATH)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model Not Found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to load model") from exc

    features = np.array(
        [
            [
                payload.nitrogen,
                payload.phosphorus,
                payload.potassium,
                payload.temperature,
                payload.humidity,
                payload.ph,
                payload.rainfall,
            ]
        ]
    )

    try:
        prediction = model.predict(features)
        recommended_crop = decode_crop_label(prediction[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))
        else:
            confidence = 1.0
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    return PredictionResponse(recommended_crop=recommended_crop, confidence=confidence)
