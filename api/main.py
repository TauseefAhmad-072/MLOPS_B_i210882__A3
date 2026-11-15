"""Containerised FastAPI inference service (the application the agent operates in Kubernetes)."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

log = logging.getLogger("inference-api")
logging.basicConfig(level=logging.INFO, format='{"level":"%(levelname)s","msg":"%(message)s"}')

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
FAIL_READINESS = os.getenv("FAIL_READINESS", "false").lower() == "true"

_model = None


def load_model():
    global _model
    if not Path(MODEL_PATH).exists():
        log.error("model file not found at %s", MODEL_PATH)
        _model = None
        return
    _model = joblib.load(MODEL_PATH)
    log.info("model loaded from %s", MODEL_PATH)


@asynccontextmanager
async def lifespan(_app):
    load_model()
    yield


app = FastAPI(title="Customer-churn inference API", version=APP_VERSION, lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")  # liveness: process is up
def health():
    return {"status": "alive"}


@app.get("/ready")  # readiness: model loaded
def ready():
    if FAIL_READINESS or _model is None:
        log.error("readiness failed: model_loaded=%s fail_flag=%s", _model is not None, FAIL_READINESS)
        raise HTTPException(503, "model not ready")
    return {"status": "ready"}


@app.get("/version")
def version():
    return {"app_version": APP_VERSION, "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(503, "model not loaded")
    n = _model.n_features_in_
    if len(req.features) != n:
        raise HTTPException(422, f"expected {n} features, got {len(req.features)}")
    proba = float(_model.predict_proba([req.features])[0][1])
    return {"prediction": int(proba >= 0.5), "probability": round(proba, 4), "model_version": MODEL_VERSION}
