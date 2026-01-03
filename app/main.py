import logging

from fastapi import FastAPI, HTTPException

from app.core.config import get_settings
from app.schemas import (
    HealthResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.predictor import Predictor

logger = logging.getLogger(__name__)

app = FastAPI(title="Ticket Router")

predictor = Predictor()
settings = get_settings()


@app.on_event("startup")
def load_model_on_startup() -> None:
    if not settings.MODEL_DIR:
        return
    try:
        predictor.load(settings.MODEL_DIR)
    except Exception:
        logger.exception("Failed to load model from MODEL_DIR")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=predictor.loaded,
        model_dir=predictor.model_dir,
        model_version=predictor.model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not predictor.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = predictor.predict(
        [request.text],
        top_k=request.top_k,
        min_confidence=request.min_confidence,
    )[0]
    return PredictResponse(**result)


@app.post(
    "/predict_batch",
    response_model=PredictBatchResponse,
    response_model_exclude_none=True,
)
def predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    if not predictor.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    texts = [item.text for item in request.items]
    results = predictor.predict(
        texts,
        top_k=request.top_k,
        min_confidence=request.min_confidence,
    )
    items = [
        {
            "id": item.id,
            "label": result["label"],
            "confidence": result["confidence"],
            "needs_human": result["needs_human"],
        }
        for item, result in zip(request.items, results)
    ]
    return PredictBatchResponse(items=items, model_version=predictor.model_version)
