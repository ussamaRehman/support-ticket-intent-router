import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

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
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(message)s")

predictor = Predictor()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.MODEL_DIR:
        try:
            predictor.load(settings.MODEL_DIR)
        except Exception:
            logger.exception("Failed to load model from MODEL_DIR")
    yield


app = FastAPI(title="Ticket Router", lifespan=lifespan)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        _log_event(
            "http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            latency_ms=round(latency_ms, 2),
            model_version=predictor.model_version,
            model_dir=predictor.model_dir,
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=predictor.loaded,
        model_dir=predictor.model_dir,
        model_version=predictor.model_version,
    )


@app.get("/ready")
def ready() -> JSONResponse:
    model_loaded = predictor.loaded
    ready_state = True
    status_code = 200
    if settings.MODEL_DIR and not model_loaded:
        ready_state = False
        status_code = 503
    payload = {
        "status": "ok",
        "ready": ready_state,
        "model_loaded": model_loaded,
        "model_dir": predictor.model_dir,
        "model_version": predictor.model_version,
    }
    return JSONResponse(content=payload, status_code=status_code)


@app.post("/predict", response_model=PredictResponse)
def predict(http_request: Request, request: PredictRequest) -> PredictResponse:
    if not predictor.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = predictor.predict(
        [request.text],
        top_k=request.top_k,
        min_confidence=request.min_confidence,
    )[0]
    _log_prediction(
        request_id=http_request.state.request_id,
        min_confidence=request.min_confidence,
        top_k=request.top_k,
        label=result["label"],
        confidence=result["confidence"],
        needs_human=result["needs_human"],
    )
    return PredictResponse(**result)


@app.post(
    "/predict_batch",
    response_model=PredictBatchResponse,
    response_model_exclude_none=True,
)
def predict_batch(http_request: Request, request: PredictBatchRequest) -> PredictBatchResponse:
    if not predictor.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    texts = [item.text for item in request.items]
    results = predictor.predict(
        texts,
        top_k=request.top_k,
        min_confidence=request.min_confidence,
    )
    needs_human_count = sum(1 for result in results if result["needs_human"])
    _log_prediction_batch(
        request_id=http_request.state.request_id,
        min_confidence=request.min_confidence,
        top_k=request.top_k,
        item_count=len(results),
        needs_human_count=needs_human_count,
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


def _log_prediction(
    request_id: str,
    min_confidence: float,
    top_k: int,
    label: str,
    confidence: float,
    needs_human: bool,
) -> None:
    _log_event(
        "prediction",
        request_id=request_id,
        model_version=predictor.model_version,
        model_dir=predictor.model_dir,
        min_confidence=min_confidence,
        top_k=top_k,
        label=label,
        confidence=confidence,
        needs_human=needs_human,
    )


def _log_prediction_batch(
    request_id: str,
    min_confidence: float,
    top_k: int,
    item_count: int,
    needs_human_count: int,
) -> None:
    _log_event(
        "prediction_batch",
        request_id=request_id,
        model_version=predictor.model_version,
        model_dir=predictor.model_dir,
        min_confidence=min_confidence,
        top_k=top_k,
        item_count=item_count,
        needs_human_count=needs_human_count,
    )


def _log_event(event: str, **fields: object) -> None:
    payload = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(fields)
    logger.info(json.dumps(payload, ensure_ascii=False))
