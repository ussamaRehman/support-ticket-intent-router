import importlib
import time

import pytest
from fastapi.testclient import TestClient


def test_payload_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_BODY_BYTES", "50")
    monkeypatch.delenv("MODEL_DIR", raising=False)
    from app import main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as client:
        payload = {"text": "x" * 200, "top_k": 3, "min_confidence": 0.55}
        response = client.post("/predict", json=payload)
        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()


def test_prediction_timeout(monkeypatch: pytest.MonkeyPatch, model_dir) -> None:
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("PREDICT_TIMEOUT_MS", "1")
    from app import main as main_module

    importlib.reload(main_module)

    def slow_predict(texts, top_k=3, min_confidence=0.55):
        time.sleep(0.05)
        return [
            {
                "label": "account",
                "confidence": 0.9,
                "alternatives": [
                    {"label": "account", "confidence": 0.9},
                    {"label": "billing", "confidence": 0.1},
                ],
                "needs_human": False,
            }
        ]

    monkeypatch.setattr(main_module.predictor, "predict", slow_predict)
    with TestClient(main_module.app) as client:
        payload = {"text": "Reset my password", "top_k": 3, "min_confidence": 0.55}
        response = client.post("/predict", json=payload)
        assert response.status_code == 503
        assert "timed out" in response.json()["detail"].lower()
