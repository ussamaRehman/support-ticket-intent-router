import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _client_with_model(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    model_dir = Path("artifacts") / "model_0.1.0"
    if not (model_dir / "model.pkl").exists():
        pytest.skip("Model artifacts not found; run make train first.")
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    from app import main as main_module

    importlib.reload(main_module)
    return TestClient(main_module.app)


def test_predict_guardrail(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client_with_model(monkeypatch) as client:
        high_payload = {"text": "I need help with my invoice", "min_confidence": 0.99, "top_k": 3}
        high_resp = client.post("/predict", json=high_payload)
        assert high_resp.status_code == 200
        high_body = high_resp.json()
        assert high_body["label"] == "human_review"
        assert high_body["needs_human"] is True

        low_payload = {"text": "I need help with my invoice", "min_confidence": 0.0, "top_k": 3}
        low_resp = client.post("/predict", json=low_payload)
        assert low_resp.status_code == 200
        low_body = low_resp.json()
        assert low_body["needs_human"] is False
        assert low_body["label"] != "human_review"


def test_predict_min_confidence_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_DIR", raising=False)
    from app import main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as client:
        base_payload = {"text": "I need help with my invoice", "top_k": 3}

        low_payload = {**base_payload, "min_confidence": -0.1}
        low_resp = client.post("/predict", json=low_payload)
        assert low_resp.status_code == 422

        high_payload = {**base_payload, "min_confidence": 1.1}
        high_resp = client.post("/predict", json=high_payload)
        assert high_resp.status_code == 422


def test_predict_batch_guardrail(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client_with_model(monkeypatch) as client:
        payload = {
            "items": [
                {"id": "1", "text": "I need help with my invoice"},
                {"id": "2", "text": "Refund this charge"},
            ],
            "min_confidence": 0.99,
            "top_k": 3,
        }
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "items" in body
        for item in body["items"]:
            assert item["label"] == "human_review"
            assert item["needs_human"] is True
            assert isinstance(item["confidence"], float)
