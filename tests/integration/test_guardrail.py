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
