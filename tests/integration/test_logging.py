import importlib
import json

import pytest
from fastapi.testclient import TestClient


def test_prediction_audit_log(monkeypatch: pytest.MonkeyPatch, caplog, model_dir) -> None:
    caplog.set_level("INFO")
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    from app import main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as client:
        request_id = "test-request-id"
        payload = {"text": "I need help with my invoice", "min_confidence": 0.0, "top_k": 3}
        response = client.post("/predict", json=payload, headers={"X-Request-ID": request_id})
        assert response.status_code == 200

    events = []
    messages = []
    for record in caplog.records:
        message = record.getMessage()
        messages.append(message)
        try:
            events.append(json.loads(message))
        except json.JSONDecodeError:
            continue
    if not events:
        pytest.fail(f"No JSON log entries found. Raw logs: {messages}")
    prediction = next((event for event in events if event.get("event") == "prediction"), None)
    assert prediction is not None
    assert prediction.get("request_id") == request_id
    for key in (
        "timestamp",
        "model_version",
        "model_dir",
        "min_confidence",
        "top_k",
        "label",
        "confidence",
        "needs_human",
    ):
        assert key in prediction

    http_request = next((event for event in events if event.get("event") == "http_request"), None)
    assert http_request is not None
    for key in (
        "request_id",
        "method",
        "path",
        "status_code",
        "latency_ms",
        "timestamp",
        "model_version",
        "model_dir",
    ):
        assert key in http_request
