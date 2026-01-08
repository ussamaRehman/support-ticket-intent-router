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
    for record in caplog.records:
        try:
            events.append(json.loads(record.getMessage()))
        except json.JSONDecodeError:
            continue
    assert any(event.get("event") == "prediction" for event in events)
    assert any(event.get("request_id") == request_id for event in events)
