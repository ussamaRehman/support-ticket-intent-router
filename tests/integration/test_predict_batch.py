import importlib

from fastapi.testclient import TestClient


def test_predict_batch_requires_model(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_DIR", raising=False)
    from app import main as main_module

    importlib.reload(main_module)
    client = TestClient(main_module.app)

    payload = {"items": [{"id": "1", "text": "Reset my password"}], "top_k": 3}
    response = client.post("/predict_batch", json=payload)

    assert response.status_code == 503


def test_predict_batch_min_confidence_bounds(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_DIR", raising=False)
    from app import main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as client:
        base_payload = {"items": [{"id": "1", "text": "Reset my password"}], "top_k": 3}

        low_payload = {**base_payload, "min_confidence": -0.1}
        low_resp = client.post("/predict_batch", json=low_payload)
        assert low_resp.status_code == 422

        high_payload = {**base_payload, "min_confidence": 1.1}
        high_resp = client.post("/predict_batch", json=high_payload)
        assert high_resp.status_code == 422
