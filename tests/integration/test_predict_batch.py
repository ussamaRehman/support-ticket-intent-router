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
