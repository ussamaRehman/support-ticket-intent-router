# Ticket Router (Intent Classification) API

Phase 1 delivers a baseline intent classifier with FastAPI endpoints, training, and evaluation artifacts.
Training uses the Banking77 dataset (PolyAI task-specific-datasets, CC BY 4.0). The first `make train` run will download and cache it in `data/banking77/`.

## Quickstart
```bash
uv venv
source .venv/bin/activate
uv sync --dev
make train
make eval
MODEL_DIR=artifacts/model_0.1.0 make serve
```

## Pip fallback (compatibility only)
```bash
pip install -e ".[dev]"
```

## Endpoints
### Health
```bash
curl http://localhost:8000/health
```

### Predict (single)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I need help with my invoice", "top_k": 3}'
```

### Predict (batch)
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"id": "1", "text": "Reset my password"}, {"id": "2", "text": "Refund this charge"}], "top_k": 3}'
```

## Make targets
- `make install`: install runtime dependencies.
- `make install-dev`: install dev dependencies.
- `make train`: train baseline model and save artifacts.
- `make eval`: run evaluation and write reports.
- `make serve`: run FastAPI service.
- `make test`: run tests.
- `make lint`: run ruff checks.
- `make format`: format with ruff.
