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

## Quick API Demo
Local:
```bash
make ci
make serve
curl http://localhost:8000/ready
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I need help with my invoice", "top_k": 3, "min_confidence": 0.55}'
```

Docker:
```bash
make docker-build
make docker-smoke
curl http://localhost:8000/ready
```

Docker with model:
```bash
make docker-smoke-model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I need help with my invoice", "top_k": 3, "min_confidence": 0.55}'
```

Troubleshooting:
- Docker Desktop must be running.
- Cold-start curl failures can happen on attempt 1; the smoke targets retry automatically.
- Ports 8000–8004 are used; free them with `lsof -i:8000 -sTCP:LISTEN`.

## Pip fallback (compatibility only)
```bash
pip install -e ".[dev]"
```

## Endpoints
### Readiness vs health
- `/ready` indicates the service is ready to accept traffic.
- `/health` reports service status and whether a model is loaded.
- If `MODEL_DIR` is set, `/ready` returns `503` until the model is loaded.
- `X-Request-ID` (optional) is echoed in logs for traceability; if omitted, one is generated.
- `/health` is stable for probes; `/ready` is stricter for traffic routing.

### Ready
```bash
curl http://localhost:8000/ready
```

### Health
```bash
curl http://localhost:8000/health
```

### Predict (single)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I need help with my invoice", "top_k": 3, "min_confidence": 0.55}'
```

### Predict (batch)
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"id": "1", "text": "Reset my password"}, {"id": "2", "text": "Refund this charge"}], "top_k": 3, "min_confidence": 0.55}'
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
- `make ci`: run lint and tests.

## Quality
```bash
make ci
```

## Docker
Build and run the API container:
```bash
docker build -t ticket-router:dev .
docker run -p 8000:8000 ticket-router:dev
curl http://localhost:8000/health
```

To mount local artifacts and set `MODEL_DIR`:
```bash
docker run -p 8000:8000 \
  -e MODEL_DIR=/artifacts/model_0.1.0 \
  -v "$(pwd)/artifacts:/artifacts" \
  ticket-router:dev
```

Health check expectations:
```bash
docker run -d --name ticket-router -p 8000:8000 ticket-router:dev
curl http://localhost:8000/health  # model_loaded=false
docker rm -f ticket-router

docker run -d --name ticket-router -p 8000:8000 \
  -e MODEL_DIR=/artifacts/model_0.1.0 \
  -v "$(pwd)/artifacts:/artifacts" \
  ticket-router:dev
curl http://localhost:8000/health  # model_loaded=true
docker rm -f ticket-router
```

## Docker smoke
```bash
make docker-build
make docker-smoke
```
Notes:
- Docker Desktop must be running.
- `make docker-smoke` will try ports 8000–8004; free them with `lsof -i:8000 -sTCP:LISTEN` (or pick another port if needed).
- It is normal to see an initial curl failure (for example, exit code 56) on cold start; the script retries automatically.

## Run Docker with a model
```bash
make train
make docker-build
make docker-run-model
make docker-smoke-model
```
Notes:
- Mounts local `./artifacts` read-only and uses ports 8000–8004.

## cURL cookbook
Ready:
```bash
curl -H "X-Request-ID: demo-req-1" http://localhost:8000/ready
```

Health:
```bash
curl -H "X-Request-ID: demo-req-2" http://localhost:8000/health
```

Predict:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-req-3" \
  -d '{"text": "Reset my password", "top_k": 3, "min_confidence": 0.55}'
```

Predict batch:
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-req-4" \
  -d '{"items": [{"id": "1", "text": "Reset my password"}, {"id": "2", "text": "Refund this charge"}], "top_k": 3, "min_confidence": 0.55}'
```

## Logging
Logs are emitted as JSON lines to stdout. Example:
```json
{"event":"prediction","timestamp":"2025-01-01T12:00:00+00:00","request_id":"req-123","model_version":"0.1.0","model_dir":"artifacts/model_0.1.0","min_confidence":0.55,"top_k":3,"label":"billing","confidence":0.91,"needs_human":false}
```
