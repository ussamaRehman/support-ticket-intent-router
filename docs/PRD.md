# Ticket Router (Intent Classification) API PRD

## MVP definition (Phase 1 only)
- Build a multiclass intent classifier for English, single-text inference.
- Provide FastAPI endpoints: `POST /predict`, `POST /predict_batch`, `GET /health`, `GET /ready`.
- Use TF-IDF + Logistic Regression baseline model.
- Train/evaluate on a held-out test split and save required artifacts.
- Generate evaluation reports and confusion matrix artifacts.
- Document how to run training, evaluation, and serving locally.

## Phased plan (hard gates)
1. Phase 1 (MVP): baseline model + evaluation + artifacts + API + docs.
2. Phase 2: confidence guardrail (needs_human / human_review).
3. Phase 3: Docker + CI + smoke tests.
4. Phase 4: structured logs + audit trail.
5. Phase 5 (optional): monitoring hooks + registry-lite improvements.

## Tech Stack (Authoritative)

### Language & Runtime
- Python 3.11 (3.10 acceptable)

### API (Phase 1+)
- FastAPI
- Uvicorn
- Pydantic v2

### Configuration
- Env vars as the source of truth
- pydantic-settings (typed settings)

### ML (Phase 1 MVP)
- scikit-learn
- TF-IDF: TfidfVectorizer
- Classifier: LogisticRegression (default). LinearSVC is allowed but would require probability handling/calibration (documentation-only in Phase 1).
- Serialization: joblib (model.pkl, vectorizer.pkl)
- numpy (required)
- pandas (optional; only if needed)

### Testing & Quality
- pytest
- httpx (integration tests)
- ruff (lint + format)

### Dependency / Environment Management
- Default: uv (required for this repo)
- Fallback: pip (compatibility only)

### Packaging & Commands
- pyproject.toml
- Makefile targets: train, eval, serve, test, lint, format

### Explicitly NOT in Phase 1 (only later phases)
- Docker + CI (Phase 3)
- Structured logs + audit trail (Phase 4)
- Monitoring hooks / Prometheus metrics (Phase 5 optional)
- OpenTelemetry (optional later observability)

## Non-goals for Phase 1
- No Docker or container orchestration.
- No CI workflows.
- No structured logs, audit trail, or monitoring.
- No auth, database, queues, or external integrations.
- No transformer models or advanced serving.
- No UI or dashboard.

## Acceptance criteria
### Phase 1
- `make train` produces `artifacts/model_0.1.0/` with `model.pkl`, `vectorizer.pkl`, `label_map.json`, `metadata.json`.
- `make eval` writes `reports/metrics.json` and `reports/confusion_matrix.json`.
- `MODEL_DIR=artifacts/model_0.1.0 make serve` boots the API.
- `GET /health` returns `model_loaded=true` when artifacts are loaded.
- `GET /ready` returns `200` when ready; if `MODEL_DIR` is set and the model is not loaded, return `503`.
- `POST /predict` and `POST /predict_batch` return label, confidence, and top-k alternatives.
- Requests return `503` when the model is not loaded.

### Phase 2
- Add confidence/threshold guardrails with `min_confidence` inputs and `needs_human` outputs.
- When confidence is below threshold, return `label=human_review` and `needs_human=true`.

### Phase 3
- Add Docker and CI with smoke tests.

### Phase 4
- Add structured logs and audit trail.
- Logs must include request_id, timestamp, model_version, and decision fields.

### Phase 5 (optional)
- Add monitoring hooks and registry-lite improvements.

## Risk control
- Scope creep: adhere to Phase 1 only; add TODOs in docs for later phases.
- Data quality: sample dataset is placeholder; real dataset loader comes next.
- Reproducibility: fixed seed, tracked versions in metadata.
- Evaluation clarity: report macro F1, per-class metrics, top-k accuracy, and confusion matrix.
