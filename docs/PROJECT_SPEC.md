# Project Spec: Ticket Router (Intent Classification) API

## Overview
This project delivers a baseline intent classifier for support tickets with a minimal FastAPI service. Phase 1 ships a TF-IDF + Logistic Regression model with evaluation reports and model artifacts.

## API surface
- `GET /health` returns service status and `model_loaded`.
- `POST /predict` accepts a single text string and optional `top_k`.
- `POST /predict_batch` accepts a list of text strings and optional `top_k`.

### Response shape (predict)
- `label`: predicted class label.
- `confidence`: probability for the predicted label.
- `alternatives`: top-k list of `{label, confidence}` sorted by confidence.

If no model is loaded, `/predict` and `/predict_batch` return `503`.

## Modeling and artifacts
- Baseline model: TF-IDF vectorizer + Logistic Regression.
- Artifacts saved to `artifacts/model_0.1.0/`:
  - `model.pkl`
  - `vectorizer.pkl`
  - `label_map.json` (id to label)
  - `metadata.json` (dataset, seed, versions, created_at, metrics summary)

## Evaluation outputs
- `reports/metrics.json`: macro F1, per-class precision/recall/F1, top-k accuracy.
- `reports/confusion_matrix.json`: label list and matrix.

## Repo structure
- `app/`: FastAPI app and prediction service.
- `training/`: baseline training/evaluation scripts and sample data loader.
- `docs/`: PRD and project spec.
- `tests/`: unit/integration tests (Phase 1 scaffold only).

## TODO (future phases)
- Phase 2: confidence guardrail with `needs_human` output.
- Phase 3: Docker and CI.
- Phase 4: structured logs and audit trail.
- Phase 5: monitoring hooks and registry-lite.
