# AGENTS — Project Workflow (support-ticket-intent-router)

This document defines how we work on this repo so the loop is repeatable and “agent-runnable”.
If project deliverables or workflow changes, we update this file + docs/PRD.md.

## Roles (locked)
- **ChatGPT (PM + Scope Guardian)** — this chat
  - Defines scope, phases, Task Cards, acceptance criteria, and prevents scope creep.
- **Ussama (Executioner)** — runs commands, verifies outputs, commits/pushes.
- **Codex in Cursor (Builder/Tester/Reviewer)** — implements changes under constraints.

## Cursor Mode Policy
- **Chat mode (default)**: Q&A, codebase understanding, small edits, debugging with human in loop.
- **Agent mode**: scoped feature work under a Task Card; touches multiple files; proposes minimal diffs.
- **Agent w/ Full Access**: rare. Only for explicitly approved repo-ops tasks with rollback plan.

## Canonical Commands (single source of truth)
Environment (uv-first):
- `uv venv`
- `source .venv/bin/activate`
- `uv sync --dev`

Core (Phase 1):
- `make train`
- `make eval`
- `MODEL_DIR=artifacts/model_0.1.0 make serve`

Quality:
- `make lint`
- `make test`
- `make ci` (preferred: lint + test)  ← if not present, use `make lint && make test`

Runtime checks:
- `curl http://localhost:8000/health`
- `curl -X POST http://localhost:8000/predict ...`
- `curl -X POST http://localhost:8000/predict_batch ...`

## Definition of Done (DoD)
A Task is “Done” only when:
- Scope matches the Task Card + PRD phase gates.
- Minimal diff (no “while I’m here” changes).
- **Green proof**:
  - `make ci` succeeds (or `make lint && make test`).
- If API behavior is involved:
  - `/health`, `/predict`, `/predict_batch` verified with curl.
- Artifacts/reports expectations met (when relevant):
  - `artifacts/model_0.1.0/` contains model/vectorizer/label_map/metadata
  - `reports/` contains metrics + confusion matrix outputs
- Docs updated if contract/commands changed.

## PM → Agent Execution Loop (repeatable)
### Step 0 — PM Task Card (required)
PM posts a Task Card with:
- Goal (user-visible)
- Scope + explicit non-goals
- Files likely touched (3–8 max)
- Commands to run (include `make ci`)
- Acceptance checks (3 bullets)
- Rollback strategy (1 commit per logical change)

### Step 1 — Builder (Codex, Agent mode)
Builder:
- Implements minimal diff
- Lists files touched
- Lists commands it expects to pass (even if it can’t run them)
- Calls out assumptions and edge cases

### Step 2 — Executioner verification (Ussama)
Run commands:
- `uv sync --dev` (if deps changed)
- `make ci` (or `make lint && make test`)
- relevant runtime checks (curl)
Paste outputs back to PM.

### Step 3 — Reviewer (Codex, Chat or Agent)
Two passes minimum:
1) Correctness/architecture/security
2) Tests/edge cases/contracts
Reviewer may propose small patch-only changes.

### Step 4 — Scope Guardian (PM, chat-only)
Confirm:
- Still within PRD phase
- No new deps without approval
- No creep in API contract

## Guardrails (non-negotiable)
- Small diffs only (1 feature or 1 refactor per change).
- Two-pass review minimum.
- No new dependencies without explicit approval in the Task Card.
- Green proof = `make ci` (or equivalent).
- Full Access requires explicit approval + rollback plan.

## Repo Map (update when structure changes)
- `app/` — FastAPI service
- `training/` — training + eval pipeline
- `docs/` — PRD/spec/workflow
- `tests/` — unit/integration tests
- `artifacts/` — generated model artifacts (not committed)
- `reports/` — generated eval outputs (not committed)

## Workflow Changelog
- 2026-01-02: Adopted “Variant 2-lite” (Task Cards + canonical commands + 2-pass review). uv-first default.
