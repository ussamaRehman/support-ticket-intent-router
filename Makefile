.PHONY: install install-dev install-pip install-dev-pip train eval serve test lint format

install:
	uv sync

install-dev:
	uv sync --dev

install-pip:
	pip install -e .

install-dev-pip:
	pip install -e ".[dev]"

train:
	python -m training.train_baseline

eval:
	python -m training.eval_baseline

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .
