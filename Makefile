.PHONY: install install-dev install-pip install-dev-pip train eval serve test lint format ci docker-build docker-run docker-smoke

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
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

ci:
	$(MAKE) lint
	$(MAKE) test

docker-build:
	docker build -t ticket-router:dev .

docker-run:
	docker run -p 8000:8000 ticket-router:dev

docker-smoke:
	@set -e; \
	PORT=8000; \
	if command -v lsof >/dev/null 2>&1; then \
		if lsof -i:8000 -sTCP:LISTEN >/dev/null 2>&1; then \
			PORT=8001; \
		fi; \
		if lsof -i:8001 -sTCP:LISTEN >/dev/null 2>&1; then \
			PORT=8002; \
		fi; \
	fi; \
	echo "Starting container on port $$PORT"; \
	CID="$$(docker run -d -p $$PORT:8000 ticket-router:dev)"; \
	trap 'docker rm -f $$CID >/dev/null 2>&1' EXIT; \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
		echo "Attempt $$i"; \
		health="$$(curl -sS --max-time 2 http://localhost:$$PORT/health || true)"; \
		if [ -n "$$health" ]; then \
			echo "$$health"; \
			exit 0; \
		fi; \
		if ! docker ps -q --no-trunc | grep -q "$$CID"; then \
			echo "Smoke test failed; container exited early:"; \
			docker logs $$CID; \
			exit 1; \
		fi; \
		sleep 1; \
	done; \
	echo "Smoke test failed; container logs:"; \
	docker logs $$CID; \
	exit 1
