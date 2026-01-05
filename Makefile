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
	PORT=""; \
	if command -v lsof >/dev/null 2>&1; then \
		for candidate in 8000 8001 8002 8003 8004; do \
			if ! lsof -i:$$candidate -sTCP:LISTEN >/dev/null 2>&1; then \
				PORT=$$candidate; \
				break; \
			fi; \
		done; \
	else \
		PORT=8000; \
	fi; \
	if [ -z "$$PORT" ]; then \
		echo "No free port found in 8000-8004. Stop the service using those ports and retry."; \
		exit 1; \
	fi; \
	echo "Starting container on port $$PORT"; \
	CID="$$(docker run -d -p $$PORT:8000 ticket-router:dev)"; \
	echo "Container ID: $$CID"; \
	trap 'docker rm -f $$CID >/dev/null 2>&1' EXIT; \
	attempt=1; \
	last_health=""; \
	for wait in 1 1 2 2 3 3 4 4; do \
		echo "Attempt $$attempt (sleep $$wait)s"; \
		curl_err="$$(mktemp)"; \
		set +e; \
		health="$$(curl -sS --connect-timeout 2 --max-time 2 http://localhost:$$PORT/health 2>$$curl_err)"; \
		code="$$?"; \
		set -e; \
		last_health="$$health"; \
		if [ -n "$$health" ]; then \
			echo "$$health"; \
			rm -f "$$curl_err"; \
			exit 0; \
		fi; \
		if [ -s "$$curl_err" ]; then \
			echo "curl exit code $$code: $$(cat $$curl_err | tr '\n' ' ')"; \
		else \
			echo "curl exit code $$code (no response yet)"; \
		fi; \
		rm -f "$$curl_err"; \
		if ! docker ps -q --no-trunc | grep -q "$$CID"; then \
			echo "Smoke test failed; container exited early:"; \
			docker logs $$CID; \
			exit 1; \
		fi; \
		sleep $$wait; \
		attempt=$$((attempt+1)); \
	done; \
	if [ -n "$$last_health" ]; then \
		echo "Last health response: $$last_health"; \
	fi; \
	echo "Smoke test failed; container logs:"; \
	docker logs $$CID; \
	exit 1
