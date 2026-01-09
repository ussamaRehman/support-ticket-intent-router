.PHONY: install install-dev install-pip install-dev-pip train eval serve serve-model demo test lint format format-check ci docker-build docker-run docker-smoke docker-run-model docker-smoke-model

PORT_CANDIDATES := 8000 8001 8002 8003 8004
READY_PATH := /ready
MODEL_DIR_IN_CONTAINER := /app/artifacts/model_0.1.0
ARTIFACTS_MOUNT := $(PWD)/artifacts:/app/artifacts:ro

install:
	uv sync

install-dev:
	uv sync --dev

install-pip:
	pip install -e .

install-dev-pip:
	pip install -e ".[dev]"

train:
	uv run python -m training.train_baseline

eval:
	uv run python -m training.eval_baseline

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

serve-model:
	@if [ -f artifacts/model_0.1.0/model.pkl ]; then \
		MODEL_DIR=artifacts/model_0.1.0 $(MAKE) serve; \
	else \
		echo "Model artifacts not found. Run 'make train' first."; \
		exit 1; \
	fi

demo:
	@echo "Checking /ready..."; \
	if curl -sS --connect-timeout 2 --max-time 2 http://localhost:8000/ready >/dev/null 2>&1; then \
		curl -sS http://localhost:8000/ready; \
	else \
		echo "Service not reachable on :8000. Run 'make serve' (or 'make serve-model')."; \
		exit 0; \
	fi; \
	echo ""; \
	echo "Checking /predict..."; \
	if curl -sS --connect-timeout 2 --max-time 2 http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"text":"Reset my password","top_k":3,"min_confidence":0.55}' >/dev/null 2>&1; then \
		curl -sS http://localhost:8000/predict \
			-H "Content-Type: application/json" \
			-d '{"text":"Reset my password","top_k":3,"min_confidence":0.55}'; \
	else \
		echo "Predict not available (model may be missing). Run 'make train' and 'make serve-model'."; \
	fi

test:
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

ci:
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test

docker-build:
	docker build -t ticket-router:dev .

docker-run:
	docker run -p 8000:8000 ticket-router:dev

define RUN_DOCKER_SMOKE
	@set -e; \
	PORT=""; \
	LSOF="$$(command -v lsof 2>/dev/null || echo /usr/sbin/lsof)"; \
	if [ -x "$$LSOF" ]; then \
		for candidate in $(PORT_CANDIDATES); do \
			if ! $$LSOF -i:$$candidate -sTCP:LISTEN >/dev/null 2>&1; then \
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
	CID="$$(docker run -d -p $$PORT:8000 $(1) ticket-router:dev)"; \
	echo "Container ID: $$CID"; \
	trap 'docker rm -f $$CID >/dev/null 2>&1' EXIT; \
	attempt=1; \
	last_health=""; \
	steps="1 1 2 2 3 3 4 4"; \
	total=$$(echo "$$steps" | wc -w | tr -d ' '); \
	for wait in $$steps; do \
		echo "Attempt $$attempt: not ready yet (sleep $$wait)s"; \
		curl_err="$$(mktemp)"; \
		set +e; \
		health="$$(curl --fail --silent --show-error --connect-timeout 2 --max-time 2 http://localhost:$$PORT$(READY_PATH) 2>$$curl_err)"; \
		code="$$?"; \
		set -e; \
		last_health="$$health"; \
		if [ -n "$$health" ] && $(2); then \
			echo "$$health"; \
			rm -f "$$curl_err"; \
			exit 0; \
		fi; \
		if ! docker ps -q --no-trunc | grep -q "$$CID"; then \
			echo "$(3)"; \
			if [ -s "$$curl_err" ]; then \
				echo "curl exit code $$code: $$(cat $$curl_err | tr '\n' ' ')"; \
			fi; \
			docker logs $$CID; \
			rm -f "$$curl_err"; \
			exit 1; \
		fi; \
		if [ $$attempt -eq $$total ]; then \
			if [ -s "$$curl_err" ]; then \
				echo "curl exit code $$code: $$(cat $$curl_err | tr '\n' ' ')"; \
			fi; \
		fi; \
		rm -f "$$curl_err"; \
		sleep $$wait; \
		attempt=$$((attempt+1)); \
	done; \
	if [ -n "$$last_health" ]; then \
		echo "Last health response: $$last_health"; \
	fi; \
	echo "$(4)"; \
	docker logs $$CID; \
	exit 1
endef

docker-smoke:
	$(call RUN_DOCKER_SMOKE,,echo "$$health" | grep -Eq '"ready":[[:space:]]*true',"Smoke test failed; container exited early:","Smoke test failed; container logs:")

docker-run-model:
	$(call RUN_DOCKER_SMOKE,-e MODEL_DIR=$(MODEL_DIR_IN_CONTAINER) -v $(ARTIFACTS_MOUNT),echo "$$health" | grep -Eq '"ready":[[:space:]]*true',"Docker run with model failed; container exited early:","Docker run with model failed; container logs:")

docker-smoke-model:
	$(call RUN_DOCKER_SMOKE,-e MODEL_DIR=$(MODEL_DIR_IN_CONTAINER) -v $(ARTIFACTS_MOUNT),echo "$$health" | grep -Eq '"ready":[[:space:]]*true' && echo "$$health" | grep -Eq '"model_loaded":[[:space:]]*true',"Docker smoke model failed; container exited early:","Docker smoke model failed; container logs:")
