.PHONY: help test format lint run-main-api run-inference-api run-backend

help:
	@echo "Available targets:"
	@echo "  test              Run backend smoke tests"
	@echo "  format            Format Python files"
	@echo "  lint              Run lightweight lint checks"
	@echo "  run-main-api      Start the main FastAPI service"
	@echo "  run-inference-api Start the inference FastAPI service"
	@echo "  run-backend       Start both backend services"

test:
	./venv/bin/python -m pytest -q tests/

format:
	./venv/bin/black app tests main.py
	./venv/bin/isort app tests main.py

lint:
	./venv/bin/black --check app tests main.py
	./venv/bin/isort --check-only app tests main.py
	./venv/bin/flake8 app tests main.py

run-main-api:
	./venv/bin/uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

run-inference-api:
	./venv/bin/python -c "from app.api.inference_service import run_inference_service; run_inference_service(port=8001)"

run-backend:
	./start_backend.sh
