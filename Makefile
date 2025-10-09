# Makefile for Sustainable Credit Risk AI System

.PHONY: help install test lint format build deploy clean

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run all tests"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  build       - Build Docker images"
	@echo "  deploy      - Deploy to Kubernetes"
	@echo "  clean       - Clean up artifacts"

# Development setup
install:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

# Testing
test:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-models:
	pytest tests/models/ -v

test-compliance:
	pytest tests/compliance/ -v

# Code quality
lint:
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/
	bandit -r src/

format:
	black src/ tests/
	isort src/ tests/

# Security
security-scan:
	safety check
	bandit -r src/
	semgrep --config=auto src/

# Docker operations
build:
	docker build -t sustainable-credit-risk-ai:latest .

build-all:
	docker build --target production -t sustainable-credit-risk-ai:production .
	docker build --target training -t sustainable-credit-risk-ai:training .
	docker build --target development -t sustainable-credit-risk-ai:development .

# Kubernetes operations
deploy:
	./k8s/deploy.sh deploy

deploy-staging:
	./k8s/deploy.sh deploy --environment staging

validate-deployment:
	./k8s/validate-deployment.sh

# Local development
run-local:
	docker-compose up -d

stop-local:
	docker-compose down

# Performance testing
performance-test:
	locust -f tests/load/locustfile.py --host=http://localhost:8000

# Cleanup
clean:
	docker system prune -f
	rm -rf htmlcov/ .coverage .pytest_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete