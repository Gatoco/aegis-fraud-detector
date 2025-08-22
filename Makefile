# Aegis Fraud Detection System - Makefile
# Professional development and deployment automation

.PHONY: help install-deps install-dev format lint test clean docker-build docker-run mlflow-server data-download

# Default target
help:
	@echo "Aegis Fraud Detection System - Available Commands:"
	@echo ""
	@echo "Development Commands:"
	@echo "  install-deps      Install production dependencies"
	@echo "  install-dev       Install development dependencies"
	@echo "  format           Format code with black"
	@echo "  lint             Run linting with flake8"
	@echo "  test             Run test suite with pytest"
	@echo "  clean            Clean cache and temporary files"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build     Build development Docker image"
	@echo "  docker-run       Run development container"
	@echo "  docker-compose   Start MLflow and development services"
	@echo ""
	@echo "Data Commands:"
	@echo "  data-download    Download IEEE-CIS dataset from Kaggle"
	@echo "  data-sync        Sync data with DVC"
	@echo ""
	@echo "MLflow Commands:"
	@echo "  mlflow-server    Start MLflow tracking server"
	@echo "  mlflow-ui        Open MLflow UI in browser"
	@echo ""

# Python environment detection
PYTHON := python
PIP := pip
ifeq ($(OS),Windows_NT)
	PYTHON := .venv/Scripts/python.exe
	PIP := .venv/Scripts/pip.exe
else
	PYTHON := .venv/bin/python
	PIP := .venv/bin/pip
endif

# Development Dependencies
install-deps:
	@echo "Installing production dependencies..."
	$(PIP) install -r requirements.txt

install-dev: install-deps
	@echo "Installing development dependencies..."
	$(PIP) install black flake8 isort pytest pytest-cov pre-commit
	$(PIP) install jupyter jupyterlab
	$(PIP) install dvc mlflow
	@echo "Setting up pre-commit hooks..."
	.venv/Scripts/pre-commit install || .venv/bin/pre-commit install

# Code Quality
format:
	@echo "Formatting code with black..."
	black src/ api_service/ scripts/ tests/ --line-length 88
	@echo "Sorting imports with isort..."
	isort src/ api_service/ scripts/ tests/
	@echo "Code formatting complete."

lint:
	@echo "Running linting with flake8..."
	flake8 src/ api_service/ scripts/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "Linting complete."

# CI-style checks (matches GitHub Actions)
ci-lint:
	@echo "Running CI-style linting checks..."
	@echo "Checking code formatting..."
	black --check --diff src/ api_service/ scripts/ tests/ --line-length 88
	@echo "Checking import sorting..."
	isort --check-only --diff src/ api_service/ scripts/ tests/
	@echo "Running flake8 linting..."
	flake8 src/ api_service/ scripts/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ api_service/ scripts/ tests/ --count --max-complexity=10 --max-line-length=88 --statistics --extend-ignore=E203,W503

test:
	@echo "Running test suite..."
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "Test suite complete."

# Individual test categories
test-unit:
	@echo "Running unit tests..."
	pytest tests/ -v --tb=short

test-api:
	@echo "Running API tests..."
	cd api_service && pytest test_api.py -v --tb=short

test-integration:
	@echo "Running integration tests..."
	cd api_service && pytest test_integration.py -v --tb=short

# CI-style testing (matches GitHub Actions)
ci-test: ci-lint test-unit test-api test-integration
	@echo "CI/CD pipeline tests complete."

# Cleanup
clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage .mypy_cache/ dist/ build/
	@echo "Cleanup complete."

# Docker Commands
docker-build:
	@echo "Building development Docker image..."
	docker build -t aegis-fraud-detector:dev .
	@echo "Docker build complete."

docker-run:
	@echo "Starting development container..."
	docker run -it --rm \
		-v $(PWD):/workspace \
		-p 8888:8888 \
		-p 5000:5000 \
		aegis-fraud-detector:dev

docker-compose:
	@echo "Starting MLflow and development services..."
	docker-compose up -d
	@echo "Services started. MLflow UI: http://localhost:5000"

# Data Management
data-download:
	@echo "Downloading IEEE-CIS dataset from Kaggle..."
	@echo "Make sure you have Kaggle API configured: ~/.kaggle/kaggle.json"
	kaggle competitions download -c ieee-fraud-detection -p data/01_raw/
	cd data/01_raw && unzip -o ieee-fraud-detection.zip && rm ieee-fraud-detection.zip
	@echo "Dataset download complete."

data-sync:
	@echo "Syncing data with DVC..."
	dvc pull
	@echo "Data sync complete."

# MLflow Commands
mlflow-server:
	@echo "Starting MLflow tracking server..."
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./artifacts \
		--host 0.0.0.0 \
		--port 5000

mlflow-ui:
	@echo "Opening MLflow UI..."
	@echo "MLflow UI available at: http://localhost:5000"
ifeq ($(OS),Windows_NT)
	start http://localhost:5000
else
	open http://localhost:5000 || xdg-open http://localhost:5000
endif

# Model Training Pipeline
train-baseline:
	@echo "Training baseline model..."
	$(PYTHON) src/models/train_baseline.py

train-advanced:
	@echo "Training advanced ensemble model..."
	$(PYTHON) src/models/train_ensemble.py

# Evaluation
evaluate:
	@echo "Evaluating model performance..."
	$(PYTHON) src/models/evaluate.py

# Feature Engineering
features:
	@echo "Running feature engineering pipeline..."
	$(PYTHON) src/features/build_features.py

# Notebooks
notebook:
	@echo "Starting Jupyter Lab..."
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Zero-Cost CI Commands
simple-ci:
	@echo "Running simple zero-cost CI..."
	$(PYTHON) scripts/simple_ci.py

zero-cost-ci:
	@echo "Running zero-cost CI pipeline..."
	$(PYTHON) scripts/zero_cost_ci.py

fast-ci:
	@echo "Running ultra-fast CI..."
	$(PYTHON) scripts/zero_cost_ci.py --fast

api-ci:
	@echo "Running API-only CI..."
	$(PYTHON) scripts/zero_cost_ci.py --api-only

# Original CI commands (for comparison)
ci-test: ci-lint test-unit test-api test-integration
	@echo "CI/CD pipeline tests complete."

# Production
deploy:
	@echo "Deploying model to production..."
	@echo "Production deployment not implemented yet."

# Version Management
version:
	@echo "Current project version information:"
	@echo "Git commit: $$(git rev-parse --short HEAD)"
	@echo "Git branch: $$(git branch --show-current)"
	@echo "DVC version: $$(dvc version)"
	@echo "Python version: $$($(PYTHON) --version)"
