# Aegis Fraud Detection System - Development Environment
# Multi-stage Docker build for efficient development and production deployment

# Base Python image with ML libraries pre-installed
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Development stage
FROM base as development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install jupyter jupyterlab mlflow dvc black flake8 pytest

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/01_raw data/02_processed models artifacts logs

# Set proper permissions
RUN chmod +x scripts/* || true

# Expose ports for Jupyter and MLflow
EXPOSE 8888 5000

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# Production stage
FROM base as production

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash aegis
USER aegis
WORKDIR /home/aegis

# Create virtual environment
RUN python -m venv /home/aegis/venv
ENV PATH="/home/aegis/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY --chown=aegis:aegis requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application code
COPY --chown=aegis:aegis src/ src/
COPY --chown=aegis:aegis models/ models/

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import src.models.predict; print('OK')" || exit 1

# Default command for production
CMD ["python", "-m", "src.api.main"]

# MLflow tracking server stage
FROM base as mlflow

# Install MLflow
RUN pip install mlflow psycopg2-binary

# Create MLflow user
RUN useradd --create-home --shell /bin/bash mlflow
USER mlflow
WORKDIR /home/mlflow

# Create directories for MLflow
RUN mkdir -p mlruns artifacts

# Expose MLflow port
EXPOSE 5000

# MLflow server command
CMD ["mlflow", "server", \
    "--backend-store-uri", "sqlite:///mlflow.db", \
    "--default-artifact-root", "/home/mlflow/artifacts", \
    "--host", "0.0.0.0", \
    "--port", "5000"]
