# Multi-stage Dockerfile for Sustainable Credit Risk AI System
# Stage 1: Base Python environment with dependencies
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development environment
FROM base AS development

# Copy source code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port for development server
EXPOSE 8000

# Default command for development
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: Production environment
FROM base AS production

# Copy only necessary files
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser model_registry/ ./model_registry/
COPY --chown=appuser:appuser Bank_data.csv ./Bank_data.csv

# Create necessary directories
RUN mkdir -p logs compliance_reports && \
    chown -R appuser:appuser logs compliance_reports

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Stage 4: Training environment
FROM base AS training

# Install additional ML dependencies for training
RUN pip install --no-cache-dir \
    tensorboard \
    wandb \
    optuna \
    ray[tune]

# Copy training-specific files
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser Bank_data.csv ./Bank_data.csv
COPY --chown=appuser:appuser app/scripts/ ./scripts/

# Create directories for training outputs
RUN mkdir -p models experiments logs && \
    chown -R appuser:appuser models experiments logs

# Switch to non-root user
USER appuser

# Default command for training
CMD ["python", "scripts/train_model.py"]

# Stage 5: Inference-only lightweight image
FROM python:3.10-slim AS inference

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install minimal Python dependencies for inference
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    torch \
    numpy \
    pandas \
    scikit-learn \
    pydantic

WORKDIR /app

# Copy only inference-related code
COPY --chown=appuser:appuser app/api/ ./app/api/
COPY --chown=appuser:appuser app/models/ ./app/models/
COPY --chown=appuser:appuser app/core/ ./app/core/
COPY --chown=appuser:appuser model_registry/ ./model_registry/

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Inference command
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]