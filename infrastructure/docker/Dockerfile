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

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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

# Copy data file if it exists (optional for production builds)
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
    tensorboard==2.15.1 \
    wandb==0.16.1 \
    optuna==3.5.0 \
    ray[tune]==2.8.1

# Copy training-specific files
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser Bank_data.csv ./Bank_data.csv
COPY --chown=appuser:appuser scripts/ ./scripts/

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
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install minimal system dependencies and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && apt-get autoremove -y

# Install minimal Python dependencies for inference
RUN pip install --no-cache-dir --no-deps \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    torch==2.1.1 \
    numpy==1.24.4 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    pydantic==2.5.2 \
    && pip cache purge \
    && find /usr/local -name "*.pyc" -delete \
    && (find /usr/local -name "__pycache__" -type d -exec rm -rf {} + || true)

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