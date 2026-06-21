"""
Main FastAPI application for the Sustainable Credit Risk AI System.

This is the platform / orchestration API (port 8000). In addition to health and
status probes it exposes *real* federated-learning and fairness-audit
computations. The heavy ML dependencies (torch, numpy, pandas) are imported
lazily inside the endpoints so health checks and app startup stay fast and the
service degrades gracefully if an optional dependency is missing.
"""

import os
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

# Add app to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import load_config
from app.core.logging import get_logger

# Initialize FastAPI app
app = FastAPI(
    title="Sustainable Credit Risk AI System",
    description="AI system for credit risk assessment with sustainability features",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def _allowed_origins() -> list[str]:
    """Resolve CORS origins from the environment (comma-separated).

    Defaults to ``*`` for local development; set ``PULSELEDGER_ALLOWED_ORIGINS``
    in production to lock this down.
    """
    raw = os.getenv("PULSELEDGER_ALLOWED_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger
logger = get_logger(__name__)

# Lightweight in-process counters (surfaced via /metrics).
_METRICS = {
    "requests_total": 0,
    "federated_runs_total": 0,
    "fairness_audits_total": 0,
    "errors_total": 0,
}


def _ok(payload):
    return {"status": "ok", "data": payload}


@app.middleware("http")
async def _count_requests(request: Request, call_next):
    _METRICS["requests_total"] += 1
    return await call_next(request)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    _METRICS["errors_total"] += 1
    logger.error(f"Unhandled API error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": "internal_server_error"},
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return _ok(
        {"message": "Sustainable Credit Risk AI System", "version": "1.1.0"}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return _ok({"service_status": "healthy", "service": "credit-risk-ai"})


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Add any readiness checks here (database, model loading, etc.)
        return _ok({"service_status": "ready", "service": "credit-risk-ai"})
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return _ok(
        {
            "service_status": "operational",
            "version": "1.1.0",
            "features": [
                "credit-risk-assessment",
                "sustainability-monitoring",
                "federated-learning",
                "fairness-audit",
                "explainable-ai",
            ],
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Federated learning (real FedAvg simulation)
# ─────────────────────────────────────────────────────────────────────────────


class FederatedRunRequest(BaseModel):
    """Parameters for a federated-learning simulation run."""

    number_of_clients: int = Field(
        3, ge=2, le=10, description="Number of participating clients"
    )
    aggregation_rounds: int = Field(
        3, ge=1, le=12, description="Number of FedAvg aggregation rounds"
    )
    local_epochs: int = Field(
        2, ge=1, le=5, description="Local training epochs per client per round"
    )


@app.post("/api/v1/federated/run")
async def federated_run(req: FederatedRunRequest):
    """Run a real multi-client FedAvg simulation and return round-by-round
    metrics (train/validation loss and accuracy).

    The torch-backed simulation is imported lazily and executed in a worker
    thread so it never blocks the event loop. Bounded parameters keep a run
    well under a few seconds.
    """
    try:
        from fastapi.concurrency import run_in_threadpool

        from app.federated.config import FLConfig
        from app.federated.utils import run_federated_simulation
    except ImportError as exc:  # pragma: no cover - optional dependency
        logger.error(f"Federated dependencies unavailable: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Federated learning dependencies are not installed.",
        )

    config = FLConfig(
        number_of_clients=req.number_of_clients,
        aggregation_rounds=req.aggregation_rounds,
        local_epochs=req.local_epochs,
    )

    started = time.time()
    try:
        result = await run_in_threadpool(run_federated_simulation, config)
    except Exception as exc:
        _METRICS["errors_total"] += 1
        logger.error(f"Federated simulation failed: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Federated simulation failed: {exc}"
        )

    _METRICS["federated_runs_total"] += 1
    result["wall_time_seconds"] = round(time.time() - started, 3)
    logger.info(
        "Federated run complete: clients=%s rounds=%s best_val_loss=%.4f",
        req.number_of_clients,
        req.aggregation_rounds,
        float(result.get("best_val_loss", 0.0)),
    )
    return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Fairness audit (real bias-detection metrics)
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/api/v1/fairness/audit")
async def fairness_audit(
    samples: int = 1000, bias_strength: float = 1.2, seed: int = 42
):
    """Run the real fairness/bias detector over a deterministic synthetic
    cohort and return demographic-parity, equalized-odds and related metrics.

    ``bias_strength`` controls how skewed the synthetic model is toward one
    group (1.0 = unbiased), which makes the audit interactive without needing a
    live training dataset.
    """
    samples = max(200, min(samples, 5000))
    bias_strength = max(0.5, min(bias_strength, 2.0))

    try:
        from fastapi.concurrency import run_in_threadpool

        import numpy as np

        from app.services.bias_detector import create_bias_detector
    except ImportError as exc:  # pragma: no cover - optional dependency
        logger.error(f"Fairness dependencies unavailable: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Fairness audit dependencies are not installed.",
        )

    def _run_audit():
        rng = np.random.default_rng(seed)
        gender = rng.choice(["male", "female"], samples, p=[0.6, 0.4])
        race = rng.choice(
            ["white", "black", "hispanic", "asian"],
            samples,
            p=[0.5, 0.2, 0.2, 0.1],
        )
        y_true = rng.binomial(1, 0.3, samples)
        factor = np.where(gender == "male", bias_strength, 2.0 - bias_strength)
        y_prob = np.clip(rng.beta(2, 5, samples) * factor, 0, 1)
        y_pred = (y_prob > 0.5).astype(int)

        detector = create_bias_detector()
        results = detector.detect_bias(
            y_true, y_pred, {"gender": gender, "race": race}, y_prob
        )
        return detector.generate_bias_report(results)

    try:
        report = await run_in_threadpool(_run_audit)
    except Exception as exc:
        _METRICS["errors_total"] += 1
        logger.error(f"Fairness audit failed: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Fairness audit failed: {exc}"
        )

    _METRICS["fairness_audits_total"] += 1
    return _ok(
        {
            "parameters": {
                "samples": samples,
                "bias_strength": bias_strength,
                "seed": seed,
            },
            "report": report,
        }
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus-style metrics exposition for the platform API."""
    lines = []
    for name, value in _METRICS.items():
        metric = f"pulseledger_{name}"
        lines.append(f"# TYPE {metric} counter")
        lines.append(f"{metric} {value}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Run the application
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=config.environment.value == "development",
    )
