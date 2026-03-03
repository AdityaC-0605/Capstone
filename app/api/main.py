"""
Main FastAPI application for the Sustainable Credit Risk AI System.
"""

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add app to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import load_config
from app.core.logging import get_logger

# Initialize FastAPI app
app = FastAPI(
    title="Sustainable Credit Risk AI System",
    description="AI system for credit risk assessment with sustainability features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger
logger = get_logger(__name__)


def _ok(payload):
    return {"status": "ok", "data": payload}


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.error(f"Unhandled API error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": "internal_server_error"},
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return _ok(
        {"message": "Sustainable Credit Risk AI System", "version": "1.0.0"}
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
            "version": "1.0.0",
            "features": [
                "credit-risk-assessment",
                "sustainability-monitoring",
                "federated-learning",
                "explainable-ai",
            ],
        }
    )


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
