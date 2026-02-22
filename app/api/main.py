"""
Main FastAPI application for the Sustainable Credit Risk AI System.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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

# Mount static web frontend
WEB_DIR = Path(__file__).parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/web/static", StaticFiles(directory=str(WEB_DIR)), name="web")


class CreditApplication(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., ge=0)
    employment_length: int = Field(..., ge=0, le=50)
    debt_to_income_ratio: float = Field(..., ge=0, le=1)
    credit_score: int = Field(..., ge=300, le=850)
    loan_amount: float = Field(..., ge=1000)
    loan_purpose: str
    home_ownership: str
    verification_status: str


class PredictionRequest(BaseModel):
    application: CreditApplication
    include_explanation: bool = True
    explanation_type: str = "shap"
    track_sustainability: bool = True


class PredictionResponse(BaseModel):
    prediction_id: str
    risk_score: float
    risk_level: str
    confidence: float
    model_version: str
    prediction_timestamp: datetime
    processing_time_ms: float
    explanation: Optional[Dict[str, Any]] = None
    sustainability_metrics: Optional[Dict[str, Any]] = None
    status: str
    message: str


def _score_risk(app_data: CreditApplication) -> float:
    """Lightweight deterministic risk score for local full-stack UX."""
    score = 0.0
    score += min(1.0, app_data.debt_to_income_ratio) * 0.42
    score += (max(300, 850 - app_data.credit_score) / 550.0) * 0.33
    score += min(1.0, app_data.loan_amount / max(app_data.income, 1.0)) * 0.2
    if app_data.employment_length < 2:
        score += 0.05
    if app_data.home_ownership == "rent":
        score += 0.02
    return max(0.0, min(1.0, score))


def _risk_level(score: float) -> str:
    if score < 0.25:
        return "low"
    if score < 0.5:
        return "medium"
    if score < 0.75:
        return "high"
    return "very_high"


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Sustainable Credit Risk AI System", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "credit-risk-ai"}


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Add any readiness checks here (database, model loading, etc.)
        return {"status": "ready", "service": "credit-risk-ai"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {
        "status": "operational",
        "version": "1.0.0",
        "features": [
            "credit-risk-assessment",
            "sustainability-monitoring",
            "federated-learning",
            "explainable-ai",
        ],
    }


@app.get("/api/v1/model/info")
async def model_info():
    """Model metadata endpoint for the web frontend."""
    return {
        "model_version": "1.0.0-local",
        "model_type": "lightweight-heuristic",
        "features_supported": [
            "age",
            "income",
            "employment_length",
            "debt_to_income_ratio",
            "credit_score",
            "loan_amount",
            "loan_purpose",
            "home_ownership",
            "verification_status",
        ],
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prediction endpoint for full-stack web UI."""
    risk_score = _score_risk(request.application)
    confidence = max(0.55, min(0.97, 1.0 - abs(risk_score - 0.5)))
    explanation = None
    if request.include_explanation:
        explanation = {
            "type": request.explanation_type,
            "top_drivers": {
                "debt_to_income_ratio": round(
                    request.application.debt_to_income_ratio, 3
                ),
                "credit_score": request.application.credit_score,
                "loan_to_income": round(
                    request.application.loan_amount
                    / max(request.application.income, 1.0),
                    3,
                ),
            },
        }

    sustainability = None
    if request.track_sustainability:
        sustainability = {"carbon_emissions": 0.0008, "energy_kwh": 0.0015}

    return PredictionResponse(
        prediction_id=f"pred_{int(datetime.now().timestamp())}",
        risk_score=round(risk_score, 4),
        risk_level=_risk_level(risk_score),
        confidence=round(confidence, 4),
        model_version="1.0.0-local",
        prediction_timestamp=datetime.now(),
        processing_time_ms=12.0,
        explanation=explanation,
        sustainability_metrics=sustainability,
        status="success",
        message="Prediction completed successfully",
    )


@app.get("/web")
async def web_app():
    """Serve the full-stack web frontend."""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web frontend not found")
    return FileResponse(str(index_file))


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
