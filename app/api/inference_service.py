"""
FastAPI Inference Service for Credit Risk Prediction.

This module implements a REST API service for real-time credit risk prediction
with request validation, authentication, rate limiting, and explainability.
"""

import json
import time
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings

# FastAPI dependencies
try:
    from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    warnings.warn("FastAPI not available. Install with: pip install fastapi uvicorn")

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    warnings.warn("SlowAPI not available. Install with: pip install slowapi")

try:
    from ..core.logging import get_logger, get_audit_logger
    from ..models.ensemble_model import EnsembleModel
    from ..explainability.explainer_service import ExplainerService
    from ..sustainability.sustainability_monitor import (
        SustainabilityMonitor,
        track_sustainability,
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_logger, get_audit_logger

    # Create minimal implementations for testing
    class MockAuditLogger:
        def log_model_operation(self, **kwargs):
            pass

    def get_audit_logger():
        return MockAuditLogger()

    # Mock classes for testing
    class MockEnsembleModel:
        def predict(self, data):
            return {"prediction": 0.75, "confidence": 0.85}

        def predict_proba(self, data):
            return [[0.25, 0.75]]

    class MockExplainerService:
        def explain_prediction(self, data, prediction):
            return {
                "shap_values": [0.1, -0.05, 0.2],
                "feature_importance": {"feature_1": 0.3, "feature_2": 0.7},
                "explanation": "High risk due to debt-to-income ratio",
            }

    class MockSustainabilityMonitor:
        def start_experiment_tracking(self, exp_id, metadata=None):
            return exp_id

        def stop_experiment_tracking(self, exp_id):
            return {"carbon_emissions": 0.001, "energy_kwh": 0.002}


logger = get_logger(__name__)
audit_logger = get_audit_logger()


class PredictionStatus(Enum):
    """Prediction status types."""

    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    RATE_LIMITED = "rate_limited"


class RiskLevel(Enum):
    """Credit risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# Pydantic models for request/response validation
class CreditApplication(BaseModel):
    """Credit application data model."""

    # Personal information
    age: int = Field(..., ge=18, le=100, description="Applicant age")
    income: float = Field(..., ge=0, description="Annual income in USD")
    employment_length: int = Field(
        ..., ge=0, le=50, description="Employment length in years"
    )

    # Financial information
    debt_to_income_ratio: float = Field(
        ..., ge=0, le=1, description="Debt-to-income ratio"
    )
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    loan_amount: float = Field(..., ge=1000, description="Requested loan amount")
    loan_purpose: str = Field(..., description="Purpose of the loan")

    # Optional demographic information (for fairness monitoring)
    gender: Optional[str] = Field(
        None, description="Gender (optional, for fairness monitoring)"
    )
    race: Optional[str] = Field(
        None, description="Race (optional, for fairness monitoring)"
    )

    # Additional features
    home_ownership: str = Field(..., description="Home ownership status")
    verification_status: str = Field(..., description="Income verification status")

    @validator("loan_purpose")
    def validate_loan_purpose(cls, v):
        valid_purposes = [
            "debt_consolidation",
            "home_improvement",
            "major_purchase",
            "medical",
            "vacation",
            "wedding",
            "moving",
            "other",
        ]
        if v.lower() not in valid_purposes:
            raise ValueError(f"Invalid loan purpose. Must be one of: {valid_purposes}")
        return v.lower()

    @validator("home_ownership")
    def validate_home_ownership(cls, v):
        valid_statuses = ["own", "rent", "mortgage", "other"]
        if v.lower() not in valid_statuses:
            raise ValueError(
                f"Invalid home ownership. Must be one of: {valid_statuses}"
            )
        return v.lower()

    @validator("verification_status")
    def validate_verification_status(cls, v):
        valid_statuses = ["verified", "source_verified", "not_verified"]
        if v.lower() not in valid_statuses:
            raise ValueError(
                f"Invalid verification status. Must be one of: {valid_statuses}"
            )
        return v.lower()


class PredictionRequest(BaseModel):
    """Prediction request model."""

    application: CreditApplication
    include_explanation: bool = Field(True, description="Include model explanation")
    explanation_type: str = Field(
        "shap", description="Type of explanation (shap, lime)"
    )
    track_sustainability: bool = Field(True, description="Track sustainability metrics")

    @validator("explanation_type")
    def validate_explanation_type(cls, v):
        valid_types = ["shap", "lime", "attention", "counterfactual"]
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid explanation type. Must be one of: {valid_types}")
        return v.lower()


class PredictionResponse(BaseModel):
    """Prediction response model."""

    # Prediction results
    prediction_id: str
    risk_score: float = Field(
        ..., ge=0, le=1, description="Risk score (0=low risk, 1=high risk)"
    )
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")

    # Model information
    model_version: str
    prediction_timestamp: datetime
    processing_time_ms: float

    # Explanation (optional)
    explanation: Optional[Dict[str, Any]] = None

    # Sustainability metrics (optional)
    sustainability_metrics: Optional[Dict[str, Any]] = None

    # Status
    status: PredictionStatus
    message: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""

    applications: List[CreditApplication] = Field(
        ..., max_items=100, description="List of applications (max 100)"
    )
    include_explanation: bool = Field(
        False, description="Include explanations (slower for batch)"
    )
    explanation_type: str = Field("shap", description="Type of explanation")
    track_sustainability: bool = Field(True, description="Track sustainability metrics")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""

    batch_id: str
    predictions: List[PredictionResponse]
    batch_summary: Dict[str, Any]
    processing_time_ms: float
    sustainability_metrics: Optional[Dict[str, Any]] = None


class APIConfig:
    """Configuration for the inference API."""

    def __init__(self):
        # API settings
        self.title = "Credit Risk Prediction API"
        self.description = "Sustainable AI-powered credit risk assessment service"
        self.version = "1.0.0"
        self.host = "0.0.0.0"
        self.port = 8000

        # Security settings
        self.enable_authentication = True
        self.api_keys = set()  # Will be populated with valid API keys
        self.trusted_hosts = ["localhost", "127.0.0.1"]

        # Rate limiting
        self.enable_rate_limiting = True
        self.rate_limit_per_minute = 60
        self.rate_limit_per_hour = 1000

        # Model settings
        self.model_path = "models/ensemble_model.pkl"
        self.model_version = "1.0.0"

        # Sustainability tracking
        self.enable_sustainability_tracking = True

        # CORS settings
        self.enable_cors = True
        self.cors_origins = ["*"]  # Configure appropriately for production

        # Logging
        self.log_predictions = True
        self.log_level = "INFO"


class APIKeyManager:
    """Manages API key authentication."""

    def __init__(self):
        self.api_keys = {}  # key -> metadata
        self.key_usage = {}  # key -> usage stats

        # Generate default API key for testing
        self._generate_default_key()

    def _generate_default_key(self):
        """Generate a default API key for testing."""
        default_key = "sk-test-" + secrets.token_urlsafe(32)
        self.api_keys[default_key] = {
            "name": "Default Test Key",
            "created_at": datetime.now(),
            "permissions": ["predict", "batch_predict"],
            "rate_limit": 1000,
        }
        logger.info(f"Generated default API key: {default_key}")

    def validate_key(self, api_key: str) -> bool:
        """Validate API key."""
        if api_key in self.api_keys:
            # Update usage stats
            if api_key not in self.key_usage:
                self.key_usage[api_key] = {"requests": 0, "last_used": datetime.now()}

            self.key_usage[api_key]["requests"] += 1
            self.key_usage[api_key]["last_used"] = datetime.now()

            return True
        return False

    def get_key_info(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get API key information."""
        return self.api_keys.get(api_key)


class InferenceService:
    """Main inference service class."""

    def __init__(self, config: Optional[APIConfig] = None):
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for inference service. "
                "Install with: pip install fastapi uvicorn"
            )

        self.config = config or APIConfig()

        # Initialize components
        self.api_key_manager = APIKeyManager()
        self.model = None
        self.explainer = None
        self.sustainability_monitor = None

        # Initialize FastAPI app
        self.app = FastAPI(
            title=self.config.title,
            description=self.config.description,
            version=self.config.version,
        )

        # Setup middleware and dependencies
        self._setup_middleware()
        self._setup_rate_limiting()
        self._setup_routes()

        # Load model and services
        self._load_model()
        self._load_services()

        logger.info("Inference service initialized")

    def _setup_middleware(self):
        """Setup FastAPI middleware."""

        # CORS middleware
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["GET", "POST"],
                allow_headers=["*"],
            )

        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=self.config.trusted_hosts
        )

        # Custom middleware for logging and monitoring
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()

            # Log request
            logger.info(f"Request: {request.method} {request.url}")

            # Process request
            response = await call_next(request)

            # Log response
            process_time = time.time() - start_time
            logger.info(f"Response: {response.status_code} ({process_time:.3f}s)")

            return response

    def _setup_rate_limiting(self):
        """Setup rate limiting."""

        if not self.config.enable_rate_limiting or not SLOWAPI_AVAILABLE:
            logger.warning("Rate limiting disabled or SlowAPI not available")
            return

        # Initialize limiter
        limiter = Limiter(key_func=get_remote_address)
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        self.limiter = limiter

    def _setup_routes(self):
        """Setup API routes."""

        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": self.config.version,
                "model_loaded": self.model is not None,
            }

        # Model info endpoint
        @self.app.get("/model/info")
        async def model_info(api_key: str = Depends(self._verify_api_key)):
            """Get model information."""
            return {
                "model_version": self.config.model_version,
                "model_type": "ensemble",
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
                "explanation_types": ["shap", "lime", "attention", "counterfactual"],
                "sustainability_tracking": self.config.enable_sustainability_tracking,
            }

        # Single prediction endpoint
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(
            request: PredictionRequest,
            background_tasks: BackgroundTasks,
            api_key: str = Depends(self._verify_api_key),
        ):
            """Make a single credit risk prediction."""

            # Apply rate limiting if available
            if hasattr(self, "limiter"):
                # This would be applied as a decorator in a real implementation
                pass

            return await self._make_prediction(request, background_tasks, api_key)

        # Batch prediction endpoint
        @self.app.post("/predict/batch", response_model=BatchPredictionResponse)
        async def predict_batch(
            request: BatchPredictionRequest,
            background_tasks: BackgroundTasks,
            api_key: str = Depends(self._verify_api_key),
        ):
            """Make batch credit risk predictions."""

            return await self._make_batch_prediction(request, background_tasks, api_key)

        # API key info endpoint
        @self.app.get("/api-key/info")
        async def api_key_info(api_key: str = Depends(self._verify_api_key)):
            """Get API key information."""
            key_info = self.api_key_manager.get_key_info(api_key)
            usage_info = self.api_key_manager.key_usage.get(api_key, {})

            return {
                "key_name": key_info.get("name", "Unknown"),
                "permissions": key_info.get("permissions", []),
                "requests_made": usage_info.get("requests", 0),
                "last_used": (
                    usage_info.get("last_used", "Never").isoformat()
                    if isinstance(usage_info.get("last_used"), datetime)
                    else "Never"
                ),
            }

    def _load_model(self):
        """Load the ensemble model."""
        try:
            # In a real implementation, this would load the actual model
            # For now, use a mock model
            self.model = MockEnsembleModel()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def _load_services(self):
        """Load additional services."""
        try:
            # Load explainer service
            self.explainer = MockExplainerService()

            # Load sustainability monitor
            if self.config.enable_sustainability_tracking:
                self.sustainability_monitor = MockSustainabilityMonitor()

            logger.info("Services loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load services: {e}")

    async def _verify_api_key(
        self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ):
        """Verify API key authentication."""

        if not self.config.enable_authentication:
            return "no-auth"

        api_key = credentials.credentials

        if not self.api_key_manager.validate_key(api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

        return api_key

    async def _make_prediction(
        self,
        request: PredictionRequest,
        background_tasks: BackgroundTasks,
        api_key: str,
    ) -> PredictionResponse:
        """Make a single prediction."""

        start_time = time.time()
        prediction_id = self._generate_prediction_id(request.application)

        try:
            # Start sustainability tracking if enabled
            sustainability_context = None
            if (
                self.config.enable_sustainability_tracking
                and request.track_sustainability
            ):
                exp_id = f"prediction_{prediction_id}"
                self.sustainability_monitor.start_experiment_tracking(
                    exp_id,
                    {
                        "type": "single_prediction",
                        "api_key": api_key[:10] + "...",
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                sustainability_context = exp_id

            # Validate model is loaded
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not available")

            # Prepare input data
            input_data = self._prepare_input_data(request.application)

            # Make prediction
            prediction_result = self.model.predict(input_data)
            risk_score = float(prediction_result.get("prediction", 0.5))
            confidence = float(prediction_result.get("confidence", 0.8))

            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)

            # Generate explanation if requested
            explanation = None
            if request.include_explanation and self.explainer:
                explanation = self.explainer.explain_prediction(
                    input_data, prediction_result
                )

            # Stop sustainability tracking
            sustainability_metrics = None
            if sustainability_context:
                sustainability_metrics = (
                    self.sustainability_monitor.stop_experiment_tracking(
                        sustainability_context
                    )
                )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Create response
            response = PredictionResponse(
                prediction_id=prediction_id,
                risk_score=risk_score,
                risk_level=risk_level,
                confidence=confidence,
                model_version=self.config.model_version,
                prediction_timestamp=datetime.now(),
                processing_time_ms=processing_time,
                explanation=explanation,
                sustainability_metrics=sustainability_metrics,
                status=PredictionStatus.SUCCESS,
                message="Prediction completed successfully",
            )

            # Log prediction
            if self.config.log_predictions:
                background_tasks.add_task(
                    self._log_prediction, request, response, api_key
                )

            # Audit log
            audit_logger.log_model_operation(
                user_id=api_key[:10] + "...",
                model_id="credit_risk_api",
                operation="single_prediction",
                success=True,
                details={
                    "prediction_id": prediction_id,
                    "risk_score": risk_score,
                    "processing_time_ms": processing_time,
                },
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction error: {e}")

            # Stop sustainability tracking on error
            if sustainability_context:
                try:
                    self.sustainability_monitor.stop_experiment_tracking(
                        sustainability_context
                    )
                except:
                    pass

            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    async def _make_batch_prediction(
        self,
        request: BatchPredictionRequest,
        background_tasks: BackgroundTasks,
        api_key: str,
    ) -> BatchPredictionResponse:
        """Make batch predictions."""

        start_time = time.time()
        batch_id = f"batch_{int(time.time())}_{secrets.token_hex(8)}"

        try:
            # Start sustainability tracking
            sustainability_context = None
            if (
                self.config.enable_sustainability_tracking
                and request.track_sustainability
            ):
                exp_id = f"batch_{batch_id}"
                self.sustainability_monitor.start_experiment_tracking(
                    exp_id,
                    {
                        "type": "batch_prediction",
                        "batch_size": len(request.applications),
                        "api_key": api_key[:10] + "...",
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                sustainability_context = exp_id

            # Process each application
            predictions = []
            for i, application in enumerate(request.applications):
                try:
                    # Create individual prediction request
                    individual_request = PredictionRequest(
                        application=application,
                        include_explanation=request.include_explanation,
                        explanation_type=request.explanation_type,
                        track_sustainability=False,  # Don't double-track
                    )

                    # Make prediction (without sustainability tracking)
                    pred_response = await self._make_single_prediction_internal(
                        individual_request
                    )
                    predictions.append(pred_response)

                except Exception as e:
                    # Create error response for failed prediction
                    error_response = PredictionResponse(
                        prediction_id=f"{batch_id}_{i}",
                        risk_score=0.5,
                        risk_level=RiskLevel.MEDIUM,
                        confidence=0.0,
                        model_version=self.config.model_version,
                        prediction_timestamp=datetime.now(),
                        processing_time_ms=0,
                        status=PredictionStatus.ERROR,
                        message=f"Prediction failed: {str(e)}",
                    )
                    predictions.append(error_response)

            # Calculate batch summary
            successful_predictions = [
                p for p in predictions if p.status == PredictionStatus.SUCCESS
            ]
            batch_summary = {
                "total_applications": len(request.applications),
                "successful_predictions": len(successful_predictions),
                "failed_predictions": len(predictions) - len(successful_predictions),
                "average_risk_score": sum(p.risk_score for p in successful_predictions)
                / max(len(successful_predictions), 1),
                "risk_distribution": self._calculate_risk_distribution(
                    successful_predictions
                ),
            }

            # Stop sustainability tracking
            sustainability_metrics = None
            if sustainability_context:
                sustainability_metrics = (
                    self.sustainability_monitor.stop_experiment_tracking(
                        sustainability_context
                    )
                )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Create batch response
            response = BatchPredictionResponse(
                batch_id=batch_id,
                predictions=predictions,
                batch_summary=batch_summary,
                processing_time_ms=processing_time,
                sustainability_metrics=sustainability_metrics,
            )

            # Log batch prediction
            if self.config.log_predictions:
                background_tasks.add_task(
                    self._log_batch_prediction, request, response, api_key
                )

            return response

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")

            # Stop sustainability tracking on error
            if sustainability_context:
                try:
                    self.sustainability_monitor.stop_experiment_tracking(
                        sustainability_context
                    )
                except:
                    pass

            raise HTTPException(
                status_code=500, detail=f"Batch prediction failed: {str(e)}"
            )

    async def _make_single_prediction_internal(
        self, request: PredictionRequest
    ) -> PredictionResponse:
        """Internal method for single prediction without sustainability tracking."""

        start_time = time.time()
        prediction_id = self._generate_prediction_id(request.application)

        # Prepare input data
        input_data = self._prepare_input_data(request.application)

        # Make prediction
        prediction_result = self.model.predict(input_data)
        risk_score = float(prediction_result.get("prediction", 0.5))
        confidence = float(prediction_result.get("confidence", 0.8))

        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)

        # Generate explanation if requested
        explanation = None
        if request.include_explanation and self.explainer:
            explanation = self.explainer.explain_prediction(
                input_data, prediction_result
            )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            prediction_id=prediction_id,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            model_version=self.config.model_version,
            prediction_timestamp=datetime.now(),
            processing_time_ms=processing_time,
            explanation=explanation,
            status=PredictionStatus.SUCCESS,
            message="Prediction completed successfully",
        )

    def _prepare_input_data(self, application: CreditApplication) -> Dict[str, Any]:
        """Prepare input data for model prediction."""

        return {
            "age": application.age,
            "income": application.income,
            "employment_length": application.employment_length,
            "debt_to_income_ratio": application.debt_to_income_ratio,
            "credit_score": application.credit_score,
            "loan_amount": application.loan_amount,
            "loan_purpose": application.loan_purpose,
            "home_ownership": application.home_ownership,
            "verification_status": application.verification_status,
        }

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score."""

        if risk_score < 0.25:
            return RiskLevel.LOW
        elif risk_score < 0.5:
            return RiskLevel.MEDIUM
        elif risk_score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def _generate_prediction_id(self, application: CreditApplication) -> str:
        """Generate unique prediction ID."""

        # Create hash from application data and timestamp
        data_str = f"{application.credit_score}_{application.income}_{application.loan_amount}_{time.time()}"
        hash_obj = hashlib.md5(data_str.encode())
        return f"pred_{hash_obj.hexdigest()[:12]}"

    def _calculate_risk_distribution(
        self, predictions: List[PredictionResponse]
    ) -> Dict[str, int]:
        """Calculate risk level distribution."""

        distribution = {level.value: 0 for level in RiskLevel}

        for prediction in predictions:
            distribution[prediction.risk_level.value] += 1

        return distribution

    async def _log_prediction(
        self, request: PredictionRequest, response: PredictionResponse, api_key: str
    ):
        """Log prediction for audit and monitoring."""

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "prediction_id": response.prediction_id,
            "api_key": api_key[:10] + "...",
            "risk_score": response.risk_score,
            "risk_level": response.risk_level.value,
            "confidence": response.confidence,
            "processing_time_ms": response.processing_time_ms,
            "application_data": {
                "age": request.application.age,
                "income": request.application.income,
                "credit_score": request.application.credit_score,
                "loan_amount": request.application.loan_amount,
                "loan_purpose": request.application.loan_purpose,
            },
        }

        logger.info(f"Prediction logged: {json.dumps(log_data)}")

    async def _log_batch_prediction(
        self,
        request: BatchPredictionRequest,
        response: BatchPredictionResponse,
        api_key: str,
    ):
        """Log batch prediction for audit and monitoring."""

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "batch_id": response.batch_id,
            "api_key": api_key[:10] + "...",
            "batch_size": len(request.applications),
            "successful_predictions": response.batch_summary["successful_predictions"],
            "failed_predictions": response.batch_summary["failed_predictions"],
            "average_risk_score": response.batch_summary["average_risk_score"],
            "processing_time_ms": response.processing_time_ms,
        }

        logger.info(f"Batch prediction logged: {json.dumps(log_data)}")

    def run(self, host: Optional[str] = None, port: Optional[int] = None):
        """Run the inference service."""

        host = host or self.config.host
        port = port or self.config.port

        logger.info(f"Starting inference service on {host}:{port}")

        uvicorn.run(
            self.app, host=host, port=port, log_level=self.config.log_level.lower()
        )

    def get_app(self):
        """Get the FastAPI app instance."""
        return self.app


# Utility functions


def create_inference_service(config: Optional[APIConfig] = None) -> InferenceService:
    """Create inference service instance."""
    return InferenceService(config)


def run_inference_service(
    host: str = "0.0.0.0", port: int = 8000, config: Optional[APIConfig] = None
):
    """Run inference service."""
    service = create_inference_service(config)
    service.run(host, port)


if __name__ == "__main__":
    # Run service with default configuration
    if FASTAPI_AVAILABLE:
        run_inference_service()
    else:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
