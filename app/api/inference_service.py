"""
FastAPI Inference Service for Credit Risk Prediction.

This module implements a REST API service for real-time credit risk prediction
with request validation, authentication, rate limiting, and explainability.
"""

import hashlib
import json
import os
import secrets
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Union

# FastAPI dependencies
try:
    import uvicorn
    from fastapi import (
        BackgroundTasks,
        Depends,
        FastAPI,
        HTTPException,
        Request,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field, field_validator

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    warnings.warn(
        "FastAPI not available. Install with: pip install fastapi uvicorn"
    )

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    warnings.warn("SlowAPI not available. Install with: pip install slowapi")

try:
    from ..core.logging import get_audit_logger, get_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_audit_logger, get_logger

    # Create minimal implementations for testing
    class MockAuditLogger:
        def log_model_operation(self, **kwargs):
            pass

    def get_audit_logger():
        return MockAuditLogger()


try:
    from ..sustainability.sustainability_monitor import SustainabilityMonitor
except ImportError:
    SustainabilityMonitor = None  # type: ignore[assignment]

try:
    from ..explainability.explanation_service import ExplainerService
except ImportError:
    ExplainerService = None  # type: ignore[assignment]

try:
    from ..models.runtime_credit_model import LightweightCreditRiskModel
except ImportError:
    LightweightCreditRiskModel = None  # type: ignore[assignment]


class MockExplainerService:
    """Fallback explainer used when the real ExplainerService fails to import.

    Returns a response matching the full explanation schema so downstream
    consumers (API, frontend) always receive a consistent structure.
    """

    def explain_prediction(self, data, prediction):
        risk_score = float(prediction.get("prediction", 0.5))
        if risk_score < 0.25:
            risk_level = "low"
        elif risk_score < 0.5:
            risk_level = "medium"
        elif risk_score < 0.75:
            risk_level = "high"
        else:
            risk_level = "very_high"

        mock_importance = {
            "debt_to_income_ratio": 0.3,
            "credit_score": -0.2,
            "income": -0.1,
        }

        risk_tag = risk_level.replace("_", " ").title()

        return {
            "prediction": risk_score,
            "risk_level": risk_level,
            "risk_threshold_context": (
                f"Score {risk_score:.3f} falls in the {risk_tag} band."
            ),
            "risk_thresholds": [
                {"level": "low", "max_score": 0.25},
                {"level": "medium", "max_score": 0.5},
                {"level": "high", "max_score": 0.75},
                {"level": "very_high", "max_score": 1.01},
            ],
            "feature_importance": mock_importance,
            "top_factors": [],
            "recommendations": [],
            "counterfactual": {
                "needed": False,
                "message": "Mock explainer: no counterfactual available.",
                "changes": {},
            },
            "risk_groups": {},
            "confidence": {
                "level": "low",
                "score": 0.0,
                "reason": "Mock explainer; real SHAP analysis unavailable.",
            },
            "methodology": {
                "method": "Mock (fallback)",
                "description": (
                    "The real SHAP explainer could not be loaded. "
                    "This is a placeholder response with static values."
                ),
                "interpretation": (
                    "These values are illustrative only and should "
                    "not be used for decision-making."
                ),
                "baseline": {
                    "description": "Not available in mock mode.",
                    "baseline_values": {},
                },
            },
            "summary": (
                f"Predicted risk: {risk_tag} (score {risk_score:.3f}). "
                f"Detailed explanation unavailable (mock explainer)."
            ),
        }


class MockSustainabilityMonitor:
    def start_experiment_tracking(self, exp_id, metadata=None):
        return exp_id

    def stop_experiment_tracking(self, exp_id):
        return {
            "carbon_emissions": 0.001,
            "energy_kwh": 0.002,
            "duration_seconds": 0.0,
            "method": "mock",
            "region": "US",
            "emissions_factor_kg_per_kwh": 0.385,
        }


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
    loan_amount: float = Field(
        ..., ge=1000, description="Requested loan amount"
    )
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
    verification_status: str = Field(
        ..., description="Income verification status"
    )

    @field_validator("loan_purpose")
    @classmethod
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
            raise ValueError(
                f"Invalid loan purpose. Must be one of: {valid_purposes}"
            )
        return v.lower()

    @field_validator("home_ownership")
    @classmethod
    def validate_home_ownership(cls, v):
        valid_statuses = ["own", "rent", "mortgage", "other"]
        if v.lower() not in valid_statuses:
            raise ValueError(
                f"Invalid home ownership. Must be one of: {valid_statuses}"
            )
        return v.lower()

    @field_validator("verification_status")
    @classmethod
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
    include_explanation: bool = Field(
        True, description="Include model explanation"
    )
    explanation_type: str = Field("shap", description="Type of explanation")
    track_sustainability: bool = Field(
        True, description="Track sustainability metrics"
    )

    @field_validator("explanation_type")
    @classmethod
    def validate_explanation_type(cls, v):
        valid_types = ["shap"]
        if v.lower() not in valid_types:
            raise ValueError(
                f"Invalid explanation type. Must be one of: {valid_types}"
            )
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
        ..., max_length=100, description="List of applications (max 100)"
    )
    include_explanation: bool = Field(
        False, description="Include explanations (slower for batch)"
    )
    explanation_type: str = Field("shap", description="Type of explanation")
    track_sustainability: bool = Field(
        True, description="Track sustainability metrics"
    )

    @field_validator("explanation_type")
    @classmethod
    def validate_explanation_type(cls, v):
        valid_types = ["shap"]
        if v.lower() not in valid_types:
            raise ValueError(
                f"Invalid explanation type. Must be one of: {valid_types}"
            )
        return v.lower()


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""

    batch_id: str
    predictions: List[PredictionResponse]
    batch_summary: Dict[str, Any]
    processing_time_ms: float
    sustainability_metrics: Optional[Dict[str, Any]] = None


@dataclass
class Principal:
    """The authenticated caller: a logged-in user or a service API key."""

    kind: str  # "user" | "service" | "anonymous"
    identifier: str
    user_id: Optional[int] = None


class RegisterRequest(BaseModel):
    """New-user registration payload."""

    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password (8+ chars)")
    full_name: str = Field("", description="Display name")


class LoginRequest(BaseModel):
    """Login payload."""

    email: str
    password: str


class AuthResponse(BaseModel):
    """Issued bearer token plus the public user record."""

    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class APIConfig:
    """Configuration for the inference API."""

    def __init__(self):
        # API settings
        self.title = "Credit Risk Prediction API"
        self.description = (
            "Sustainable AI-powered credit risk assessment service"
        )
        self.version = "1.0.0"
        self.host = "0.0.0.0"
        self.port = 8001

        # Security settings
        self.enable_authentication = True
        self.api_keys = set()  # Will be populated with valid API keys
        # Trusted hosts: env override (comma list or "*") for deployments;
        # defaults to local + "testserver" so the test client keeps working.
        _hosts = os.getenv("PULSELEDGER_TRUSTED_HOSTS", "").strip()
        if _hosts == "*":
            self.trusted_hosts = ["*"]
        elif _hosts:
            self.trusted_hosts = [
                h.strip() for h in _hosts.split(",") if h.strip()
            ]
        else:
            self.trusted_hosts = ["localhost", "127.0.0.1", "testserver"]

        # Rate limiting
        self.enable_rate_limiting = True
        self.rate_limit_per_minute = 60
        self.rate_limit_per_hour = 1000

        # Model settings
        self.model_path = "models/mlp_logistic_model.pkl"
        self.model_version = "1.0.0"

        # Sustainability tracking
        self.enable_sustainability_tracking = True

        # CORS settings — env override for deployments (comma list or "*").
        self.enable_cors = True
        _origins = os.getenv("PULSELEDGER_ALLOWED_ORIGINS", "*").strip()
        if not _origins or _origins == "*":
            self.cors_origins = ["*"]
        else:
            self.cors_origins = [
                o.strip() for o in _origins.split(",") if o.strip()
            ]

        # Logging
        self.log_predictions = True
        self.log_level = "INFO"


class APIKeyManager:
    """Manages API key authentication."""

    def __init__(self, key_path: Optional[str] = None):
        self.api_keys = {}  # key -> metadata
        self.key_usage = {}  # key -> usage stats
        self.key_path = Path(
            key_path
            or os.getenv("PULSELEDGER_API_KEY_FILE", "keys/api_key.txt")
        )
        self.default_key: Optional[str] = None

        # Resolve a *persistent* default API key.
        self._load_or_create_key()

    def _load_or_create_key(self):
        """Resolve the default API key from env -> file -> freshly generated.

        Persisting the key means the frontend's saved bearer token keeps
        working across backend restarts. Previously a brand-new key was minted
        on every startup, silently invalidating the token stored in the UI.
        Precedence: ``PULSELEDGER_API_KEY`` env var, then the on-disk key file,
        then a newly generated key written back to that file.
        """
        env_key = os.getenv("PULSELEDGER_API_KEY")
        if env_key and env_key.strip():
            key = env_key.strip()
            source = "environment"
        elif self.key_path.exists():
            key = self.key_path.read_text(encoding="utf-8").strip()
            source = f"file {self.key_path}"
        else:
            key = "sk-test-" + secrets.token_urlsafe(32)
            try:
                self.key_path.parent.mkdir(parents=True, exist_ok=True)
                self.key_path.write_text(key, encoding="utf-8")
                try:
                    os.chmod(self.key_path, 0o600)
                except OSError:
                    pass
                source = f"generated, saved to {self.key_path}"
            except OSError as exc:
                source = f"generated (in-memory only; persist failed: {exc})"

        self.api_keys[key] = {
            "name": "Default Test Key",
            "created_at": datetime.now(),
            "permissions": ["predict", "batch_predict"],
            "rate_limit": 1000,
        }
        self.default_key = key
        logger.info(f"Active API key [{source}]: {key}")

    def validate_key(self, api_key: str) -> bool:
        """Validate API key."""
        if api_key in self.api_keys:
            # Update usage stats
            if api_key not in self.key_usage:
                self.key_usage[api_key] = {
                    "requests": 0,
                    "last_used": datetime.now(),
                }

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

        # In-memory rolling prediction history + counters (surfaced via API).
        # The deque is a fast fallback; durable history lives in the database.
        self.prediction_history: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.metrics = {
            "predictions_total": 0,
            "batch_predictions_total": 0,
            "prediction_errors_total": 0,
            "auth_failures_total": 0,
        }

        # Holder for the most recent carbon-aware NAS run (single job at a
        # time; a background thread fills this in). Guarded by a lock.
        self._nas_lock = threading.Lock()
        self._nas_job: Dict[str, Any] = {
            "state": "idle",
            "result": None,
            "started_at": None,
        }

        # Durable persistence (SQLite by default; Postgres via
        # PULSELEDGER_DATABASE_URL). Degrades to memory-only if unavailable.
        self.db_available = False
        try:
            from app.db import repository

            if not repository.is_configured():
                repository.configure()
            self.db_available = True
            self._seed_demo_user()
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning(
                f"Persistence unavailable, using memory only: {exc}"
            )

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
            logger.info(
                f"Response: {response.status_code} ({process_time:.3f}s)"
            )

            return response

    def _setup_rate_limiting(self):
        """Setup rate limiting."""

        if not self.config.enable_rate_limiting or not SLOWAPI_AVAILABLE:
            logger.warning("Rate limiting disabled or SlowAPI not available")
            return

        # Initialize limiter
        limiter = Limiter(key_func=get_remote_address)
        self.app.state.limiter = limiter
        self.app.add_exception_handler(
            RateLimitExceeded, _rate_limit_exceeded_handler
        )

        self.limiter = limiter

    def _rate_limit(self, limit_value: str):
        """Return a slowapi rate-limit decorator, or a no-op if rate limiting
        is unavailable/disabled. This lets the route definitions stay uniform.
        """
        limiter = getattr(self, "limiter", None)
        if limiter is not None:
            return limiter.limit(limit_value)

        def _noop(func):
            return func

        return _noop

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/")
        async def root():
            """Service metadata endpoint."""
            return {
                "service": "credit-risk-inference",
                "version": self.config.version,
                "docs_url": "/docs",
                "health_url": "/health",
            }

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
            """Get model information (incl. trained-registry metadata)."""
            model_source = getattr(self.model, "model_source", "unavailable")
            registry: Dict[str, Any] = {}
            try:
                from pathlib import Path

                meta_path = Path("model_registry/registry.json")
                if meta_path.exists():
                    registry = json.loads(meta_path.read_text()).get(
                        "credit_risk_model", {}
                    )
            except Exception:
                registry = {}
            return {
                "model_version": self.config.model_version,
                "model_type": "runtime_credit_risk",
                "model_source": model_source,
                "algorithm": registry.get("type"),
                "roc_auc": registry.get("holdout_roc_auc"),
                "trained_at": registry.get("trained_at"),
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
                "explanation_types": [
                    "shap",
                ],
                "sustainability_tracking": self.config.enable_sustainability_tracking,
            }

        # Single prediction endpoint (rate limited per client IP)
        @self.app.post("/predict", response_model=PredictionResponse)
        @self._rate_limit(f"{self.config.rate_limit_per_minute}/minute")
        async def predict(
            request: Request,
            payload: PredictionRequest,
            background_tasks: BackgroundTasks,
            principal: "Principal" = Depends(self._authenticate),
        ):
            """Make a single credit risk prediction."""
            return await self._make_prediction(
                payload, background_tasks, principal
            )

        # Lightweight live preview: score only, no explanation, no
        # persistence, no metrics — safe to call on every input change.
        @self.app.post("/predict/preview")
        async def predict_preview(
            payload: PredictionRequest,
            principal: "Principal" = Depends(self._authenticate),
        ):
            if self.model is None:
                raise HTTPException(
                    status_code=503, detail="Model not available"
                )
            input_data = self._prepare_input_data(payload.application)
            result = self.model.predict(input_data)
            score = float(result.get("prediction", 0.5))
            return {
                "risk_score": score,
                "risk_level": self._determine_risk_level(score).value,
                "confidence": float(result.get("confidence", 0.8)),
            }

        # Batch prediction endpoint (tighter limit; each call fans out)
        @self.app.post(
            "/predict/batch", response_model=BatchPredictionResponse
        )
        @self._rate_limit(
            f"{max(1, self.config.rate_limit_per_minute // 4)}/minute"
        )
        async def predict_batch(
            request: Request,
            payload: BatchPredictionRequest,
            background_tasks: BackgroundTasks,
            principal: "Principal" = Depends(self._authenticate),
        ):
            """Make batch credit risk predictions."""
            return await self._make_batch_prediction(
                payload, background_tasks, principal.identifier
            )

        # Recent prediction history (durable when a database is configured;
        # falls back to the in-memory rolling window otherwise).
        @self.app.get("/predict/history")
        async def prediction_history(
            limit: int = 25,
            principal: "Principal" = Depends(self._authenticate),
        ):
            """Return recent predictions, newest first.

            A logged-in user sees only their own assessments; a service key
            sees all of them.
            """
            limit = max(1, min(limit, 200))
            scope = principal.user_id if principal.kind == "user" else None
            if self.db_available:
                try:
                    from fastapi.concurrency import run_in_threadpool

                    from app.db import repository

                    items = await run_in_threadpool(
                        repository.list_predictions, limit, scope
                    )
                    total = await run_in_threadpool(
                        repository.count_predictions, scope
                    )
                    return {
                        "count": len(items),
                        "total_served": total,
                        "items": items,
                    }
                except Exception as exc:
                    logger.error(f"History DB read failed: {exc}")

            items = list(self.prediction_history)[-limit:][::-1]
            return {
                "count": len(items),
                "total_served": self.metrics["predictions_total"],
                "items": items,
            }

        # A single persisted assessment by id (full record for detail views).
        # Declared after /predict/history so the literal path wins.
        @self.app.get("/predict/{prediction_id}")
        async def get_one_prediction(
            prediction_id: str,
            principal: "Principal" = Depends(self._authenticate),
        ):
            """Fetch one persisted assessment a user owns (or any, for keys)."""
            if self.db_available:
                from fastapi.concurrency import run_in_threadpool

                from app.db import repository

                scope = principal.user_id if principal.kind == "user" else None
                record = await run_in_threadpool(
                    repository.get_prediction, prediction_id, scope
                )
                if record:
                    return {"status": "ok", "data": record}
            raise HTTPException(status_code=404, detail="Prediction not found")

        @self.app.get("/sustainability/summary")
        async def sustainability_summary(
            principal: "Principal" = Depends(self._authenticate),
        ):
            """Account-wide energy/carbon totals from persisted history.

            Survives reloads and aggregates every assessment the principal has
            scored — not just the current browser session.
            """
            scope = principal.user_id if principal.kind == "user" else None
            if self.db_available:
                try:
                    from fastapi.concurrency import run_in_threadpool

                    from app.db import repository

                    totals = await run_in_threadpool(
                        repository.sustainability_totals, scope
                    )
                    return {"status": "ok", "data": totals}
                except Exception as exc:
                    logger.error(f"Sustainability summary failed: {exc}")
            return {
                "status": "ok",
                "data": {
                    "count": 0,
                    "total_energy_kwh": 0.0,
                    "total_carbon_kg": 0.0,
                    "total_duration_seconds": 0.0,
                    "method": None,
                    "region": None,
                },
            }

        @self.app.post("/sustainability/nas")
        async def start_nas_run(
            principal: "Principal" = Depends(self._authenticate),
        ):
            """Kick off a real (bounded) carbon-aware NAS run in the
            background. Returns immediately; poll /sustainability/nas/status.
            """
            with self._nas_lock:
                if self._nas_job["state"] == "running":
                    return {"status": "running", "message": "already running"}
                self._nas_job = {
                    "state": "running",
                    "result": None,
                    "started_at": datetime.now().isoformat(),
                }

            def _worker():
                from app.sustainability.nas_runner import _safe_run_quick_nas

                outcome = _safe_run_quick_nas()
                with self._nas_lock:
                    self._nas_job["state"] = (
                        "error" if outcome.get("status") == "error" else "done"
                    )
                    self._nas_job["result"] = outcome

            threading.Thread(target=_worker, daemon=True).start()
            return {"status": "running", "message": "NAS run started"}

        @self.app.get("/sustainability/nas/status")
        async def nas_status(
            principal: "Principal" = Depends(self._authenticate),
        ):
            with self._nas_lock:
                return {
                    "state": self._nas_job["state"],
                    "started_at": self._nas_job["started_at"],
                    "result": self._nas_job["result"],
                }

        # Prometheus-style metrics exposition
        @self.app.get("/metrics")
        async def metrics():
            lines = []
            for name, value in self.metrics.items():
                metric = f"pulseledger_inference_{name}"
                lines.append(f"# TYPE {metric} counter")
                lines.append(f"{metric} {value}")
            from fastapi.responses import PlainTextResponse

            return PlainTextResponse("\n".join(lines) + "\n")

        # API key info endpoint
        @self.app.get("/api-key/info")
        async def api_key_info(api_key: str = Depends(self._verify_api_key)):
            """Get API key information (service keys only)."""
            key_info = self.api_key_manager.get_key_info(api_key)
            if not key_info:
                return {
                    "key_name": "session",
                    "permissions": ["predict", "batch_predict"],
                    "requests_made": 0,
                    "last_used": "Never",
                }
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

        # ── Authentication ──────────────────────────────────────────────
        @self.app.post("/auth/register", response_model=AuthResponse)
        async def register(payload: RegisterRequest):
            return await self._register_user(payload)

        @self.app.post("/auth/login", response_model=AuthResponse)
        async def login(payload: LoginRequest):
            return await self._login_user(payload)

        @self.app.get("/auth/me")
        async def me(principal: "Principal" = Depends(self._authenticate)):
            if principal.kind != "user" or principal.user_id is None:
                raise HTTPException(
                    status_code=401, detail="Not a user session"
                )
            from fastapi.concurrency import run_in_threadpool

            from app.db import repository

            user = await run_in_threadpool(
                repository.get_user_by_id, principal.user_id
            )
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            return {"status": "ok", "data": user}

    def _load_model(self):
        """Load the lightweight runtime model."""
        try:
            if LightweightCreditRiskModel is None:
                raise ImportError("Runtime credit risk model is unavailable")
            self.model = LightweightCreditRiskModel()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def _load_services(self):
        """Load additional services."""
        try:
            if ExplainerService is not None:
                self.explainer = ExplainerService(self.model)  # type: ignore[operator]
            else:
                self.explainer = MockExplainerService()

            # Load sustainability monitor
            if self.config.enable_sustainability_tracking:
                if SustainabilityMonitor is not None:
                    self.sustainability_monitor = SustainabilityMonitor()
                else:
                    self.sustainability_monitor = MockSustainabilityMonitor()

            logger.info("Services loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load services: {e}")

    def _seed_demo_user(self) -> None:
        """Create the demo analyst account if it doesn't exist yet."""
        try:
            from app.core.security import hash_password
            from app.db import repository

            email = os.getenv("PULSELEDGER_DEMO_EMAIL", "demo@pulseledger.app")
            password = os.getenv("PULSELEDGER_DEMO_PASSWORD", "demo12345")
            repository.ensure_user(
                email, hash_password(password), full_name="Demo Analyst"
            )
        except Exception as exc:
            logger.warning(f"Demo user seed skipped: {exc}")

    def _principal_from_token(self, token: str) -> "Principal":
        """Resolve a bearer token to a Principal.

        Accepts a user-session JWT first, then a legacy service API key.
        """
        from app.core.security import decode_access_token

        payload = decode_access_token(token)
        if payload and payload.get("sub"):
            try:
                user_id = int(payload["sub"])
            except (TypeError, ValueError):
                user_id = None
            if user_id is not None:
                email = payload.get("email") or f"user:{user_id}"
                return Principal(
                    kind="user", identifier=email, user_id=user_id
                )

        if self.api_key_manager.validate_key(token):
            return Principal(kind="service", identifier=token[:10] + "...")

        self.metrics["auth_failures_total"] += 1
        raise HTTPException(status_code=401, detail="Invalid credentials")

    async def _authenticate(
        self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> "Principal":
        """FastAPI dependency yielding the authenticated Principal."""
        if not self.config.enable_authentication:
            return Principal(kind="anonymous", identifier="no-auth")
        return self._principal_from_token(credentials.credentials)

    async def _verify_api_key(
        self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> str:
        """Backwards-compatible dependency returning the principal's id."""
        if not self.config.enable_authentication:
            return "no-auth"
        return self._principal_from_token(credentials.credentials).identifier

    async def _register_user(
        self, payload: "RegisterRequest"
    ) -> "AuthResponse":
        from fastapi.concurrency import run_in_threadpool

        from app.core.security import create_access_token, hash_password
        from app.db import repository

        if not self.db_available:
            raise HTTPException(
                status_code=503, detail="Auth requires a database"
            )
        email = payload.email.lower().strip()
        if "@" not in email or "." not in email:
            raise HTTPException(status_code=422, detail="Invalid email")
        existing = await run_in_threadpool(repository.get_user_by_email, email)
        if existing:
            raise HTTPException(
                status_code=409, detail="Email already registered"
            )
        user = await run_in_threadpool(
            repository.create_user,
            email,
            hash_password(payload.password),
            payload.full_name,
        )
        token = create_access_token(
            user["id"], email=user["email"], role=user["role"]
        )
        return AuthResponse(access_token=token, user=user)

    async def _login_user(self, payload: "LoginRequest") -> "AuthResponse":
        from fastapi.concurrency import run_in_threadpool

        from app.core.security import create_access_token, verify_password
        from app.db import repository

        if not self.db_available:
            raise HTTPException(
                status_code=503, detail="Auth requires a database"
            )
        record = await run_in_threadpool(
            repository.get_user_by_email, payload.email.lower().strip()
        )
        if not record or not verify_password(
            payload.password, record["hashed_password"]
        ):
            self.metrics["auth_failures_total"] += 1
            raise HTTPException(
                status_code=401, detail="Invalid email or password"
            )
        public = {
            key: record[key] for key in ("id", "email", "full_name", "role")
        }
        token = create_access_token(
            record["id"], email=record["email"], role=record["role"]
        )
        return AuthResponse(access_token=token, user=public)

    async def _make_prediction(
        self,
        request: PredictionRequest,
        background_tasks: BackgroundTasks,
        principal: "Principal",
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
                        "api_key": principal.identifier,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                sustainability_context = exp_id

            # Validate model is loaded
            if self.model is None:
                raise HTTPException(
                    status_code=503, detail="Model not available"
                )

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

            # Track metrics + rolling server-side history
            self.metrics["predictions_total"] += 1
            self.prediction_history.append(
                {
                    "prediction_id": prediction_id,
                    "timestamp": response.prediction_timestamp.isoformat(),
                    "risk_score": round(risk_score, 4),
                    "risk_level": risk_level.value,
                    "confidence": round(confidence, 4),
                    "processing_time_ms": round(processing_time, 2),
                    "loan_amount": request.application.loan_amount,
                    "loan_purpose": request.application.loan_purpose,
                }
            )

            # Log prediction
            if self.config.log_predictions:
                background_tasks.add_task(
                    self._log_prediction, request, response, principal
                )

            # Audit log
            audit_logger.log_model_operation(
                user_id=principal.identifier,
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
            self.metrics["prediction_errors_total"] += 1
            logger.error(f"Prediction error: {e}")

            # Stop sustainability tracking on error
            if sustainability_context:
                try:
                    self.sustainability_monitor.stop_experiment_tracking(
                        sustainability_context
                    )
                except Exception:
                    pass

            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {str(e)}"
            )

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
                    pred_response = (
                        await self._make_single_prediction_internal(
                            individual_request
                        )
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
                "failed_predictions": len(predictions)
                - len(successful_predictions),
                "average_risk_score": sum(
                    p.risk_score for p in successful_predictions
                )
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

            self.metrics["batch_predictions_total"] += 1

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
                except Exception:
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

    def _prepare_input_data(
        self, application: CreditApplication
    ) -> Dict[str, Any]:
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
        hash_obj = hashlib.sha256(data_str.encode())
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
        self,
        request: PredictionRequest,
        response: PredictionResponse,
        principal: "Principal",
    ):
        """Log prediction for audit and monitoring."""

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "prediction_id": response.prediction_id,
            "principal": principal.identifier,
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

        # Durably persist the assessment (off the response path).
        if self.db_available:
            try:
                from fastapi.concurrency import run_in_threadpool

                from app.db import repository

                record = {
                    "prediction_id": response.prediction_id,
                    "user_id": principal.user_id,
                    "risk_score": float(response.risk_score),
                    "risk_level": response.risk_level.value,
                    "confidence": float(response.confidence),
                    "processing_time_ms": float(response.processing_time_ms),
                    "model_version": response.model_version,
                    "api_key_prefix": principal.identifier[:20],
                    "application": request.application.model_dump(),
                    "explanation": response.explanation,
                    "sustainability": response.sustainability_metrics,
                }
                await run_in_threadpool(repository.save_prediction, record)
            except Exception as exc:
                logger.error(f"Failed to persist prediction: {exc}")

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
            "successful_predictions": response.batch_summary[
                "successful_predictions"
            ],
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
            self.app,
            host=host,
            port=port,
            log_level=self.config.log_level.lower(),
        )

    def get_app(self):
        """Get the FastAPI app instance."""
        return self.app


# Utility functions


def create_inference_service(
    config: Optional[APIConfig] = None,
) -> InferenceService:
    """Create inference service instance."""
    return InferenceService(config)


def run_inference_service(
    host: str = "0.0.0.0", port: int = 8000, config: Optional[APIConfig] = None
):
    """Run inference service."""
    service = create_inference_service(config)
    service.run(host, port)


def asgi_app():
    """ASGI factory for production servers.

    Use with: ``uvicorn app.api.inference_service:asgi_app --factory``.
    Defined as a factory (not a module-level instance) so importing this
    module never constructs the service or touches the database.
    """
    return create_inference_service().get_app()


if __name__ == "__main__":
    # Run service with default configuration
    if FASTAPI_AVAILABLE:
        run_inference_service()
    else:
        print(
            "FastAPI not available. Install with: pip install fastapi uvicorn"
        )
