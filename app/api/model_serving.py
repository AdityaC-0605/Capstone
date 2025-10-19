"""
Model Serving Infrastructure for Credit Risk Prediction.

This module implements comprehensive model serving capabilities including
model loading, caching, version management, A/B testing, health checks,
circuit breakers, and multi-model routing.
"""

import asyncio
import hashlib
import json
import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Circuit breaker and caching dependencies
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Install with: pip install redis")

try:
    from ..core.logging import get_audit_logger, get_logger
    from ..models.ensemble_model import EnsembleModel
    from .inference_service import CreditApplication, PredictionResponse
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_audit_logger, get_logger

    # Mock classes for testing
    class MockEnsembleModel:
        def __init__(self, model_id: str = "mock_model", version: str = "1.0.0"):
            self.model_id = model_id
            self.version = version
            self.loaded_at = datetime.now()

        def predict(self, data):
            return {"prediction": 0.5, "confidence": 0.8}

        def predict_proba(self, data):
            return [[0.5, 0.5]]

        def get_model_info(self):
            return {
                "model_id": self.model_id,
                "version": self.version,
                "loaded_at": self.loaded_at.isoformat(),
                "type": "ensemble",
            }

    class MockCreditApplication:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockPredictionResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    EnsembleModel = MockEnsembleModel
    CreditApplication = MockCreditApplication
    PredictionResponse = MockPredictionResponse

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class ModelStatus(Enum):
    """Model status types."""

    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADING = "unloading"
    DEPRECATED = "deprecated"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class RoutingStrategy(Enum):
    """Model routing strategies."""

    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    A_B_TEST = "a_b_test"
    CANARY = "canary"
    CHAMPION_CHALLENGER = "champion_challenger"


@dataclass
class ModelMetadata:
    """Model metadata container."""

    model_id: str
    version: str
    model_type: str
    file_path: Optional[str] = None

    # Performance metrics
    accuracy: Optional[float] = None
    latency_ms: Optional[float] = None
    memory_mb: Optional[float] = None

    # Deployment info
    loaded_at: Optional[datetime] = None
    status: ModelStatus = ModelStatus.LOADING

    # A/B testing
    traffic_percentage: float = 0.0
    is_champion: bool = False
    is_challenger: bool = False

    # Health metrics
    request_count: int = 0
    error_count: int = 0
    last_used: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "model_type": self.model_type,
            "file_path": self.file_path,
            "accuracy": self.accuracy,
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "status": self.status.value,
            "traffic_percentage": self.traffic_percentage,
            "is_champion": self.is_champion,
            "is_challenger": self.is_challenger,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "error_rate": self.error_count / max(self.request_count, 1),
        }


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    request_timeout: int = 30  # seconds
    half_open_max_calls: int = 3


@dataclass
class CacheConfig:
    """Cache configuration."""

    enable_caching: bool = True
    cache_ttl: int = 300  # seconds
    max_cache_size: int = 10000
    cache_backend: str = "memory"  # "memory" or "redis"
    redis_url: Optional[str] = None


@dataclass
class ModelServingConfig:
    """Model serving configuration."""

    # Model storage
    model_storage_dir: str = "models"
    model_registry_file: str = "model_registry.json"

    # Caching
    cache_config: CacheConfig = field(default_factory=CacheConfig)

    # Circuit breaker
    circuit_breaker_config: CircuitBreakerConfig = field(
        default_factory=CircuitBreakerConfig
    )

    # Health checks
    health_check_interval: int = 30  # seconds
    readiness_check_timeout: int = 5  # seconds

    # A/B testing
    enable_ab_testing: bool = True
    default_routing_strategy: RoutingStrategy = RoutingStrategy.WEIGHTED

    # Performance
    max_concurrent_requests: int = 100
    request_queue_size: int = 1000

    # Model management
    auto_unload_unused_models: bool = True
    unused_model_threshold_hours: int = 24


class CircuitBreaker:
    """Circuit breaker implementation for model resilience."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""

        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise Exception("Circuit breaker is OPEN")

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise Exception(
                        "Circuit breaker is HALF_OPEN with max calls reached"
                    )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_calls += 1
                if self.half_open_calls >= self.config.half_open_max_calls:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
            elif (
                self.state == CircuitBreakerState.CLOSED
                and self.failure_count >= self.config.failure_threshold
            ):
                self.state = CircuitBreakerState.OPEN

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "half_open_calls": self.half_open_calls,
        }


class PredictionCache:
    """Prediction caching system."""

    def __init__(self, config: CacheConfig):
        self.config = config

        if config.cache_backend == "redis" and REDIS_AVAILABLE:
            self.cache = redis.from_url(config.redis_url or "redis://localhost:6379")
            self.backend = "redis"
        else:
            self.cache = {}
            self.cache_timestamps = {}
            self.backend = "memory"

        logger.info(f"Prediction cache initialized with {self.backend} backend")

    def get(self, key: str) -> Optional[Any]:
        """Get cached prediction."""
        if not self.config.enable_caching:
            return None

        try:
            if self.backend == "redis":
                cached_data = self.cache.get(key)
                if cached_data:
                    return json.loads(cached_data)
            else:
                if key in self.cache:
                    # Check TTL
                    if key in self.cache_timestamps:
                        age = time.time() - self.cache_timestamps[key]
                        if age > self.config.cache_ttl:
                            del self.cache[key]
                            del self.cache_timestamps[key]
                            return None
                    return self.cache[key]
        except Exception as e:
            logger.warning(f"Cache get error: {e}")

        return None

    def set(self, key: str, value: Any) -> None:
        """Set cached prediction."""
        if not self.config.enable_caching:
            return

        try:
            if self.backend == "redis":
                self.cache.setex(
                    key, self.config.cache_ttl, json.dumps(value, default=str)
                )
            else:
                # Memory cache size management
                if len(self.cache) >= self.config.max_cache_size:
                    # Remove oldest entries
                    oldest_keys = sorted(
                        self.cache_timestamps.items(), key=lambda x: x[1]
                    )[:10]
                    for old_key, _ in oldest_keys:
                        del self.cache[old_key]
                        del self.cache_timestamps[old_key]

                self.cache[key] = value
                self.cache_timestamps[key] = time.time()
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def generate_key(self, model_id: str, input_data: Dict[str, Any]) -> str:
        """Generate cache key from model ID and input data."""
        # Create deterministic hash from input data
        input_str = json.dumps(input_data, sort_keys=True)
        input_hash = hashlib.md5(input_str.encode()).hexdigest()
        return f"pred:{model_id}:{input_hash}"

    def clear(self) -> None:
        """Clear all cached predictions."""
        try:
            if self.backend == "redis":
                self.cache.flushdb()
            else:
                self.cache.clear()
                self.cache_timestamps.clear()
            logger.info("Prediction cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if self.backend == "redis":
                info = self.cache.info()
                return {
                    "backend": "redis",
                    "keys": info.get("db0", {}).get("keys", 0),
                    "memory_usage": info.get("used_memory", 0),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0),
                }
            else:
                return {
                    "backend": "memory",
                    "keys": len(self.cache),
                    "memory_usage": sys.getsizeof(self.cache),
                    "hits": 0,  # Not tracked in memory backend
                    "misses": 0,
                }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"error": str(e)}


class ModelLoader:
    """Model loading and management system."""

    def __init__(self, config: ModelServingConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.Lock()

        # Create model storage directory
        Path(config.model_storage_dir).mkdir(parents=True, exist_ok=True)

        # Load model registry
        self._load_model_registry()

        logger.info("Model loader initialized")

    def load_model(
        self, model_id: str, version: str, file_path: Optional[str] = None
    ) -> bool:
        """Load a model into memory."""

        with self.lock:
            full_model_id = f"{model_id}:{version}"

            if full_model_id in self.models:
                logger.info(f"Model {full_model_id} already loaded")
                return True

            try:
                # Update status to loading
                metadata = ModelMetadata(
                    model_id=model_id,
                    version=version,
                    model_type="ensemble",
                    file_path=file_path,
                    status=ModelStatus.LOADING,
                )
                self.metadata[full_model_id] = metadata

                # Load model (mock implementation)
                model = EnsembleModel(model_id=model_id, version=version)

                # Update metadata
                metadata.loaded_at = datetime.now()
                metadata.status = ModelStatus.READY

                # Store model and create circuit breaker
                self.models[full_model_id] = model
                self.circuit_breakers[full_model_id] = CircuitBreaker(
                    self.config.circuit_breaker_config
                )

                logger.info(f"Model {full_model_id} loaded successfully")

                # Save to registry
                self._save_model_registry()

                return True

            except Exception as e:
                logger.error(f"Failed to load model {full_model_id}: {e}")
                if full_model_id in self.metadata:
                    self.metadata[full_model_id].status = ModelStatus.ERROR
                return False

    def unload_model(self, model_id: str, version: str) -> bool:
        """Unload a model from memory."""

        with self.lock:
            full_model_id = f"{model_id}:{version}"

            if full_model_id not in self.models:
                logger.warning(f"Model {full_model_id} not loaded")
                return False

            try:
                # Update status
                if full_model_id in self.metadata:
                    self.metadata[full_model_id].status = ModelStatus.UNLOADING

                # Remove model and circuit breaker
                del self.models[full_model_id]
                if full_model_id in self.circuit_breakers:
                    del self.circuit_breakers[full_model_id]

                # Remove metadata
                if full_model_id in self.metadata:
                    del self.metadata[full_model_id]

                logger.info(f"Model {full_model_id} unloaded successfully")

                # Save to registry
                self._save_model_registry()

                return True

            except Exception as e:
                logger.error(f"Failed to unload model {full_model_id}: {e}")
                return False

    def get_model(self, model_id: str, version: str) -> Optional[Any]:
        """Get a loaded model."""
        full_model_id = f"{model_id}:{version}"
        return self.models.get(full_model_id)

    def get_model_metadata(
        self, model_id: str, version: str
    ) -> Optional[ModelMetadata]:
        """Get model metadata."""
        full_model_id = f"{model_id}:{version}"
        return self.metadata.get(full_model_id)

    def list_models(self) -> List[ModelMetadata]:
        """List all loaded models."""
        return list(self.metadata.values())

    def predict_with_circuit_breaker(
        self, model_id: str, version: str, input_data: Dict[str, Any]
    ) -> Any:
        """Make prediction with circuit breaker protection."""
        full_model_id = f"{model_id}:{version}"

        model = self.models.get(full_model_id)
        if not model:
            raise ValueError(f"Model {full_model_id} not loaded")

        circuit_breaker = self.circuit_breakers.get(full_model_id)
        if not circuit_breaker:
            raise ValueError(f"Circuit breaker not found for {full_model_id}")

        # Update metadata
        if full_model_id in self.metadata:
            self.metadata[full_model_id].request_count += 1
            self.metadata[full_model_id].last_used = datetime.now()

        try:
            result = circuit_breaker.call(model.predict, input_data)
            return result
        except Exception as e:
            # Update error count
            if full_model_id in self.metadata:
                self.metadata[full_model_id].error_count += 1
            raise e

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all models."""
        health_status = {
            "total_models": len(self.models),
            "ready_models": 0,
            "error_models": 0,
            "models": {},
        }

        for full_model_id, metadata in self.metadata.items():
            model_health = {
                "status": metadata.status.value,
                "error_rate": metadata.error_count / max(metadata.request_count, 1),
                "last_used": (
                    metadata.last_used.isoformat() if metadata.last_used else None
                ),
                "circuit_breaker": (
                    self.circuit_breakers[full_model_id].get_state()
                    if full_model_id in self.circuit_breakers
                    else None
                ),
            }

            health_status["models"][full_model_id] = model_health

            if metadata.status == ModelStatus.READY:
                health_status["ready_models"] += 1
            elif metadata.status == ModelStatus.ERROR:
                health_status["error_models"] += 1

        return health_status

    def cleanup_unused_models(self) -> List[str]:
        """Clean up unused models based on configuration."""
        if not self.config.auto_unload_unused_models:
            return []

        threshold_time = datetime.now() - timedelta(
            hours=self.config.unused_model_threshold_hours
        )
        unloaded_models = []

        for full_model_id, metadata in list(self.metadata.items()):
            if (
                metadata.last_used
                and metadata.last_used < threshold_time
                and not metadata.is_champion
            ):
                model_id, version = full_model_id.split(":", 1)
                if self.unload_model(model_id, version):
                    unloaded_models.append(full_model_id)

        if unloaded_models:
            logger.info(f"Cleaned up unused models: {unloaded_models}")

        return unloaded_models

    def _load_model_registry(self):
        """Load model registry from file."""
        registry_file = Path(self.config.model_registry_file)

        if not registry_file.exists():
            logger.info("Model registry file not found, starting with empty registry")
            return

        try:
            with open(registry_file, "r") as f:
                registry_data = json.load(f)

            for model_data in registry_data.get("models", []):
                full_model_id = f"{model_data['model_id']}:{model_data['version']}"

                metadata = ModelMetadata(
                    model_id=model_data["model_id"],
                    version=model_data["version"],
                    model_type=model_data.get("model_type", "ensemble"),
                    file_path=model_data.get("file_path"),
                    accuracy=model_data.get("accuracy"),
                    latency_ms=model_data.get("latency_ms"),
                    memory_mb=model_data.get("memory_mb"),
                    traffic_percentage=model_data.get("traffic_percentage", 0.0),
                    is_champion=model_data.get("is_champion", False),
                    is_challenger=model_data.get("is_challenger", False),
                    status=ModelStatus.READY,  # Will be updated when actually loaded
                )

                if model_data.get("loaded_at"):
                    metadata.loaded_at = datetime.fromisoformat(model_data["loaded_at"])

                self.metadata[full_model_id] = metadata

            logger.info(
                f"Loaded {len(registry_data.get('models', []))} models from registry"
            )

        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")

    def _save_model_registry(self):
        """Save model registry to file."""
        try:
            registry_data = {
                "updated_at": datetime.now().isoformat(),
                "models": [metadata.to_dict() for metadata in self.metadata.values()],
            }

            with open(self.config.model_registry_file, "w") as f:
                json.dump(registry_data, f, indent=2)

            logger.debug("Model registry saved")

        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")


class ModelRouter:
    """Model routing system for A/B testing and traffic management."""

    def __init__(self, config: ModelServingConfig):
        self.config = config
        self.routing_strategy = config.default_routing_strategy
        self.model_weights: Dict[str, float] = {}
        self.ab_test_config: Dict[str, Any] = {}
        self.request_count = 0
        self.lock = threading.Lock()

        logger.info(
            f"Model router initialized with {self.routing_strategy.value} strategy"
        )

    def route_request(
        self, available_models: List[str], user_id: Optional[str] = None
    ) -> str:
        """Route request to appropriate model based on strategy."""

        if not available_models:
            raise ValueError("No available models for routing")

        if len(available_models) == 1:
            return available_models[0]

        with self.lock:
            self.request_count += 1

            if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
                return self._round_robin_routing(available_models)
            elif self.routing_strategy == RoutingStrategy.WEIGHTED:
                return self._weighted_routing(available_models)
            elif self.routing_strategy == RoutingStrategy.A_B_TEST:
                return self._ab_test_routing(available_models, user_id)
            elif self.routing_strategy == RoutingStrategy.CANARY:
                return self._canary_routing(available_models)
            elif self.routing_strategy == RoutingStrategy.CHAMPION_CHALLENGER:
                return self._champion_challenger_routing(available_models)
            else:
                # Default to first available model
                return available_models[0]

    def _round_robin_routing(self, models: List[str]) -> str:
        """Round-robin routing strategy."""
        index = self.request_count % len(models)
        return models[index]

    def _weighted_routing(self, models: List[str]) -> str:
        """Weighted routing based on model weights."""
        import random

        # Get weights for models
        weights = []
        for model in models:
            weight = self.model_weights.get(model, 1.0)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return models[0]

        normalized_weights = [w / total_weight for w in weights]

        # Weighted random selection
        rand = random.random()
        cumulative = 0
        for i, weight in enumerate(normalized_weights):
            cumulative += weight
            if rand <= cumulative:
                return models[i]

        return models[-1]  # Fallback

    def _ab_test_routing(self, models: List[str], user_id: Optional[str]) -> str:
        """A/B test routing based on user ID hash."""
        if not user_id:
            # Fallback to weighted routing
            return self._weighted_routing(models)

        # Hash user ID to get consistent routing
        import hashlib

        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)

        # Get A/B test configuration
        test_config = self.ab_test_config.get("current_test", {})
        if not test_config:
            return models[0]

        # Route based on hash and test configuration
        bucket = user_hash % 100

        for model, percentage in test_config.items():
            if model in models and bucket < percentage:
                return model
            bucket -= percentage

        return models[0]  # Fallback

    def _canary_routing(self, models: List[str]) -> str:
        """Canary deployment routing."""
        # Find canary model (challenger)
        canary_model = None
        stable_model = None

        for model in models:
            if "challenger" in model.lower() or "canary" in model.lower():
                canary_model = model
            else:
                stable_model = model

        if not canary_model:
            return stable_model or models[0]

        # Route small percentage to canary
        canary_percentage = 10  # 10% to canary
        if (self.request_count % 100) < canary_percentage:
            return canary_model
        else:
            return stable_model or models[0]

    def _champion_challenger_routing(self, models: List[str]) -> str:
        """Champion-challenger routing strategy."""
        champion = None
        challenger = None

        for model in models:
            if "champion" in model.lower():
                champion = model
            elif "challenger" in model.lower():
                challenger = model

        if not champion:
            champion = models[0]

        if not challenger:
            return champion

        # Route 80% to champion, 20% to challenger
        if (self.request_count % 100) < 80:
            return champion
        else:
            return challenger

    def set_routing_strategy(self, strategy: RoutingStrategy):
        """Set routing strategy."""
        self.routing_strategy = strategy
        logger.info(f"Routing strategy changed to {strategy.value}")

    def set_model_weights(self, weights: Dict[str, float]):
        """Set model weights for weighted routing."""
        self.model_weights = weights.copy()
        logger.info(f"Model weights updated: {weights}")

    def configure_ab_test(self, test_config: Dict[str, float]):
        """Configure A/B test parameters."""
        self.ab_test_config["current_test"] = test_config
        logger.info(f"A/B test configured: {test_config}")

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "strategy": self.routing_strategy.value,
            "total_requests": self.request_count,
            "model_weights": self.model_weights,
            "ab_test_config": self.ab_test_config,
        }


class ModelServingManager:
    """Main model serving management system."""

    def __init__(self, config: Optional[ModelServingConfig] = None):
        self.config = config or ModelServingConfig()

        # Initialize components
        self.model_loader = ModelLoader(self.config)
        self.model_router = ModelRouter(self.config)
        self.prediction_cache = PredictionCache(self.config.cache_config)

        # Health monitoring
        self.health_check_thread = None
        self.is_healthy = True
        self.last_health_check = datetime.now()

        # Request tracking
        self.request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        logger.info("Model serving manager initialized")

    async def predict(
        self,
        input_data: Dict[str, Any],
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make prediction with full serving infrastructure."""

        async with self.request_semaphore:
            start_time = time.time()

            try:
                # Get available models
                available_models = self._get_available_models(model_id, version)
                if not available_models:
                    raise ValueError("No available models for prediction")

                # Route request to appropriate model
                selected_model = self.model_router.route_request(
                    available_models, user_id
                )
                selected_model_id, selected_version = selected_model.split(":", 1)

                # Check cache first
                cache_key = self.prediction_cache.generate_key(
                    selected_model, input_data
                )
                cached_result = self.prediction_cache.get(cache_key)

                if cached_result:
                    cached_result["cached"] = True
                    cached_result["processing_time_ms"] = (
                        time.time() - start_time
                    ) * 1000
                    return cached_result

                # Make prediction with circuit breaker
                prediction_result = self.model_loader.predict_with_circuit_breaker(
                    selected_model_id, selected_version, input_data
                )

                # Format response
                response = {
                    "prediction": prediction_result.get("prediction", 0.5),
                    "confidence": prediction_result.get("confidence", 0.8),
                    "model_id": selected_model_id,
                    "model_version": selected_version,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "cached": False,
                    "timestamp": datetime.now().isoformat(),
                }

                # Cache result
                self.prediction_cache.set(cache_key, response)

                # Audit log
                audit_logger.log_model_operation(
                    user_id=user_id or "anonymous",
                    model_id=selected_model,
                    operation="predict",
                    success=True,
                    details={
                        "processing_time_ms": response["processing_time_ms"],
                        "cached": False,
                    },
                )

                return response

            except Exception as e:
                logger.error(f"Prediction error: {e}")

                # Audit log error
                audit_logger.log_model_operation(
                    user_id=user_id or "anonymous",
                    model_id=model_id or "unknown",
                    operation="predict",
                    success=False,
                    details={"error": str(e)},
                )

                raise e

    def _get_available_models(
        self, model_id: Optional[str] = None, version: Optional[str] = None
    ) -> List[str]:
        """Get list of available models for prediction."""
        available = []

        for metadata in self.model_loader.list_models():
            if metadata.status != ModelStatus.READY:
                continue

            full_model_id = f"{metadata.model_id}:{metadata.version}"

            # Filter by model_id and version if specified
            if model_id and metadata.model_id != model_id:
                continue
            if version and metadata.version != version:
                continue

            available.append(full_model_id)

        return available

    def load_model(
        self,
        model_id: str,
        version: str,
        file_path: Optional[str] = None,
        is_champion: bool = False,
        is_challenger: bool = False,
        traffic_percentage: float = 0.0,
    ) -> bool:
        """Load model with serving configuration."""

        success = self.model_loader.load_model(model_id, version, file_path)

        if success:
            # Update metadata for serving
            metadata = self.model_loader.get_model_metadata(model_id, version)
            if metadata:
                metadata.is_champion = is_champion
                metadata.is_challenger = is_challenger
                metadata.traffic_percentage = traffic_percentage

        return success

    def unload_model(self, model_id: str, version: str) -> bool:
        """Unload model from serving."""
        return self.model_loader.unload_model(model_id, version)

    def update_model_weights(self, weights: Dict[str, float]):
        """Update model routing weights."""
        self.model_router.set_model_weights(weights)

    def configure_ab_test(self, test_config: Dict[str, float]):
        """Configure A/B test."""
        self.model_router.configure_ab_test(test_config)

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        model_health = self.model_loader.get_health_status()
        cache_stats = self.prediction_cache.get_stats()
        routing_stats = self.model_router.get_routing_stats()

        return {
            "overall_health": "healthy" if self.is_healthy else "unhealthy",
            "last_health_check": self.last_health_check.isoformat(),
            "models": model_health,
            "cache": cache_stats,
            "routing": routing_stats,
            "concurrent_requests": self.config.max_concurrent_requests
            - self.request_semaphore._value,
        }

    def get_readiness_status(self) -> Dict[str, Any]:
        """Get readiness status for load balancer."""
        ready_models = sum(
            1 for m in self.model_loader.list_models() if m.status == ModelStatus.READY
        )

        is_ready = ready_models > 0 and self.is_healthy

        return {
            "ready": is_ready,
            "ready_models": ready_models,
            "total_models": len(self.model_loader.list_models()),
            "timestamp": datetime.now().isoformat(),
        }

    def start_health_monitoring(self):
        """Start background health monitoring."""

        def health_check_loop():
            while True:
                try:
                    # Check model health
                    health_status = self.model_loader.get_health_status()

                    # Update overall health
                    self.is_healthy = (
                        health_status["error_models"] == 0
                        and health_status["ready_models"] > 0
                    )

                    self.last_health_check = datetime.now()

                    # Cleanup unused models
                    self.model_loader.cleanup_unused_models()

                    time.sleep(self.config.health_check_interval)

                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    self.is_healthy = False
                    time.sleep(self.config.health_check_interval)

        self.health_check_thread = threading.Thread(
            target=health_check_loop, daemon=True
        )
        self.health_check_thread.start()

        logger.info("Health monitoring started")

    def stop_health_monitoring(self):
        """Stop health monitoring."""
        # In a real implementation, you'd use a proper shutdown mechanism
        logger.info("Health monitoring stopped")


# Utility functions


def create_model_serving_manager(
    config: Optional[ModelServingConfig] = None,
) -> ModelServingManager:
    """Create model serving manager."""
    return ModelServingManager(config)


async def serve_prediction(
    input_data: Dict[str, Any], manager: Optional[ModelServingManager] = None, **kwargs
) -> Dict[str, Any]:
    """Serve prediction with full infrastructure."""
    if manager is None:
        manager = create_model_serving_manager()

    return await manager.predict(input_data, **kwargs)
