"""
Performance Monitoring and Resilience System for Credit Risk API.

This module implements comprehensive performance monitoring, SLA tracking,
model drift detection, alerting, and resilience features for production APIs.
"""

import asyncio
import json
import statistics
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Install with: pip install numpy")

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Install with: pip install scipy")

try:
    from ..core.logging import get_audit_logger, get_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_audit_logger, get_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to track."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUEUE_SIZE = "queue_size"
    DRIFT_SCORE = "drift_score"
    SLA_VIOLATION = "sla_violation"


class DriftDetectionMethod(Enum):
    """Model drift detection methods."""

    KS_TEST = "ks_test"
    PSI = "psi"  # Population Stability Index
    WASSERSTEIN = "wasserstein"
    JENSEN_SHANNON = "jensen_shannon"


@dataclass
class SLAConfig:
    """Service Level Agreement configuration."""

    max_latency_ms: float = 100.0
    max_error_rate: float = 0.05  # 5%
    min_availability: float = 0.99  # 99%
    min_throughput_rps: float = 10.0  # requests per second
    measurement_window_minutes: int = 5


@dataclass
class AlertConfig:
    """Alert configuration."""

    enabled: bool = True
    email_recipients: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None
    slack_webhook: Optional[str] = None
    cooldown_minutes: int = 15  # Minimum time between similar alerts


@dataclass
class ThrottleConfig:
    """Request throttling configuration."""

    enabled: bool = True
    max_requests_per_second: int = 100
    max_requests_per_minute: int = 1000
    max_requests_per_hour: int = 10000
    burst_allowance: int = 20
    queue_size_limit: int = 1000


@dataclass
class RetryConfig:
    """Retry mechanism configuration."""

    enabled: bool = True
    max_retries: int = 3
    initial_delay_ms: int = 100
    max_delay_ms: int = 5000
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class PerformanceMetric:
    """Performance metric data point."""

    timestamp: datetime
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert data structure."""

    id: str
    level: AlertLevel
    message: str
    metric_type: MetricType
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """Collects and stores performance metrics."""

    def __init__(self, max_history_hours: int = 24):
        self.max_history_hours = max_history_hours
        self.metrics: Dict[MetricType, deque] = {
            metric_type: deque() for metric_type in MetricType
        }
        self.lock = threading.Lock()

    def record_metric(
        self, metric_type: MetricType, value: float, metadata: Dict[str, Any] = None
    ):
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            metadata=metadata or {},
        )

        with self.lock:
            self.metrics[metric_type].append(metric)
            self._cleanup_old_metrics(metric_type)

    def _cleanup_old_metrics(self, metric_type: MetricType):
        """Remove metrics older than max_history_hours."""
        cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)

        while (
            self.metrics[metric_type]
            and self.metrics[metric_type][0].timestamp < cutoff_time
        ):
            self.metrics[metric_type].popleft()

    def get_metrics(
        self, metric_type: MetricType, since: Optional[datetime] = None
    ) -> List[PerformanceMetric]:
        """Get metrics of a specific type since a given time."""
        with self.lock:
            metrics = list(self.metrics[metric_type])

        if since:
            metrics = [m for m in metrics if m.timestamp >= since]

        return metrics

    def get_metric_summary(
        self, metric_type: MetricType, window_minutes: int = 5
    ) -> Dict[str, float]:
        """Get summary statistics for a metric type."""
        since = datetime.now() - timedelta(minutes=window_minutes)
        metrics = self.get_metrics(metric_type, since)

        if not metrics:
            return {
                "count": 0,
                "mean": 0,
                "median": 0,
                "p95": 0,
                "p99": 0,
                "min": 0,
                "max": 0,
            }

        values = [m.value for m in metrics]

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": (
                np.percentile(values, 95)
                if NUMPY_AVAILABLE
                else sorted(values)[int(0.95 * len(values))]
            ),
            "p99": (
                np.percentile(values, 99)
                if NUMPY_AVAILABLE
                else sorted(values)[int(0.99 * len(values))]
            ),
            "min": min(values),
            "max": max(values),
        }


class DriftDetector:
    """Detects model drift in input data and predictions."""

    def __init__(
        self, reference_window_size: int = 1000, detection_window_size: int = 100
    ):
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.reference_data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=reference_window_size)
        )
        self.current_data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=detection_window_size)
        )
        self.lock = threading.Lock()

    def add_reference_data(self, feature_name: str, value: float):
        """Add data point to reference distribution."""
        with self.lock:
            self.reference_data[feature_name].append(value)

    def add_current_data(self, feature_name: str, value: float):
        """Add data point to current distribution."""
        with self.lock:
            self.current_data[feature_name].append(value)

    def detect_drift(
        self,
        feature_name: str,
        method: DriftDetectionMethod = DriftDetectionMethod.KS_TEST,
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Detect drift for a specific feature."""
        with self.lock:
            reference = list(self.reference_data[feature_name])
            current = list(self.current_data[feature_name])

        if len(reference) < 50 or len(current) < 20:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "p_value": 1.0,
                "method": method.value,
                "message": "Insufficient data for drift detection",
            }

        try:
            if method == DriftDetectionMethod.KS_TEST:
                return self._ks_test_drift(reference, current, threshold)
            elif method == DriftDetectionMethod.PSI:
                return self._psi_drift(reference, current, threshold)
            elif method == DriftDetectionMethod.WASSERSTEIN:
                return self._wasserstein_drift(reference, current, threshold)
            elif method == DriftDetectionMethod.JENSEN_SHANNON:
                return self._jensen_shannon_drift(reference, current, threshold)
            else:
                raise ValueError(f"Unknown drift detection method: {method}")

        except Exception as e:
            logger.error(f"Drift detection error for {feature_name}: {e}")
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "p_value": 1.0,
                "method": method.value,
                "error": str(e),
            }

    def _ks_test_drift(
        self, reference: List[float], current: List[float], threshold: float
    ) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test for drift detection."""
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for KS test")

        statistic, p_value = stats.ks_2samp(reference, current)

        return {
            "drift_detected": p_value < threshold,
            "drift_score": statistic,
            "p_value": p_value,
            "method": "ks_test",
            "threshold": threshold,
        }

    def _psi_drift(
        self, reference: List[float], current: List[float], threshold: float
    ) -> Dict[str, Any]:
        """Population Stability Index for drift detection."""
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy required for PSI calculation")

        # Create bins based on reference data
        bins = np.histogram_bin_edges(reference, bins=10)

        # Calculate histograms
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)

        # Convert to proportions
        ref_prop = ref_hist / len(reference)
        cur_prop = cur_hist / len(current)

        # Avoid division by zero
        ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
        cur_prop = np.where(cur_prop == 0, 0.0001, cur_prop)

        # Calculate PSI
        psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))

        return {
            "drift_detected": psi > threshold,
            "drift_score": float(psi),
            "p_value": None,
            "method": "psi",
            "threshold": threshold,
        }

    def _wasserstein_drift(
        self, reference: List[float], current: List[float], threshold: float
    ) -> Dict[str, Any]:
        """Wasserstein distance for drift detection."""
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for Wasserstein distance")

        distance = stats.wasserstein_distance(reference, current)

        return {
            "drift_detected": distance > threshold,
            "drift_score": distance,
            "p_value": None,
            "method": "wasserstein",
            "threshold": threshold,
        }

    def _jensen_shannon_drift(
        self, reference: List[float], current: List[float], threshold: float
    ) -> Dict[str, Any]:
        """Jensen-Shannon divergence for drift detection."""
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy required for Jensen-Shannon divergence")

        # Create histograms
        bins = np.histogram_bin_edges(reference + current, bins=20)
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)

        # Convert to probabilities
        ref_prob = ref_hist / np.sum(ref_hist)
        cur_prob = cur_hist / np.sum(cur_hist)

        # Avoid log(0)
        ref_prob = np.where(ref_prob == 0, 1e-10, ref_prob)
        cur_prob = np.where(cur_prob == 0, 1e-10, cur_prob)

        # Calculate Jensen-Shannon divergence
        m = 0.5 * (ref_prob + cur_prob)
        js_div = 0.5 * np.sum(ref_prob * np.log(ref_prob / m)) + 0.5 * np.sum(
            cur_prob * np.log(cur_prob / m)
        )

        return {
            "drift_detected": js_div > threshold,
            "drift_score": float(js_div),
            "p_value": None,
            "method": "jensen_shannon",
            "threshold": threshold,
        }


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self, config: AlertConfig):
        self.config = config
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self.lock = threading.Lock()

    def create_alert(
        self,
        level: AlertLevel,
        message: str,
        metric_type: MetricType,
        value: float,
        threshold: float,
    ) -> Optional[Alert]:
        """Create and send an alert if conditions are met."""

        # Check cooldown period
        alert_key = f"{level.value}_{metric_type.value}"
        if self._is_in_cooldown(alert_key):
            return None

        alert_id = f"{alert_key}_{int(time.time())}"
        alert = Alert(
            id=alert_id,
            level=level,
            message=message,
            metric_type=metric_type,
            value=value,
            threshold=threshold,
            timestamp=datetime.now(),
        )

        with self.lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.last_alert_times[alert_key] = datetime.now()

        # Send alert
        if self.config.enabled:
            self._send_alert(alert)

        logger.warning(f"Alert created: {alert.message}")

        return alert

    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                del self.active_alerts[alert_id]

                logger.info(f"Alert resolved: {alert.message}")

    def _is_in_cooldown(self, alert_key: str) -> bool:
        """Check if alert is in cooldown period."""
        if alert_key not in self.last_alert_times:
            return False

        last_time = self.last_alert_times[alert_key]
        cooldown_period = timedelta(minutes=self.config.cooldown_minutes)

        return datetime.now() - last_time < cooldown_period

    def _send_alert(self, alert: Alert):
        """Send alert through configured channels."""
        try:
            # Log alert
            audit_logger.log_model_operation(
                user_id="system",
                model_id="performance_monitor",
                operation="alert",
                success=True,
                details={
                    "alert_id": alert.id,
                    "level": alert.level.value,
                    "metric_type": alert.metric_type.value,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "message": alert.message,
                },
            )

            # Here you would implement actual alert sending:
            # - Email notifications
            # - Slack/Teams webhooks
            # - PagerDuty integration
            # - Custom webhook calls

            logger.info(f"Alert sent: {alert.level.value} - {alert.message}")

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self.lock:
            return [
                alert for alert in self.alert_history if alert.timestamp >= cutoff_time
            ]


class RequestThrottler:
    """Implements request throttling and queue management."""

    def __init__(self, config: ThrottleConfig):
        self.config = config
        self.request_counts = {"second": deque(), "minute": deque(), "hour": deque()}
        self.request_queue = asyncio.Queue(maxsize=config.queue_size_limit)
        self.lock = threading.Lock()

    async def check_rate_limit(self, client_id: str = "default") -> bool:
        """Check if request is within rate limits."""
        if not self.config.enabled:
            return True

        current_time = time.time()

        with self.lock:
            # Clean old entries
            self._cleanup_old_requests(current_time)

            # Check limits
            if (
                len(self.request_counts["second"])
                >= self.config.max_requests_per_second
                or len(self.request_counts["minute"])
                >= self.config.max_requests_per_minute
                or len(self.request_counts["hour"]) >= self.config.max_requests_per_hour
            ):
                return False

            # Record request
            self.request_counts["second"].append(current_time)
            self.request_counts["minute"].append(current_time)
            self.request_counts["hour"].append(current_time)

        return True

    def _cleanup_old_requests(self, current_time: float):
        """Remove old request timestamps."""
        # Remove requests older than 1 second
        while (
            self.request_counts["second"]
            and current_time - self.request_counts["second"][0] > 1
        ):
            self.request_counts["second"].popleft()

        # Remove requests older than 1 minute
        while (
            self.request_counts["minute"]
            and current_time - self.request_counts["minute"][0] > 60
        ):
            self.request_counts["minute"].popleft()

        # Remove requests older than 1 hour
        while (
            self.request_counts["hour"]
            and current_time - self.request_counts["hour"][0] > 3600
        ):
            self.request_counts["hour"].popleft()

    async def enqueue_request(self, request_data: Any) -> bool:
        """Add request to processing queue."""
        try:
            await asyncio.wait_for(self.request_queue.put(request_data), timeout=1.0)
            return True
        except asyncio.TimeoutError:
            return False

    async def dequeue_request(self) -> Optional[Any]:
        """Get next request from queue."""
        try:
            return await self.request_queue.get()
        except asyncio.QueueEmpty:
            return None

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.request_queue.qsize()

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limiting status."""
        current_time = time.time()

        with self.lock:
            self._cleanup_old_requests(current_time)

            return {
                "requests_per_second": len(self.request_counts["second"]),
                "requests_per_minute": len(self.request_counts["minute"]),
                "requests_per_hour": len(self.request_counts["hour"]),
                "queue_size": self.get_queue_size(),
                "limits": {
                    "max_requests_per_second": self.config.max_requests_per_second,
                    "max_requests_per_minute": self.config.max_requests_per_minute,
                    "max_requests_per_hour": self.config.max_requests_per_hour,
                    "queue_size_limit": self.config.queue_size_limit,
                },
            }


class RetryManager:
    """Manages retry logic with exponential backoff."""

    def __init__(self, config: RetryConfig):
        self.config = config

    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        if not self.config.enabled:
            return await func(*args, **kwargs)

        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt == self.config.max_retries:
                    break

                # Calculate delay with exponential backoff
                delay = min(
                    self.config.initial_delay_ms
                    * (self.config.exponential_base**attempt),
                    self.config.max_delay_ms,
                )

                # Add jitter if enabled
                if self.config.jitter:
                    import random

                    delay *= 0.5 + random.random() * 0.5

                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.0f}ms: {e}"
                )
                await asyncio.sleep(delay / 1000)

        # All retries exhausted
        logger.error(f"All retry attempts exhausted. Last error: {last_exception}")
        raise last_exception


class PerformanceMonitor:
    """Main performance monitoring and resilience system."""

    def __init__(
        self,
        sla_config: Optional[SLAConfig] = None,
        alert_config: Optional[AlertConfig] = None,
        throttle_config: Optional[ThrottleConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.sla_config = sla_config or SLAConfig()
        self.alert_config = alert_config or AlertConfig()
        self.throttle_config = throttle_config or ThrottleConfig()
        self.retry_config = retry_config or RetryConfig()

        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.drift_detector = DriftDetector()
        self.alert_manager = AlertManager(self.alert_config)
        self.throttler = RequestThrottler(self.throttle_config)
        self.retry_manager = RetryManager(self.retry_config)

        # Fallback models
        self.fallback_models: List[str] = []
        self.current_fallback_index = 0

        # Background monitoring
        self.monitoring_active = True
        self._start_background_monitoring()

        logger.info("Performance monitor initialized")

    def _start_background_monitoring(self):
        """Start background monitoring tasks."""
        asyncio.create_task(self._sla_monitoring_loop())
        asyncio.create_task(self._drift_monitoring_loop())
        asyncio.create_task(self._health_monitoring_loop())

    async def _sla_monitoring_loop(self):
        """Background SLA monitoring loop."""
        while self.monitoring_active:
            try:
                await self._check_sla_violations()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"SLA monitoring error: {e}")
                await asyncio.sleep(60)

    async def _drift_monitoring_loop(self):
        """Background drift monitoring loop."""
        while self.monitoring_active:
            try:
                await self._check_model_drift()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Drift monitoring error: {e}")
                await asyncio.sleep(300)

    async def _health_monitoring_loop(self):
        """Background health monitoring loop."""
        while self.monitoring_active:
            try:
                await self._check_system_health()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)

    async def _check_sla_violations(self):
        """Check for SLA violations and create alerts."""
        window_minutes = self.sla_config.measurement_window_minutes

        # Check latency SLA
        latency_summary = self.metrics_collector.get_metric_summary(
            MetricType.LATENCY, window_minutes
        )

        if (
            latency_summary["count"] > 0
            and latency_summary["p95"] > self.sla_config.max_latency_ms
        ):
            self.alert_manager.create_alert(
                AlertLevel.WARNING,
                f"Latency SLA violation: P95 {latency_summary['p95']:.1f}ms > {self.sla_config.max_latency_ms}ms",
                MetricType.LATENCY,
                latency_summary["p95"],
                self.sla_config.max_latency_ms,
            )

        # Check error rate SLA
        error_summary = self.metrics_collector.get_metric_summary(
            MetricType.ERROR_RATE, window_minutes
        )

        if (
            error_summary["count"] > 0
            and error_summary["mean"] > self.sla_config.max_error_rate
        ):
            self.alert_manager.create_alert(
                AlertLevel.ERROR,
                f"Error rate SLA violation: {error_summary['mean']:.3f} > {self.sla_config.max_error_rate:.3f}",
                MetricType.ERROR_RATE,
                error_summary["mean"],
                self.sla_config.max_error_rate,
            )

        # Check throughput SLA
        throughput_summary = self.metrics_collector.get_metric_summary(
            MetricType.THROUGHPUT, window_minutes
        )

        if (
            throughput_summary["count"] > 0
            and throughput_summary["mean"] < self.sla_config.min_throughput_rps
        ):
            self.alert_manager.create_alert(
                AlertLevel.WARNING,
                f"Throughput SLA violation: {throughput_summary['mean']:.1f} RPS < {self.sla_config.min_throughput_rps} RPS",
                MetricType.THROUGHPUT,
                throughput_summary["mean"],
                self.sla_config.min_throughput_rps,
            )

    async def _check_model_drift(self):
        """Check for model drift and create alerts."""
        # This would typically check multiple features
        # For now, we'll implement a placeholder

        # Example: Check drift for a key feature
        drift_result = self.drift_detector.detect_drift(
            "credit_score", DriftDetectionMethod.KS_TEST, threshold=0.05
        )

        if drift_result["drift_detected"]:
            self.alert_manager.create_alert(
                AlertLevel.WARNING,
                f"Model drift detected for credit_score: {drift_result['method']} score {drift_result['drift_score']:.4f}",
                MetricType.DRIFT_SCORE,
                drift_result["drift_score"],
                0.05,
            )

    async def _check_system_health(self):
        """Check overall system health."""
        # Check queue sizes
        queue_size = self.throttler.get_queue_size()
        queue_limit = self.throttle_config.queue_size_limit

        if queue_size > queue_limit * 0.8:  # 80% of limit
            self.alert_manager.create_alert(
                AlertLevel.WARNING,
                f"Request queue near capacity: {queue_size}/{queue_limit}",
                MetricType.QUEUE_SIZE,
                queue_size,
                queue_limit * 0.8,
            )

    # Public API methods

    def record_request_latency(self, latency_ms: float, endpoint: str = ""):
        """Record request latency metric."""
        self.metrics_collector.record_metric(
            MetricType.LATENCY, latency_ms, {"endpoint": endpoint}
        )

    def record_error(self, error_type: str = "", endpoint: str = ""):
        """Record error occurrence."""
        self.metrics_collector.record_metric(
            MetricType.ERROR_RATE, 1.0, {"error_type": error_type, "endpoint": endpoint}
        )

    def record_throughput(self, requests_per_second: float):
        """Record throughput metric."""
        self.metrics_collector.record_metric(MetricType.THROUGHPUT, requests_per_second)

    def add_reference_data(self, feature_name: str, value: float):
        """Add reference data for drift detection."""
        self.drift_detector.add_reference_data(feature_name, value)

    def add_current_data(self, feature_name: str, value: float):
        """Add current data for drift detection."""
        self.drift_detector.add_current_data(feature_name, value)

    def register_fallback_model(self, model_id: str):
        """Register a fallback model."""
        if model_id not in self.fallback_models:
            self.fallback_models.append(model_id)
            logger.info(f"Registered fallback model: {model_id}")

    def get_next_fallback_model(self) -> Optional[str]:
        """Get next fallback model in rotation."""
        if not self.fallback_models:
            return None

        model = self.fallback_models[self.current_fallback_index]
        self.current_fallback_index = (self.current_fallback_index + 1) % len(
            self.fallback_models
        )

        return model

    async def check_rate_limit(self, client_id: str = "default") -> bool:
        """Check if request is within rate limits."""
        return await self.throttler.check_rate_limit(client_id)

    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        return await self.retry_manager.execute_with_retry(func, *args, **kwargs)

    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard."""
        return {
            "sla_status": {
                "latency": self.metrics_collector.get_metric_summary(
                    MetricType.LATENCY
                ),
                "error_rate": self.metrics_collector.get_metric_summary(
                    MetricType.ERROR_RATE
                ),
                "throughput": self.metrics_collector.get_metric_summary(
                    MetricType.THROUGHPUT
                ),
            },
            "rate_limiting": self.throttler.get_rate_limit_status(),
            "active_alerts": [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in self.alert_manager.get_active_alerts()
            ],
            "fallback_models": self.fallback_models,
            "system_health": {
                "monitoring_active": self.monitoring_active,
                "queue_size": self.throttler.get_queue_size(),
            },
        }

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")


# Utility functions


def create_performance_monitor(
    max_latency_ms: float = 100.0,
    max_error_rate: float = 0.05,
    enable_alerts: bool = True,
    enable_throttling: bool = True,
) -> PerformanceMonitor:
    """Create performance monitor with common configuration."""

    sla_config = SLAConfig(max_latency_ms=max_latency_ms, max_error_rate=max_error_rate)

    alert_config = AlertConfig(enabled=enable_alerts)
    throttle_config = ThrottleConfig(enabled=enable_throttling)
    retry_config = RetryConfig(enabled=True)

    return PerformanceMonitor(
        sla_config=sla_config,
        alert_config=alert_config,
        throttle_config=throttle_config,
        retry_config=retry_config,
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        monitor = create_performance_monitor()

        # Simulate some metrics
        monitor.record_request_latency(85.5, "/predict")
        monitor.record_request_latency(120.0, "/predict")  # SLA violation
        monitor.record_throughput(15.5)

        # Add some reference data for drift detection
        for i in range(100):
            monitor.add_reference_data("credit_score", 700 + i * 2)

        # Add current data (with drift)
        for i in range(50):
            monitor.add_current_data("credit_score", 650 + i * 3)

        # Get dashboard data
        dashboard = monitor.get_performance_dashboard_data()
        print(f"Dashboard data: {json.dumps(dashboard, indent=2, default=str)}")

        # Wait a bit for background monitoring
        await asyncio.sleep(2)

        monitor.stop_monitoring()

    asyncio.run(main())
