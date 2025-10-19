"""
Error Handling and Input Protection System for Credit Risk API.

This module implements comprehensive error handling, input sanitization,
adversarial input protection, dead letter queues, and anomaly detection
for production API security and reliability.
"""

import re
import json
import time
import hashlib
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import warnings
import traceback
from abc import ABC, abstractmethod

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Install with: pip install numpy")

try:
    from pydantic import BaseModel, ValidationError, validator

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    warnings.warn("Pydantic not available. Install with: pip install pydantic")

try:
    from ..core.logging import get_logger, get_audit_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class ErrorType(Enum):
    """Types of errors that can occur."""

    VALIDATION_ERROR = "validation_error"
    SANITIZATION_ERROR = "sanitization_error"
    ADVERSARIAL_INPUT = "adversarial_input"
    ANOMALY_DETECTED = "anomaly_detected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    MODEL_ERROR = "model_error"
    SYSTEM_ERROR = "system_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"


class SeverityLevel(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InputAnomalyType(Enum):
    """Types of input anomalies."""

    STATISTICAL_OUTLIER = "statistical_outlier"
    PATTERN_ANOMALY = "pattern_anomaly"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    SIZE_ANOMALY = "size_anomaly"
    ENCODING_ANOMALY = "encoding_anomaly"
    INJECTION_ATTEMPT = "injection_attempt"


@dataclass
class ErrorEvent:
    """Error event data structure."""

    id: str
    error_type: ErrorType
    severity: SeverityLevel
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    client_id: Optional[str] = None
    endpoint: Optional[str] = None
    request_id: Optional[str] = None
    stack_trace: Optional[str] = None
    resolved: bool = False


@dataclass
class InputValidationRule:
    """Input validation rule definition."""

    field_name: str
    rule_type: str  # "range", "pattern", "length", "type", "custom"
    parameters: Dict[str, Any]
    error_message: str
    severity: SeverityLevel = SeverityLevel.MEDIUM


@dataclass
class SanitizationRule:
    """Input sanitization rule definition."""

    field_name: str
    sanitizer_type: str  # "strip", "escape", "normalize", "filter", "custom"
    parameters: Dict[str, Any]
    preserve_original: bool = True


@dataclass
class DeadLetterMessage:
    """Dead letter queue message."""

    id: str
    original_request: Dict[str, Any]
    error_type: ErrorType
    error_message: str
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: Optional[datetime] = None


class InputSanitizer:
    """Handles input sanitization and cleaning."""

    def __init__(self):
        self.sanitization_rules: Dict[str, List[SanitizationRule]] = {}
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\'\s*(OR|AND)\s*\'\w*\'\s*=\s*\'\w*\')",
        ]
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
        ]

    def add_sanitization_rule(self, rule: SanitizationRule):
        """Add a sanitization rule."""
        if rule.field_name not in self.sanitization_rules:
            self.sanitization_rules[rule.field_name] = []
        self.sanitization_rules[rule.field_name].append(rule)

    def sanitize_input(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Sanitize input data and return cleaned data with warnings."""
        sanitized_data = data.copy()
        warnings = []

        for field_name, value in data.items():
            if field_name in self.sanitization_rules:
                for rule in self.sanitization_rules[field_name]:
                    try:
                        sanitized_value, warning = self._apply_sanitization_rule(
                            value, rule
                        )
                        sanitized_data[field_name] = sanitized_value
                        if warning:
                            warnings.append(f"{field_name}: {warning}")
                    except Exception as e:
                        warnings.append(f"{field_name}: Sanitization failed - {str(e)}")

            # Apply general sanitization
            if isinstance(value, str):
                sanitized_value, general_warnings = self._general_string_sanitization(
                    value
                )
                sanitized_data[field_name] = sanitized_value
                warnings.extend([f"{field_name}: {w}" for w in general_warnings])

        return sanitized_data, warnings

    def _apply_sanitization_rule(
        self, value: Any, rule: SanitizationRule
    ) -> Tuple[Any, Optional[str]]:
        """Apply a specific sanitization rule."""
        if rule.sanitizer_type == "strip":
            if isinstance(value, str):
                return value.strip(), None

        elif rule.sanitizer_type == "escape":
            if isinstance(value, str):
                escaped = (
                    value.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&#x27;")
                )
                return escaped, "HTML characters escaped" if escaped != value else None

        elif rule.sanitizer_type == "normalize":
            if isinstance(value, str):
                normalized = re.sub(r"\s+", " ", value).strip()
                return normalized, (
                    "Whitespace normalized" if normalized != value else None
                )

        elif rule.sanitizer_type == "filter":
            if isinstance(value, str):
                pattern = rule.parameters.get("pattern", "")
                replacement = rule.parameters.get("replacement", "")
                filtered = re.sub(pattern, replacement, value)
                return filtered, "Content filtered" if filtered != value else None

        elif rule.sanitizer_type == "custom":
            custom_func = rule.parameters.get("function")
            if custom_func and callable(custom_func):
                return custom_func(value), "Custom sanitization applied"

        return value, None

    def _general_string_sanitization(self, value: str) -> Tuple[str, List[str]]:
        """Apply general string sanitization."""
        warnings = []
        sanitized = value

        # Check for SQL injection patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                warnings.append("Potential SQL injection pattern detected")
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        # Check for XSS patterns
        for pattern in self.xss_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                warnings.append("Potential XSS pattern detected")
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        # Remove null bytes
        if "\x00" in sanitized:
            warnings.append("Null bytes removed")
            sanitized = sanitized.replace("\x00", "")

        # Limit length
        max_length = 10000  # Configurable
        if len(sanitized) > max_length:
            warnings.append(f"Input truncated to {max_length} characters")
            sanitized = sanitized[:max_length]

        return sanitized, warnings

    def detect_adversarial_patterns(self, data: Dict[str, Any]) -> List[str]:
        """Detect potential adversarial input patterns."""
        threats = []

        for field_name, value in data.items():
            if isinstance(value, str):
                # Check for common attack patterns
                if self._check_injection_patterns(value):
                    threats.append(f"{field_name}: Injection pattern detected")

                if self._check_encoding_attacks(value):
                    threats.append(f"{field_name}: Encoding attack detected")

                if self._check_buffer_overflow(value):
                    threats.append(f"{field_name}: Potential buffer overflow")

            elif isinstance(value, (int, float)):
                if self._check_numeric_attacks(value):
                    threats.append(f"{field_name}: Suspicious numeric value")

        return threats

    def _check_injection_patterns(self, value: str) -> bool:
        """Check for injection attack patterns."""
        patterns = self.sql_injection_patterns + self.xss_patterns
        return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)

    def _check_encoding_attacks(self, value: str) -> bool:
        """Check for encoding-based attacks."""
        # Check for multiple encoding layers
        encoded_patterns = [
            r"%[0-9a-fA-F]{2}",  # URL encoding
            r"&#\d+;",  # HTML entity encoding
            r"\\u[0-9a-fA-F]{4}",  # Unicode encoding
        ]

        encoding_count = sum(
            1 for pattern in encoded_patterns if re.search(pattern, value)
        )

        return encoding_count > 2  # Multiple encoding layers suspicious

    def _check_buffer_overflow(self, value: str) -> bool:
        """Check for potential buffer overflow attempts."""
        # Very long strings with repeated patterns
        if len(value) > 1000:
            # Check for repeated characters (potential overflow)
            for char in set(value):
                if value.count(char) > len(value) * 0.8:
                    return True

        return False

    def _check_numeric_attacks(self, value: Union[int, float]) -> bool:
        """Check for suspicious numeric values."""
        # Check for extreme values that might cause overflow
        if isinstance(value, int):
            return abs(value) > 2**31 - 1  # 32-bit integer limit
        elif isinstance(value, float):
            return abs(value) > 1e308 or (value != 0 and abs(value) < 1e-308)

        return False


class InputValidator:
    """Handles input validation with configurable rules."""

    def __init__(self):
        self.validation_rules: Dict[str, List[InputValidationRule]] = {}
        self.schema_cache: Dict[str, Any] = {}

    def add_validation_rule(self, rule: InputValidationRule):
        """Add a validation rule."""
        if rule.field_name not in self.validation_rules:
            self.validation_rules[rule.field_name] = []
        self.validation_rules[rule.field_name].append(rule)

    def validate_input(
        self, data: Dict[str, Any], schema_name: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Validate input data against rules and schema."""
        errors = []

        # Schema validation if available
        if schema_name and PYDANTIC_AVAILABLE:
            schema_errors = self._validate_schema(data, schema_name)
            errors.extend(schema_errors)

        # Rule-based validation
        for field_name, value in data.items():
            if field_name in self.validation_rules:
                for rule in self.validation_rules[field_name]:
                    rule_errors = self._apply_validation_rule(value, rule)
                    errors.extend(rule_errors)

        return len(errors) == 0, errors

    def _validate_schema(self, data: Dict[str, Any], schema_name: str) -> List[str]:
        """Validate against Pydantic schema."""
        errors = []

        if schema_name in self.schema_cache:
            schema_class = self.schema_cache[schema_name]
            try:
                schema_class(**data)
            except ValidationError as e:
                for error in e.errors():
                    field = ".".join(str(loc) for loc in error["loc"])
                    errors.append(f"{field}: {error['msg']}")

        return errors

    def _apply_validation_rule(
        self, value: Any, rule: InputValidationRule
    ) -> List[str]:
        """Apply a specific validation rule."""
        errors = []

        try:
            if rule.rule_type == "range":
                if isinstance(value, (int, float)):
                    min_val = rule.parameters.get("min")
                    max_val = rule.parameters.get("max")

                    if min_val is not None and value < min_val:
                        errors.append(f"{rule.field_name}: {rule.error_message}")
                    elif max_val is not None and value > max_val:
                        errors.append(f"{rule.field_name}: {rule.error_message}")

            elif rule.rule_type == "pattern":
                if isinstance(value, str):
                    pattern = rule.parameters.get("pattern", "")
                    if not re.match(pattern, value):
                        errors.append(f"{rule.field_name}: {rule.error_message}")

            elif rule.rule_type == "length":
                if hasattr(value, "__len__"):
                    min_len = rule.parameters.get("min", 0)
                    max_len = rule.parameters.get("max", float("inf"))

                    if len(value) < min_len or len(value) > max_len:
                        errors.append(f"{rule.field_name}: {rule.error_message}")

            elif rule.rule_type == "type":
                expected_type = rule.parameters.get("type")
                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"{rule.field_name}: {rule.error_message}")

            elif rule.rule_type == "custom":
                validator_func = rule.parameters.get("function")
                if validator_func and callable(validator_func):
                    if not validator_func(value):
                        errors.append(f"{rule.field_name}: {rule.error_message}")

        except Exception as e:
            errors.append(f"{rule.field_name}: Validation error - {str(e)}")

        return errors

    def register_schema(self, schema_name: str, schema_class: Any):
        """Register a Pydantic schema for validation."""
        self.schema_cache[schema_name] = schema_class


class AnomalyDetector:
    """Detects anomalous input patterns and suspicious requests."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.request_history: deque = deque(maxlen=window_size)
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.client_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def add_request(self, request_data: Dict[str, Any], client_id: str = "unknown"):
        """Add request to history for pattern analysis."""
        request_info = {
            "data": request_data,
            "client_id": client_id,
            "timestamp": datetime.now(),
            "size": len(json.dumps(request_data)),
        }

        self.request_history.append(request_info)
        self._update_feature_stats(request_data)
        self._update_client_patterns(client_id, request_info)

    def detect_anomalies(
        self, request_data: Dict[str, Any], client_id: str = "unknown"
    ) -> List[Tuple[InputAnomalyType, str, float]]:
        """Detect anomalies in the current request."""
        anomalies = []

        # Statistical outlier detection
        stat_anomalies = self._detect_statistical_outliers(request_data)
        anomalies.extend(stat_anomalies)

        # Pattern anomaly detection
        pattern_anomalies = self._detect_pattern_anomalies(request_data, client_id)
        anomalies.extend(pattern_anomalies)

        # Frequency anomaly detection
        freq_anomalies = self._detect_frequency_anomalies(client_id)
        anomalies.extend(freq_anomalies)

        # Size anomaly detection
        size_anomalies = self._detect_size_anomalies(request_data)
        anomalies.extend(size_anomalies)

        return anomalies

    def _update_feature_stats(self, request_data: Dict[str, Any]):
        """Update feature statistics for anomaly detection."""
        for field_name, value in request_data.items():
            if isinstance(value, (int, float)):
                if field_name not in self.feature_stats:
                    self.feature_stats[field_name] = {
                        "values": deque(maxlen=self.window_size),
                        "mean": 0.0,
                        "std": 0.0,
                    }

                stats = self.feature_stats[field_name]
                stats["values"].append(value)

                if len(stats["values"]) > 1:
                    values = list(stats["values"])
                    stats["mean"] = (
                        np.mean(values)
                        if NUMPY_AVAILABLE
                        else sum(values) / len(values)
                    )
                    if NUMPY_AVAILABLE:
                        stats["std"] = np.std(values)
                    else:
                        mean = stats["mean"]
                        variance = sum((x - mean) ** 2 for x in values) / len(values)
                        stats["std"] = variance**0.5

    def _update_client_patterns(self, client_id: str, request_info: Dict[str, Any]):
        """Update client behavior patterns."""
        if client_id not in self.client_patterns:
            self.client_patterns[client_id] = {
                "request_count": 0,
                "last_request": None,
                "request_intervals": deque(maxlen=100),
                "request_sizes": deque(maxlen=100),
            }

        pattern = self.client_patterns[client_id]
        pattern["request_count"] += 1

        if pattern["last_request"]:
            interval = (
                request_info["timestamp"] - pattern["last_request"]
            ).total_seconds()
            pattern["request_intervals"].append(interval)

        pattern["last_request"] = request_info["timestamp"]
        pattern["request_sizes"].append(request_info["size"])

    def _detect_statistical_outliers(
        self, request_data: Dict[str, Any]
    ) -> List[Tuple[InputAnomalyType, str, float]]:
        """Detect statistical outliers in numeric fields."""
        anomalies = []

        for field_name, value in request_data.items():
            if isinstance(value, (int, float)) and field_name in self.feature_stats:
                stats = self.feature_stats[field_name]

                if stats["std"] > 0:
                    z_score = abs(value - stats["mean"]) / stats["std"]

                    if z_score > 3.0:  # 3-sigma rule
                        anomalies.append(
                            (
                                InputAnomalyType.STATISTICAL_OUTLIER,
                                f"{field_name}: value {value} is {z_score:.2f} standard deviations from mean",
                                z_score,
                            )
                        )

        return anomalies

    def _detect_pattern_anomalies(
        self, request_data: Dict[str, Any], client_id: str
    ) -> List[Tuple[InputAnomalyType, str, float]]:
        """Detect pattern anomalies in request structure."""
        anomalies = []

        # Check for unusual field combinations
        current_fields = set(request_data.keys())

        # Compare with historical field patterns
        historical_patterns = []
        for req in list(self.request_history)[-100:]:  # Last 100 requests
            historical_patterns.append(set(req["data"].keys()))

        if historical_patterns:
            # Calculate Jaccard similarity with historical patterns
            similarities = []
            for pattern in historical_patterns:
                intersection = len(current_fields.intersection(pattern))
                union = len(current_fields.union(pattern))
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)

            avg_similarity = sum(similarities) / len(similarities)

            if avg_similarity < 0.5:  # Low similarity threshold
                anomalies.append(
                    (
                        InputAnomalyType.PATTERN_ANOMALY,
                        f"Request structure differs significantly from historical patterns (similarity: {avg_similarity:.2f})",
                        1.0 - avg_similarity,
                    )
                )

        return anomalies

    def _detect_frequency_anomalies(
        self, client_id: str
    ) -> List[Tuple[InputAnomalyType, str, float]]:
        """Detect frequency-based anomalies."""
        anomalies = []

        if client_id in self.client_patterns:
            pattern = self.client_patterns[client_id]
            intervals = list(pattern["request_intervals"])

            if len(intervals) > 10:
                # Check for unusually high frequency (potential bot/attack)
                recent_intervals = intervals[-10:]
                avg_interval = sum(recent_intervals) / len(recent_intervals)

                if avg_interval < 0.1:  # Less than 100ms between requests
                    anomalies.append(
                        (
                            InputAnomalyType.FREQUENCY_ANOMALY,
                            f"Unusually high request frequency: {1/avg_interval:.1f} requests/second",
                            1.0 / max(avg_interval, 0.001),
                        )
                    )

        return anomalies

    def _detect_size_anomalies(
        self, request_data: Dict[str, Any]
    ) -> List[Tuple[InputAnomalyType, str, float]]:
        """Detect size-based anomalies."""
        anomalies = []

        request_size = len(json.dumps(request_data))

        # Check against historical sizes
        if len(self.request_history) > 10:
            historical_sizes = [
                req["size"] for req in list(self.request_history)[-100:]
            ]
            avg_size = sum(historical_sizes) / len(historical_sizes)

            if request_size > avg_size * 5:  # 5x larger than average
                anomalies.append(
                    (
                        InputAnomalyType.SIZE_ANOMALY,
                        f"Request size {request_size} bytes is unusually large (avg: {avg_size:.0f} bytes)",
                        request_size / avg_size,
                    )
                )

        return anomalies


class DeadLetterQueue:
    """Manages failed requests and retry logic."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queue: Dict[str, DeadLetterMessage] = {}
        self.retry_queue: List[str] = []

    def add_message(
        self,
        request_data: Dict[str, Any],
        error_type: ErrorType,
        error_message: str,
        max_retries: int = 3,
    ) -> str:
        """Add a failed request to the dead letter queue."""
        message_id = hashlib.md5(
            f"{json.dumps(request_data, sort_keys=True)}_{time.time()}".encode()
        ).hexdigest()

        message = DeadLetterMessage(
            id=message_id,
            original_request=request_data,
            error_type=error_type,
            error_message=error_message,
            timestamp=datetime.now(),
            max_retries=max_retries,
        )

        # Remove oldest messages if queue is full
        if len(self.queue) >= self.max_size:
            oldest_id = min(self.queue.keys(), key=lambda k: self.queue[k].timestamp)
            del self.queue[oldest_id]

        self.queue[message_id] = message

        logger.warning(
            f"Added message to dead letter queue: {message_id} - {error_message}"
        )

        return message_id

    def get_retry_candidates(self) -> List[DeadLetterMessage]:
        """Get messages that are ready for retry."""
        now = datetime.now()
        candidates = []

        for message in self.queue.values():
            if message.retry_count < message.max_retries and (
                message.next_retry_at is None or message.next_retry_at <= now
            ):
                candidates.append(message)

        return candidates

    def mark_retry_attempt(self, message_id: str, success: bool):
        """Mark a retry attempt as successful or failed."""
        if message_id in self.queue:
            message = self.queue[message_id]

            if success:
                del self.queue[message_id]
                logger.info(f"Dead letter message {message_id} successfully processed")
            else:
                message.retry_count += 1

                if message.retry_count >= message.max_retries:
                    logger.error(
                        f"Dead letter message {message_id} exceeded max retries"
                    )
                else:
                    # Exponential backoff
                    delay_minutes = 2**message.retry_count
                    message.next_retry_at = datetime.now() + timedelta(
                        minutes=delay_minutes
                    )
                    logger.warning(
                        f"Dead letter message {message_id} scheduled for retry in {delay_minutes} minutes"
                    )

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get dead letter queue statistics."""
        error_counts = defaultdict(int)
        retry_counts = defaultdict(int)

        for message in self.queue.values():
            error_counts[message.error_type.value] += 1
            retry_counts[message.retry_count] += 1

        return {
            "total_messages": len(self.queue),
            "error_type_distribution": dict(error_counts),
            "retry_count_distribution": dict(retry_counts),
            "oldest_message": (
                min(
                    self.queue.values(), key=lambda m: m.timestamp
                ).timestamp.isoformat()
                if self.queue
                else None
            ),
        }


class ErrorLogger:
    """Comprehensive error logging and alerting system."""

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.error_history: deque = deque(maxlen=max_history)
        self.error_counts: Dict[ErrorType, int] = defaultdict(int)
        self.client_error_counts: Dict[str, Dict[ErrorType, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def log_error(
        self,
        error_type: ErrorType,
        message: str,
        details: Dict[str, Any] = None,
        client_id: str = None,
        endpoint: str = None,
        request_id: str = None,
        severity: SeverityLevel = SeverityLevel.MEDIUM,
    ) -> str:
        """Log an error event."""

        error_id = (
            f"err_{int(time.time())}_{hashlib.md5(message.encode()).hexdigest()[:8]}"
        )

        error_event = ErrorEvent(
            id=error_id,
            error_type=error_type,
            severity=severity,
            message=message,
            details=details or {},
            timestamp=datetime.now(),
            client_id=client_id,
            endpoint=endpoint,
            request_id=request_id,
            stack_trace=(
                traceback.format_exc()
                if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
                else None
            ),
        )

        self.error_history.append(error_event)
        self.error_counts[error_type] += 1

        if client_id:
            self.client_error_counts[client_id][error_type] += 1

        # Log to system logger
        log_level = {
            SeverityLevel.LOW: logger.info,
            SeverityLevel.MEDIUM: logger.warning,
            SeverityLevel.HIGH: logger.error,
            SeverityLevel.CRITICAL: logger.critical,
        }[severity]

        log_level(f"Error {error_id}: {message} - {details}")

        # Audit log for critical errors
        if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            audit_logger.log_model_operation(
                user_id=client_id or "unknown",
                model_id="error_handler",
                operation="error_logged",
                success=False,
                details={
                    "error_id": error_id,
                    "error_type": error_type.value,
                    "severity": severity.value,
                    "message": message,
                    "endpoint": endpoint,
                },
            )

        return error_id

    def get_error_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time window."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)

        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]

        error_type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        endpoint_counts = defaultdict(int)
        hourly_counts = defaultdict(int)

        for error in recent_errors:
            error_type_counts[error.error_type.value] += 1
            severity_counts[error.severity.value] += 1

            if error.endpoint:
                endpoint_counts[error.endpoint] += 1

            hour_key = error.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_counts[hour_key] += 1

        return {
            "total_errors": len(recent_errors),
            "error_rate": len(recent_errors) / max(time_window_hours, 1),
            "error_type_distribution": dict(error_type_counts),
            "severity_distribution": dict(severity_counts),
            "endpoint_distribution": dict(endpoint_counts),
            "hourly_distribution": dict(hourly_counts),
            "top_error_clients": self._get_top_error_clients(recent_errors),
        }

    def _get_top_error_clients(
        self, errors: List[ErrorEvent], top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top clients by error count."""
        client_counts = defaultdict(int)

        for error in errors:
            if error.client_id:
                client_counts[error.client_id] += 1

        sorted_clients = sorted(client_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            {"client_id": client_id, "error_count": count}
            for client_id, count in sorted_clients[:top_n]
        ]


class ErrorHandler:
    """Main error handling and input protection system."""

    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.validator = InputValidator()
        self.anomaly_detector = AnomalyDetector()
        self.dead_letter_queue = DeadLetterQueue()
        self.error_logger = ErrorLogger()

        # Initialize default rules
        self._setup_default_rules()

        logger.info("Error handler initialized")

    def _setup_default_rules(self):
        """Setup default validation and sanitization rules."""

        # Credit score validation
        self.validator.add_validation_rule(
            InputValidationRule(
                field_name="credit_score",
                rule_type="range",
                parameters={"min": 300, "max": 850},
                error_message="Credit score must be between 300 and 850",
                severity=SeverityLevel.HIGH,
            )
        )

        # Income validation
        self.validator.add_validation_rule(
            InputValidationRule(
                field_name="income",
                rule_type="range",
                parameters={"min": 0, "max": 10000000},
                error_message="Income must be between 0 and 10,000,000",
                severity=SeverityLevel.MEDIUM,
            )
        )

        # Age validation
        self.validator.add_validation_rule(
            InputValidationRule(
                field_name="age",
                rule_type="range",
                parameters={"min": 18, "max": 120},
                error_message="Age must be between 18 and 120",
                severity=SeverityLevel.HIGH,
            )
        )

        # String field sanitization
        for field in ["loan_purpose", "home_ownership", "verification_status"]:
            self.sanitizer.add_sanitization_rule(
                SanitizationRule(
                    field_name=field, sanitizer_type="normalize", parameters={}
                )
            )

            self.sanitizer.add_sanitization_rule(
                SanitizationRule(
                    field_name=field, sanitizer_type="escape", parameters={}
                )
            )

    async def process_request(
        self,
        request_data: Dict[str, Any],
        client_id: str = "unknown",
        endpoint: str = "unknown",
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Process and validate a request with comprehensive protection."""

        warnings = []
        request_id = f"req_{int(time.time())}_{hashlib.md5(str(request_data).encode()).hexdigest()[:8]}"

        try:
            # 1. Input sanitization
            sanitized_data, sanitization_warnings = self.sanitizer.sanitize_input(
                request_data
            )
            warnings.extend(sanitization_warnings)

            # 2. Adversarial pattern detection
            adversarial_threats = self.sanitizer.detect_adversarial_patterns(
                sanitized_data
            )
            if adversarial_threats:
                error_id = self.error_logger.log_error(
                    ErrorType.ADVERSARIAL_INPUT,
                    f"Adversarial input detected: {'; '.join(adversarial_threats)}",
                    {"threats": adversarial_threats, "original_data": request_data},
                    client_id,
                    endpoint,
                    request_id,
                    SeverityLevel.HIGH,
                )

                self.dead_letter_queue.add_message(
                    request_data,
                    ErrorType.ADVERSARIAL_INPUT,
                    f"Adversarial input: {'; '.join(adversarial_threats)}",
                )

                return (
                    False,
                    {},
                    [
                        f"Request blocked due to adversarial input (Error ID: {error_id})"
                    ],
                )

            # 3. Input validation
            is_valid, validation_errors = self.validator.validate_input(sanitized_data)
            if not is_valid:
                error_id = self.error_logger.log_error(
                    ErrorType.VALIDATION_ERROR,
                    f"Input validation failed: {'; '.join(validation_errors)}",
                    {
                        "validation_errors": validation_errors,
                        "sanitized_data": sanitized_data,
                    },
                    client_id,
                    endpoint,
                    request_id,
                    SeverityLevel.MEDIUM,
                )

                self.dead_letter_queue.add_message(
                    request_data,
                    ErrorType.VALIDATION_ERROR,
                    f"Validation failed: {'; '.join(validation_errors)}",
                )

                return False, {}, validation_errors

            # 4. Anomaly detection
            self.anomaly_detector.add_request(sanitized_data, client_id)
            anomalies = self.anomaly_detector.detect_anomalies(
                sanitized_data, client_id
            )

            if anomalies:
                high_risk_anomalies = [
                    a for a in anomalies if a[2] > 2.0
                ]  # High confidence anomalies

                if high_risk_anomalies:
                    anomaly_descriptions = [
                        f"{a[0].value}: {a[1]}" for a in high_risk_anomalies
                    ]

                    error_id = self.error_logger.log_error(
                        ErrorType.ANOMALY_DETECTED,
                        f"High-risk anomalies detected: {'; '.join(anomaly_descriptions)}",
                        {
                            "anomalies": [
                                {"type": a[0].value, "description": a[1], "score": a[2]}
                                for a in anomalies
                            ]
                        },
                        client_id,
                        endpoint,
                        request_id,
                        SeverityLevel.HIGH,
                    )

                    return (
                        False,
                        {},
                        [f"Request blocked due to anomalies (Error ID: {error_id})"],
                    )

                # Log low-risk anomalies as warnings
                for anomaly in anomalies:
                    warnings.append(
                        f"Anomaly detected - {anomaly[0].value}: {anomaly[1]} (score: {anomaly[2]:.2f})"
                    )

            # 5. Success - return sanitized data
            return True, sanitized_data, warnings

        except Exception as e:
            error_id = self.error_logger.log_error(
                ErrorType.SYSTEM_ERROR,
                f"Error processing request: {str(e)}",
                {"exception": str(e), "request_data": request_data},
                client_id,
                endpoint,
                request_id,
                SeverityLevel.CRITICAL,
            )

            self.dead_letter_queue.add_message(
                request_data, ErrorType.SYSTEM_ERROR, f"System error: {str(e)}"
            )

            return (
                False,
                {},
                [f"Internal error processing request (Error ID: {error_id})"],
            )

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "error_statistics": self.error_logger.get_error_statistics(),
            "dead_letter_queue": self.dead_letter_queue.get_queue_stats(),
            "validation_rules": len(self.validator.validation_rules),
            "sanitization_rules": len(self.sanitizer.sanitization_rules),
            "anomaly_detector_history": len(self.anomaly_detector.request_history),
        }

    async def process_dead_letter_retries(self):
        """Process retry candidates from dead letter queue."""
        candidates = self.dead_letter_queue.get_retry_candidates()

        for message in candidates:
            try:
                # Attempt to reprocess the request
                success, processed_data, warnings = await self.process_request(
                    message.original_request, "retry_client", "retry_endpoint"
                )

                self.dead_letter_queue.mark_retry_attempt(message.id, success)

                if success:
                    logger.info(
                        f"Successfully reprocessed dead letter message {message.id}"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to reprocess dead letter message {message.id}: {e}"
                )
                self.dead_letter_queue.mark_retry_attempt(message.id, False)


# Utility functions


def create_error_handler() -> ErrorHandler:
    """Create error handler with default configuration."""
    return ErrorHandler()


if __name__ == "__main__":
    # Example usage
    async def main():
        handler = create_error_handler()

        # Test normal request
        normal_request = {
            "age": 30,
            "income": 50000,
            "credit_score": 720,
            "loan_amount": 25000,
            "loan_purpose": "debt_consolidation",
        }

        success, data, warnings = await handler.process_request(
            normal_request, "test_client", "/predict"
        )
        print(f"Normal request - Success: {success}, Warnings: {len(warnings)}")

        # Test malicious request
        malicious_request = {
            "age": 30,
            "income": 50000,
            "credit_score": "720'; DROP TABLE users; --",
            "loan_amount": 25000,
            "loan_purpose": "<script>alert('xss')</script>",
        }

        success, data, warnings = await handler.process_request(
            malicious_request, "malicious_client", "/predict"
        )
        print(f"Malicious request - Success: {success}, Errors: {len(warnings)}")

        # Get system status
        status = handler.get_system_status()
        print(f"System status: {json.dumps(status, indent=2, default=str)}")

    asyncio.run(main())
