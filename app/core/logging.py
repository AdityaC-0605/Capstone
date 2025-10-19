"""
Logging and monitoring infrastructure for the sustainable credit risk AI system.
"""

import logging
import logging.handlers
import sys
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
from enum import Enum

from .config import get_config


class LogLevel(Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    extra_data: Dict[str, Any]


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            process_id=record.process,
            extra_data=getattr(record, "extra_data", {}),
        )

        return json.dumps(asdict(log_entry), default=str)


class AuditLogger:
    """Specialized logger for audit events."""

    def __init__(self, name: str = "audit"):
        self.logger = logging.getLogger(name)
        self._setup_audit_logger()

    def _setup_audit_logger(self):
        """Set up audit logger with specific configuration."""
        config = get_config()

        # Create audit log file handler
        audit_log_path = Path(config.logs_path) / "audit.log"
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            audit_log_path, maxBytes=10 * 1024 * 1024, backupCount=10  # 10MB
        )

        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_data_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log data access events."""
        self.logger.info(
            f"Data access: {action} on {resource}",
            extra={
                "extra_data": {
                    "event_type": "data_access",
                    "user_id": user_id,
                    "resource": resource,
                    "action": action,
                    "success": success,
                    "details": details or {},
                }
            },
        )

    def log_model_operation(
        self,
        user_id: str,
        model_id: str,
        operation: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log model operations."""
        self.logger.info(
            f"Model operation: {operation} on {model_id}",
            extra={
                "extra_data": {
                    "event_type": "model_operation",
                    "user_id": user_id,
                    "model_id": model_id,
                    "operation": operation,
                    "success": success,
                    "details": details or {},
                }
            },
        )

    def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str],
        severity: str,
        details: Dict[str, Any],
    ):
        """Log security events."""
        self.logger.warning(
            f"Security event: {event_type}",
            extra={
                "extra_data": {
                    "event_type": "security",
                    "security_event_type": event_type,
                    "user_id": user_id,
                    "severity": severity,
                    "details": details,
                }
            },
        )


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""

    def __init__(self):
        self.logger = logging.getLogger("performance")
        self._metrics_storage = {}
        self._lock = threading.Lock()

    @contextmanager
    def measure_time(
        self, operation_name: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager to measure operation time."""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time

            self.record_metric(
                metric_name="operation_duration",
                value=duration,
                unit="seconds",
                tags={"operation": operation_name, **(metadata or {})},
            )

    def record_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """Record a performance metric."""
        timestamp = datetime.now().isoformat()

        metric_entry = {
            "timestamp": timestamp,
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "tags": tags or {},
        }

        with self._lock:
            if metric_name not in self._metrics_storage:
                self._metrics_storage[metric_name] = []
            self._metrics_storage[metric_name].append(metric_entry)

        self.logger.info(
            f"Metric recorded: {metric_name}={value} {unit}",
            extra={"extra_data": metric_entry},
        )

    def get_metrics(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get stored metrics."""
        with self._lock:
            if metric_name:
                return self._metrics_storage.get(metric_name, [])
            return self._metrics_storage.copy()

    def clear_metrics(self, metric_name: Optional[str] = None):
        """Clear stored metrics."""
        with self._lock:
            if metric_name:
                self._metrics_storage.pop(metric_name, None)
            else:
                self._metrics_storage.clear()


class LoggingManager:
    """Central logging manager."""

    def __init__(self):
        self.config = get_config()
        self.audit_logger = AuditLogger()
        self.performance_monitor = PerformanceMonitor()
        self._setup_root_logger()

    def _setup_root_logger(self):
        """Set up root logger configuration."""
        # Create logs directory
        logs_path = Path(self.config.logs_path)
        logs_path.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler
        file_handler = logging.handlers.RotatingFileHandler(
            logs_path / "application.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5,
        )
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            logs_path / "error.log", maxBytes=10 * 1024 * 1024, backupCount=10  # 10MB
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(error_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        return logging.getLogger(name)

    def get_audit_logger(self) -> AuditLogger:
        """Get the audit logger."""
        return self.audit_logger

    def get_performance_monitor(self) -> PerformanceMonitor:
        """Get the performance monitor."""
        return self.performance_monitor


# Global logging manager instance
logging_manager = LoggingManager()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging_manager.get_logger(name)


def get_audit_logger() -> AuditLogger:
    """Get the audit logger."""
    return logging_manager.get_audit_logger()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the performance monitor."""
    return logging_manager.get_performance_monitor()
