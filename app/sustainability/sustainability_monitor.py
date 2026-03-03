"""
Lightweight sustainability monitoring utilities.

This compatibility module provides the API expected by inference and batch
services after the sustainability package refactor.
"""

from dataclasses import dataclass
from datetime import datetime
from functools import wraps
import inspect
from typing import Any, Callable, Dict, Optional


@dataclass
class SustainabilityReport:
    total_energy_kwh: float
    total_emissions_kg: float
    duration_seconds: float
    experiment_id: str
    timestamp: str


class SustainabilityMonitor:
    """Tracks coarse-grained energy/carbon usage per experiment."""

    def __init__(self):
        self._active: Dict[str, datetime] = {}

    def start_experiment_tracking(
        self, experiment_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        self._active[experiment_id] = datetime.now()
        return experiment_id

    def stop_experiment_tracking(self, experiment_id: str) -> Dict[str, Any]:
        start = self._active.pop(experiment_id, datetime.now())
        duration_seconds = max(
            0.0, (datetime.now() - start).total_seconds()
        )

        # Conservative synthetic estimates for compatibility.
        total_energy_kwh = max(0.0002, duration_seconds * 0.00003)
        total_emissions_kg = total_energy_kwh * 0.4

        return {
            "experiment_id": experiment_id,
            "duration_seconds": duration_seconds,
            "energy_kwh": total_energy_kwh,
            "carbon_emissions": total_emissions_kg,
            "timestamp": datetime.now().isoformat(),
        }


def track_sustainability(func: Callable) -> Callable:
    """Decorator placeholder to keep backwards compatibility."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
