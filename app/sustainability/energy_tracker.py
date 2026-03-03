"""
Minimal energy tracker compatibility layer.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class EnergyReport:
    experiment_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    total_energy_kwh: float


class EnergyTracker:
    def __init__(self):
        self._active: Dict[str, datetime] = {}

    def start_tracking(self, experiment_id: str) -> str:
        self._active[experiment_id] = datetime.now()
        return experiment_id

    def stop_tracking(self, experiment_id: Optional[str] = None) -> EnergyReport:
        if experiment_id is None and self._active:
            experiment_id = next(iter(self._active.keys()))
        experiment_id = experiment_id or "unknown"

        start = self._active.pop(experiment_id, datetime.now())
        end = datetime.now()
        duration = max(0.0, (end - start).total_seconds())
        # Conservative synthetic estimate.
        energy_kwh = max(0.0001, duration * 0.000025)

        return EnergyReport(
            experiment_id=experiment_id,
            start_time=start.isoformat(),
            end_time=end.isoformat(),
            duration_seconds=duration,
            total_energy_kwh=energy_kwh,
        )
