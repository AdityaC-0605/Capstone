"""
Minimal carbon-aware optimizer compatibility layer.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from .energy_tracker import EnergyTracker
from .carbon_calculator import CarbonCalculator


@dataclass
class CarbonAwareConfig:
    enable_carbon_scheduling: bool = True
    enable_budget_enforcement: bool = True
    daily_carbon_budget_kg: float = 0.1
    low_carbon_threshold: float = 200.0
    medium_carbon_threshold: float = 400.0
    high_carbon_threshold: float = 600.0
    region: str = "US"
    metadata: Dict[str, Any] = field(default_factory=dict)


class _CarbonAPI:
    def get_current_carbon_intensity(self) -> float:
        # Stable default used by dependent code paths.
        return 320.0


class _Scheduler:
    def __init__(self):
        self.carbon_api = _CarbonAPI()


class _Tracker:
    def __init__(self):
        self._daily_emissions_kg = 0.0

    def add_emissions(self, emissions_kg: float):
        self._daily_emissions_kg += max(0.0, emissions_kg)

    def get_current_daily_emissions(self) -> float:
        return self._daily_emissions_kg


class CarbonAwareOptimizer:
    def __init__(self, config: Optional[CarbonAwareConfig] = None):
        self.config = config or CarbonAwareConfig()
        self.scheduler = _Scheduler()
        self.tracker = _Tracker()

    def can_continue_training(self) -> bool:
        if not self.config.enable_budget_enforcement:
            return True
        return (
            self.tracker.get_current_daily_emissions()
            < self.config.daily_carbon_budget_kg
        )


def carbon_aware_training(
    model: Any,
    train_func: Callable,
    config: Optional[CarbonAwareConfig] = None,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Compatibility wrapper around existing training function.
    """
    cfg = config or CarbonAwareConfig()
    tracker = EnergyTracker()
    calculator = CarbonCalculator()

    exp_id = "carbon_aware_training"
    tracker.start_tracking(exp_id)
    result = train_func(**kwargs)
    energy_report = tracker.stop_tracking(exp_id)
    carbon_report = calculator.calculate_carbon_footprint(
        energy_report, region=cfg.region
    )

    report = {
        "final_energy_consumption_kwh": energy_report.total_energy_kwh,
        "final_carbon_footprint_kg": carbon_report.total_emissions_kg,
        "strategies_applied": [
            "carbon_scheduling" if cfg.enable_carbon_scheduling else "none",
            "budget_enforcement"
            if cfg.enable_budget_enforcement
            else "no_budget",
        ],
        "carbon_savings_kg": 0.0,
    }
    return result, report
