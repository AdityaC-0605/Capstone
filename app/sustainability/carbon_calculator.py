"""
Minimal carbon calculator compatibility layer.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class CarbonFootprintReport:
    region: str
    total_emissions_kg: float
    emissions_factor_kg_per_kwh: float
    total_energy_kwh: float


class CarbonCalculator:
    """Converts energy usage into carbon footprint estimates."""

    REGION_FACTORS = {
        "US": 0.4,
        "EU": 0.25,
        "IN": 0.7,
    }

    def calculate_carbon_footprint(
        self, energy_report: Any, region: str = "US"
    ) -> CarbonFootprintReport:
        total_energy_kwh = float(
            getattr(energy_report, "total_energy_kwh", 0.0) or 0.0
        )
        factor = self.REGION_FACTORS.get(region.upper(), 0.4)
        emissions = total_energy_kwh * factor
        return CarbonFootprintReport(
            region=region.upper(),
            total_emissions_kg=emissions,
            emissions_factor_kg_per_kwh=factor,
            total_energy_kwh=total_energy_kwh,
        )
