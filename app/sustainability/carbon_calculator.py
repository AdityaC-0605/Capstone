"""
Carbon Footprint Calculation System for Sustainable AI.

This module implements comprehensive carbon footprint calculation including
regional energy mix integration, CO2e emissions calculation from energy data,
carbon footprint tracking across experiments, and carbon budget monitoring.
"""

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

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


logger = get_logger(__name__)
audit_logger = get_audit_logger()


class EnergySource(Enum):
    """Energy source types for carbon intensity calculation."""

    COAL = "coal"
    NATURAL_GAS = "natural_gas"
    NUCLEAR = "nuclear"
    HYDRO = "hydro"
    WIND = "wind"
    SOLAR = "solar"
    BIOMASS = "biomass"
    GEOTHERMAL = "geothermal"
    OIL = "oil"
    OTHER_FOSSIL = "other_fossil"
    OTHER_RENEWABLE = "other_renewable"


@dataclass
class EnergyMix:
    """Regional energy mix composition."""

    region: str
    country_code: str
    sources: Dict[EnergySource, float]  # Percentage of each source (0-100)
    carbon_intensity_gco2_kwh: float  # gCO2/kWh
    last_updated: datetime
    data_source: str = "default"

    def __post_init__(self):
        """Validate energy mix data."""
        total_percentage = sum(self.sources.values())
        if abs(total_percentage - 100.0) > 1.0:  # Allow 1% tolerance
            logger.warning(
                f"Energy mix percentages sum to {total_percentage}%, not 100%"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "region": self.region,
            "country_code": self.country_code,
            "sources": {
                source.value: percentage for source, percentage in self.sources.items()
            },
            "carbon_intensity_gco2_kwh": self.carbon_intensity_gco2_kwh,
            "last_updated": self.last_updated.isoformat(),
            "data_source": self.data_source,
        }


@dataclass
class CarbonFootprintConfig:
    """Configuration for carbon footprint calculation."""

    # Regional settings
    default_region: str = "US"
    default_country_code: str = "USA"

    # Data sources
    energy_mix_data_path: Optional[str] = None
    use_real_time_data: bool = False
    api_key: Optional[str] = None

    # Calculation settings
    include_embodied_carbon: bool = True
    embodied_carbon_factor: float = 0.1  # Additional 10% for hardware manufacturing

    # Budget and alerting
    enable_budget_monitoring: bool = True
    daily_carbon_budget_kg: Optional[float] = None
    monthly_carbon_budget_kg: Optional[float] = None
    annual_carbon_budget_kg: Optional[float] = None

    # Alert thresholds (percentage of budget)
    warning_threshold: float = 0.8  # 80%
    critical_threshold: float = 0.95  # 95%

    # Reporting
    output_dir: str = "carbon_reports"
    save_detailed_reports: bool = True

    # Offset calculation
    enable_offset_calculation: bool = True
    offset_price_per_ton_co2: float = 15.0  # USD per ton CO2


@dataclass
class CarbonFootprint:
    """Container for carbon footprint calculation results."""

    experiment_id: str
    timestamp: datetime

    # Energy consumption
    energy_kwh: float

    # Carbon emissions
    operational_emissions_kg: float  # Direct emissions from energy use
    embodied_emissions_kg: float  # Emissions from hardware manufacturing
    total_emissions_kg: float  # Total carbon footprint

    # Regional data
    region: str
    carbon_intensity_gco2_kwh: float
    energy_mix: Optional[EnergyMix] = None

    # Additional metrics
    equivalent_metrics: Dict[str, float] = field(default_factory=dict)
    offset_cost_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp.isoformat(),
            "energy_kwh": self.energy_kwh,
            "operational_emissions_kg": self.operational_emissions_kg,
            "embodied_emissions_kg": self.embodied_emissions_kg,
            "total_emissions_kg": self.total_emissions_kg,
            "region": self.region,
            "carbon_intensity_gco2_kwh": self.carbon_intensity_gco2_kwh,
            "energy_mix": self.energy_mix.to_dict() if self.energy_mix else None,
            "equivalent_metrics": self.equivalent_metrics,
            "offset_cost_usd": self.offset_cost_usd,
        }


@dataclass
class CarbonBudgetStatus:
    """Carbon budget monitoring status."""

    budget_period: str  # "daily", "monthly", "annual"
    budget_limit_kg: float
    current_usage_kg: float
    remaining_budget_kg: float
    usage_percentage: float

    # Time information
    period_start: datetime
    period_end: datetime
    days_remaining: int

    # Status
    is_over_budget: bool
    alert_level: str  # "normal", "warning", "critical"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "budget_period": self.budget_period,
            "budget_limit_kg": self.budget_limit_kg,
            "current_usage_kg": self.current_usage_kg,
            "remaining_budget_kg": self.remaining_budget_kg,
            "usage_percentage": self.usage_percentage,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "days_remaining": self.days_remaining,
            "is_over_budget": self.is_over_budget,
            "alert_level": self.alert_level,
        }


class EnergyMixDatabase:
    """Database of regional energy mix data for carbon intensity calculation."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.energy_mixes = {}
        self._load_default_data()

        if data_path:
            self._load_custom_data(data_path)

    def _load_default_data(self):
        """Load default energy mix data for major regions."""
        # Data based on 2023 estimates from various sources (IEA, EIA, etc.)
        default_mixes = {
            "US": EnergyMix(
                region="United States",
                country_code="USA",
                sources={
                    EnergySource.NATURAL_GAS: 40.0,
                    EnergySource.COAL: 20.0,
                    EnergySource.NUCLEAR: 20.0,
                    EnergySource.WIND: 10.0,
                    EnergySource.HYDRO: 6.0,
                    EnergySource.SOLAR: 3.0,
                    EnergySource.OTHER_RENEWABLE: 1.0,
                },
                carbon_intensity_gco2_kwh=386.0,
                last_updated=datetime(2023, 1, 1),
                data_source="EIA 2023",
            ),
            "EU": EnergyMix(
                region="European Union",
                country_code="EUR",
                sources={
                    EnergySource.NATURAL_GAS: 20.0,
                    EnergySource.COAL: 15.0,
                    EnergySource.NUCLEAR: 25.0,
                    EnergySource.WIND: 15.0,
                    EnergySource.HYDRO: 12.0,
                    EnergySource.SOLAR: 8.0,
                    EnergySource.BIOMASS: 4.0,
                    EnergySource.OTHER_RENEWABLE: 1.0,
                },
                carbon_intensity_gco2_kwh=275.0,
                last_updated=datetime(2023, 1, 1),
                data_source="Eurostat 2023",
            ),
            "CN": EnergyMix(
                region="China",
                country_code="CHN",
                sources={
                    EnergySource.COAL: 57.0,
                    EnergySource.NATURAL_GAS: 8.0,
                    EnergySource.NUCLEAR: 5.0,
                    EnergySource.HYDRO: 18.0,
                    EnergySource.WIND: 8.0,
                    EnergySource.SOLAR: 3.0,
                    EnergySource.OTHER_RENEWABLE: 1.0,
                },
                carbon_intensity_gco2_kwh=555.0,
                last_updated=datetime(2023, 1, 1),
                data_source="China Energy Portal 2023",
            ),
            "IN": EnergyMix(
                region="India",
                country_code="IND",
                sources={
                    EnergySource.COAL: 70.0,
                    EnergySource.NATURAL_GAS: 3.0,
                    EnergySource.NUCLEAR: 3.0,
                    EnergySource.HYDRO: 17.0,
                    EnergySource.WIND: 4.0,
                    EnergySource.SOLAR: 2.0,
                    EnergySource.OTHER_RENEWABLE: 1.0,
                },
                carbon_intensity_gco2_kwh=632.0,
                last_updated=datetime(2023, 1, 1),
                data_source="CEA India 2023",
            ),
            "BR": EnergyMix(
                region="Brazil",
                country_code="BRA",
                sources={
                    EnergySource.HYDRO: 65.0,
                    EnergySource.NATURAL_GAS: 8.0,
                    EnergySource.WIND: 9.0,
                    EnergySource.BIOMASS: 8.0,
                    EnergySource.NUCLEAR: 3.0,
                    EnergySource.SOLAR: 2.0,
                    EnergySource.COAL: 3.0,
                    EnergySource.OIL: 2.0,
                },
                carbon_intensity_gco2_kwh=85.0,
                last_updated=datetime(2023, 1, 1),
                data_source="EPE Brazil 2023",
            ),
            "NO": EnergyMix(
                region="Norway",
                country_code="NOR",
                sources={
                    EnergySource.HYDRO: 96.0,
                    EnergySource.WIND: 3.0,
                    EnergySource.NATURAL_GAS: 1.0,
                },
                carbon_intensity_gco2_kwh=17.0,
                last_updated=datetime(2023, 1, 1),
                data_source="Statistics Norway 2023",
            ),
            "FR": EnergyMix(
                region="France",
                country_code="FRA",
                sources={
                    EnergySource.NUCLEAR: 70.0,
                    EnergySource.HYDRO: 12.0,
                    EnergySource.WIND: 8.0,
                    EnergySource.SOLAR: 3.0,
                    EnergySource.NATURAL_GAS: 4.0,
                    EnergySource.COAL: 1.0,
                    EnergySource.BIOMASS: 2.0,
                },
                carbon_intensity_gco2_kwh=57.0,
                last_updated=datetime(2023, 1, 1),
                data_source="RTE France 2023",
            ),
        }

        self.energy_mixes.update(default_mixes)
        logger.info(f"Loaded {len(default_mixes)} default energy mix profiles")

    def _load_custom_data(self, data_path: str):
        """Load custom energy mix data from file."""
        try:
            with open(data_path, "r") as f:
                custom_data = json.load(f)

            for region_code, mix_data in custom_data.items():
                # Convert sources back to enum
                sources = {
                    EnergySource(source): percentage
                    for source, percentage in mix_data["sources"].items()
                }

                energy_mix = EnergyMix(
                    region=mix_data["region"],
                    country_code=mix_data["country_code"],
                    sources=sources,
                    carbon_intensity_gco2_kwh=mix_data["carbon_intensity_gco2_kwh"],
                    last_updated=datetime.fromisoformat(mix_data["last_updated"]),
                    data_source=mix_data.get("data_source", "custom"),
                )

                self.energy_mixes[region_code] = energy_mix

            logger.info(f"Loaded {len(custom_data)} custom energy mix profiles")

        except Exception as e:
            logger.error(f"Failed to load custom energy mix data: {e}")

    def get_energy_mix(self, region_code: str) -> Optional[EnergyMix]:
        """Get energy mix for a specific region."""
        return self.energy_mixes.get(region_code.upper())

    def list_available_regions(self) -> List[str]:
        """List all available region codes."""
        return list(self.energy_mixes.keys())

    def get_carbon_intensity(self, region_code: str) -> float:
        """Get carbon intensity for a region in gCO2/kWh."""
        energy_mix = self.get_energy_mix(region_code)
        if energy_mix:
            return energy_mix.carbon_intensity_gco2_kwh
        else:
            logger.warning(f"Region {region_code} not found, using global average")
            return 475.0  # Global average carbon intensity


class CarbonCalculator:
    """Main carbon footprint calculator."""

    def __init__(self, config: Optional[CarbonFootprintConfig] = None):
        self.config = config or CarbonFootprintConfig()
        self.energy_mix_db = EnergyMixDatabase(self.config.energy_mix_data_path)
        self.carbon_history = []

        logger.info("Carbon calculator initialized")

    def calculate_carbon_footprint(
        self, energy_report, region: Optional[str] = None
    ) -> CarbonFootprint:
        """Calculate carbon footprint from energy consumption report."""

        region = region or self.config.default_region
        energy_mix = self.energy_mix_db.get_energy_mix(region)
        carbon_intensity = self.energy_mix_db.get_carbon_intensity(region)

        # Calculate operational emissions (direct energy use)
        operational_emissions_kg = (
            energy_report.total_energy_kwh * carbon_intensity
        ) / 1000

        # Calculate embodied emissions (hardware manufacturing)
        embodied_emissions_kg = 0.0
        if self.config.include_embodied_carbon:
            embodied_emissions_kg = (
                operational_emissions_kg * self.config.embodied_carbon_factor
            )

        # Total emissions
        total_emissions_kg = operational_emissions_kg + embodied_emissions_kg

        # Calculate equivalent metrics
        equivalent_metrics = self._calculate_equivalent_metrics(total_emissions_kg)

        # Calculate offset cost
        offset_cost_usd = None
        if self.config.enable_offset_calculation:
            offset_cost_usd = (
                total_emissions_kg / 1000
            ) * self.config.offset_price_per_ton_co2

        carbon_footprint = CarbonFootprint(
            experiment_id=energy_report.experiment_id,
            timestamp=energy_report.end_time,
            energy_kwh=energy_report.total_energy_kwh,
            operational_emissions_kg=operational_emissions_kg,
            embodied_emissions_kg=embodied_emissions_kg,
            total_emissions_kg=total_emissions_kg,
            region=region,
            carbon_intensity_gco2_kwh=carbon_intensity,
            energy_mix=energy_mix,
            equivalent_metrics=equivalent_metrics,
            offset_cost_usd=offset_cost_usd,
        )

        # Store in history
        self.carbon_history.append(carbon_footprint)

        # Log calculation
        audit_logger.log_model_operation(
            user_id="system",
            model_id="carbon_calculator",
            operation="calculate_carbon_footprint",
            success=True,
            details={
                "experiment_id": energy_report.experiment_id,
                "region": region,
                "total_emissions_kg": total_emissions_kg,
                "carbon_intensity": carbon_intensity,
            },
        )

        logger.info(
            f"Carbon footprint calculated: {total_emissions_kg:.6f} kg CO2e "
            f"for {energy_report.total_energy_kwh:.6f} kWh"
        )

        return carbon_footprint

    def _calculate_equivalent_metrics(self, emissions_kg: float) -> Dict[str, float]:
        """Calculate equivalent metrics for better understanding."""

        # Conversion factors (approximate)
        equivalents = {
            "km_driven_gasoline_car": emissions_kg / 0.251,  # kg CO2/km for average car
            "km_driven_electric_car": emissions_kg
            / 0.053,  # kg CO2/km for electric car
            "hours_laptop_use": emissions_kg / 0.0086,  # kg CO2/hour for laptop
            "smartphone_charges": emissions_kg / 0.0084,  # kg CO2 per charge
            "trees_needed_annual": emissions_kg
            / 21.77,  # kg CO2 absorbed per tree per year
            "coal_burned_kg": emissions_kg / 2.86,  # kg CO2 per kg coal
            "natural_gas_m3": emissions_kg / 2.03,  # kg CO2 per m3 natural gas
        }

        return equivalents

    def track_carbon_budget(self, budget_period: str = "monthly") -> CarbonBudgetStatus:
        """Track carbon budget usage and generate status report."""

        # Get budget limit
        budget_limit_kg = self._get_budget_limit(budget_period)
        if budget_limit_kg is None:
            raise ValueError(f"No {budget_period} carbon budget configured")

        # Calculate period boundaries
        period_start, period_end = self._get_period_boundaries(budget_period)

        # Calculate current usage in period
        current_usage_kg = self._calculate_period_usage(period_start, period_end)

        # Calculate remaining budget and usage percentage
        remaining_budget_kg = max(0, budget_limit_kg - current_usage_kg)
        usage_percentage = (current_usage_kg / budget_limit_kg) * 100

        # Calculate days remaining in period
        days_remaining = (period_end - datetime.now()).days

        # Determine alert level
        is_over_budget = current_usage_kg > budget_limit_kg
        if is_over_budget or usage_percentage >= self.config.critical_threshold * 100:
            alert_level = "critical"
        elif usage_percentage >= self.config.warning_threshold * 100:
            alert_level = "warning"
        else:
            alert_level = "normal"

        budget_status = CarbonBudgetStatus(
            budget_period=budget_period,
            budget_limit_kg=budget_limit_kg,
            current_usage_kg=current_usage_kg,
            remaining_budget_kg=remaining_budget_kg,
            usage_percentage=usage_percentage,
            period_start=period_start,
            period_end=period_end,
            days_remaining=days_remaining,
            is_over_budget=is_over_budget,
            alert_level=alert_level,
        )

        # Log budget status
        audit_logger.log_model_operation(
            user_id="system",
            model_id="carbon_calculator",
            operation="track_carbon_budget",
            success=True,
            details={
                "budget_period": budget_period,
                "usage_percentage": usage_percentage,
                "alert_level": alert_level,
                "is_over_budget": is_over_budget,
            },
        )

        # Generate alerts if necessary
        if alert_level in ["warning", "critical"]:
            self._generate_budget_alert(budget_status)

        return budget_status

    def _get_budget_limit(self, budget_period: str) -> Optional[float]:
        """Get budget limit for the specified period."""
        if budget_period == "daily":
            return self.config.daily_carbon_budget_kg
        elif budget_period == "monthly":
            return self.config.monthly_carbon_budget_kg
        elif budget_period == "annual":
            return self.config.annual_carbon_budget_kg
        else:
            return None

    def _get_period_boundaries(self, budget_period: str) -> Tuple[datetime, datetime]:
        """Get start and end dates for the budget period."""
        now = datetime.now()

        if budget_period == "daily":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif budget_period == "monthly":
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)
        elif budget_period == "annual":
            start = now.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            end = start.replace(year=now.year + 1)
        else:
            raise ValueError(f"Invalid budget period: {budget_period}")

        return start, end

    def _calculate_period_usage(
        self, period_start: datetime, period_end: datetime
    ) -> float:
        """Calculate carbon usage within the specified period."""
        period_usage = 0.0

        for footprint in self.carbon_history:
            if period_start <= footprint.timestamp < period_end:
                period_usage += footprint.total_emissions_kg

        return period_usage

    def _generate_budget_alert(self, budget_status: CarbonBudgetStatus):
        """Generate budget alert notification."""
        alert_message = (
            f"Carbon Budget Alert ({budget_status.alert_level.upper()}): "
            f"{budget_status.usage_percentage:.1f}% of {budget_status.budget_period} "
            f"budget used ({budget_status.current_usage_kg:.3f} kg CO2e / "
            f"{budget_status.budget_limit_kg:.3f} kg CO2e)"
        )

        if budget_status.is_over_budget:
            alert_message += f" - OVER BUDGET by {abs(budget_status.remaining_budget_kg):.3f} kg CO2e"

        logger.warning(alert_message)

        # Here you could integrate with alerting systems (email, Slack, etc.)
        # For now, we just log the alert

    def get_carbon_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get carbon footprint trends over the specified period."""

        cutoff_date = datetime.now() - timedelta(days=days)
        recent_footprints = [
            fp for fp in self.carbon_history if fp.timestamp >= cutoff_date
        ]

        if not recent_footprints:
            return {
                "error": "No carbon footprint data available for the specified period"
            }

        # Calculate trends
        daily_emissions = {}
        for footprint in recent_footprints:
            date_key = footprint.timestamp.date()
            if date_key not in daily_emissions:
                daily_emissions[date_key] = 0.0
            daily_emissions[date_key] += footprint.total_emissions_kg

        # Calculate statistics
        total_emissions = sum(fp.total_emissions_kg for fp in recent_footprints)
        avg_daily_emissions = total_emissions / max(1, len(daily_emissions))

        # Find peak day
        peak_day = (
            max(daily_emissions.items(), key=lambda x: x[1])
            if daily_emissions
            else (None, 0)
        )

        # Calculate trend (simple linear regression)
        if len(daily_emissions) > 1:
            dates = list(daily_emissions.keys())
            emissions = list(daily_emissions.values())

            # Convert dates to numeric values for regression
            date_nums = [(d - dates[0]).days for d in dates]

            # Simple linear regression
            n = len(date_nums)
            sum_x = sum(date_nums)
            sum_y = sum(emissions)
            sum_xy = sum(x * y for x, y in zip(date_nums, emissions))
            sum_x2 = sum(x * x for x in date_nums)

            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                trend_direction = (
                    "increasing"
                    if slope > 0
                    else "decreasing" if slope < 0 else "stable"
                )
            else:
                trend_direction = "stable"
        else:
            trend_direction = "insufficient_data"

        return {
            "period_days": days,
            "total_experiments": len(recent_footprints),
            "total_emissions_kg": total_emissions,
            "avg_daily_emissions_kg": avg_daily_emissions,
            "peak_day": {
                "date": peak_day[0].isoformat() if peak_day[0] else None,
                "emissions_kg": peak_day[1],
            },
            "trend_direction": trend_direction,
            "daily_emissions": {
                date.isoformat(): emissions
                for date, emissions in daily_emissions.items()
            },
        }

    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare carbon footprints across multiple experiments."""

        # Find matching experiments
        matching_footprints = [
            fp for fp in self.carbon_history if fp.experiment_id in experiment_ids
        ]

        if not matching_footprints:
            return {"error": "No matching experiments found"}

        # Calculate comparison metrics
        comparison = {
            "experiments": {},
            "summary": {
                "total_experiments": len(matching_footprints),
                "total_emissions_kg": sum(
                    fp.total_emissions_kg for fp in matching_footprints
                ),
                "avg_emissions_kg": np.mean(
                    [fp.total_emissions_kg for fp in matching_footprints]
                ),
                "min_emissions_kg": min(
                    fp.total_emissions_kg for fp in matching_footprints
                ),
                "max_emissions_kg": max(
                    fp.total_emissions_kg for fp in matching_footprints
                ),
                "std_emissions_kg": np.std(
                    [fp.total_emissions_kg for fp in matching_footprints]
                ),
            },
        }

        # Add individual experiment details
        for footprint in matching_footprints:
            comparison["experiments"][footprint.experiment_id] = {
                "total_emissions_kg": footprint.total_emissions_kg,
                "operational_emissions_kg": footprint.operational_emissions_kg,
                "embodied_emissions_kg": footprint.embodied_emissions_kg,
                "energy_kwh": footprint.energy_kwh,
                "carbon_intensity": footprint.carbon_intensity_gco2_kwh,
                "region": footprint.region,
                "timestamp": footprint.timestamp.isoformat(),
                "offset_cost_usd": footprint.offset_cost_usd,
            }

        return comparison

    def save_carbon_report(self, footprint: CarbonFootprint) -> str:
        """Save detailed carbon footprint report."""

        if not self.config.save_detailed_reports:
            return None

        try:
            # Create output directory
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate comprehensive report
            report = {
                "carbon_footprint": footprint.to_dict(),
                "calculation_details": {
                    "methodology": "Operational emissions calculated from energy consumption and regional carbon intensity",
                    "embodied_carbon_included": self.config.include_embodied_carbon,
                    "embodied_carbon_factor": self.config.embodied_carbon_factor,
                    "carbon_intensity_source": (
                        footprint.energy_mix.data_source
                        if footprint.energy_mix
                        else "default"
                    ),
                    "calculation_timestamp": datetime.now().isoformat(),
                },
                "regional_context": (
                    footprint.energy_mix.to_dict() if footprint.energy_mix else None
                ),
                "equivalent_metrics": footprint.equivalent_metrics,
                "recommendations": self._generate_recommendations(footprint),
            }

            # Save report
            report_file = output_dir / f"{footprint.experiment_id}_carbon_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            logger.debug(f"Carbon report saved to {report_file}")
            return str(report_file)

        except Exception as e:
            logger.error(f"Failed to save carbon report: {e}")
            return None

    def _generate_recommendations(self, footprint: CarbonFootprint) -> List[str]:
        """Generate recommendations for reducing carbon footprint."""

        recommendations = []

        # Energy efficiency recommendations
        if footprint.energy_kwh > 0.01:  # More than 10 Wh
            recommendations.append(
                "Consider model optimization techniques like pruning, quantization, or knowledge distillation to reduce energy consumption"
            )

        # Regional recommendations
        if footprint.carbon_intensity_gco2_kwh > 400:  # High carbon intensity
            recommendations.append(
                f"Consider running experiments in regions with cleaner energy grids (current: {footprint.carbon_intensity_gco2_kwh:.0f} gCO2/kWh)"
            )

        # Training efficiency recommendations
        if footprint.total_emissions_kg > 0.1:  # More than 100g CO2e
            recommendations.extend(
                [
                    "Use mixed precision training to reduce computational requirements",
                    "Implement early stopping to avoid unnecessary training epochs",
                    "Consider using pre-trained models to reduce training time",
                ]
            )

        # Offset recommendations
        if footprint.offset_cost_usd and footprint.offset_cost_usd > 0.01:
            recommendations.append(
                f"Consider purchasing carbon offsets (estimated cost: ${footprint.offset_cost_usd:.2f}) to neutralize emissions"
            )

        return recommendations


# Utility functions for easy integration


def calculate_carbon_footprint_from_energy(
    energy_kwh: float,
    region: str = "US",
    config: Optional[CarbonFootprintConfig] = None,
) -> CarbonFootprint:
    """
    Calculate carbon footprint from energy consumption.

    Args:
        energy_kwh: Energy consumption in kWh
        region: Region code for carbon intensity lookup
        config: Carbon calculation configuration

    Returns:
        CarbonFootprint object
    """
    from .energy_tracker import EnergyReport

    # Create a minimal energy report
    energy_report = EnergyReport(
        experiment_id=f"energy_calc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        start_time=datetime.now() - timedelta(hours=1),
        end_time=datetime.now(),
        duration_seconds=3600,
        total_energy_kwh=energy_kwh,
        cpu_energy_kwh=energy_kwh * 0.7,  # Assume 70% CPU
        gpu_energy_kwh=energy_kwh * 0.3,  # Assume 30% GPU
    )

    calculator = CarbonCalculator(config)
    return calculator.calculate_carbon_footprint(energy_report, region)


def get_regional_carbon_intensity(region: str) -> float:
    """
    Get carbon intensity for a specific region.

    Args:
        region: Region code (e.g., "US", "EU", "CN")

    Returns:
        Carbon intensity in gCO2/kWh
    """
    db = EnergyMixDatabase()
    return db.get_carbon_intensity(region)


def compare_regional_impact(
    energy_kwh: float, regions: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compare carbon impact across different regions.

    Args:
        energy_kwh: Energy consumption in kWh
        regions: List of region codes to compare

    Returns:
        Dictionary with regional comparison data
    """
    comparison = {}

    for region in regions:
        footprint = calculate_carbon_footprint_from_energy(energy_kwh, region)
        comparison[region] = {
            "carbon_intensity_gco2_kwh": footprint.carbon_intensity_gco2_kwh,
            "total_emissions_kg": footprint.total_emissions_kg,
            "offset_cost_usd": footprint.offset_cost_usd,
        }

    return comparison


# Utility functions for easy integration


def calculate_carbon_footprint_from_energy(
    energy_kwh: float, region: str = "US", experiment_id: str = None
) -> CarbonFootprint:
    """
    Calculate carbon footprint directly from energy consumption.

    Args:
        energy_kwh: Energy consumption in kWh
        region: Region code for carbon intensity lookup
        experiment_id: Optional experiment identifier

    Returns:
        CarbonFootprint object with calculated emissions
    """
    from datetime import datetime

    from .energy_tracker import EnergyReport

    # Create a mock energy report
    experiment_id = (
        experiment_id or f"energy_calc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    energy_report = EnergyReport(
        experiment_id=experiment_id,
        start_time=datetime.now(),
        end_time=datetime.now(),
        duration_seconds=0,
        total_energy_kwh=energy_kwh,
        cpu_energy_kwh=energy_kwh * 0.7,  # Assume 70% CPU, 30% GPU
        gpu_energy_kwh=energy_kwh * 0.3,
    )

    # Calculate carbon footprint
    calculator = CarbonCalculator()
    return calculator.calculate_carbon_footprint(energy_report, region)


def get_regional_carbon_intensity(region: str) -> float:
    """
    Get carbon intensity for a specific region.

    Args:
        region: Region code (e.g., 'US', 'EU', 'CN')

    Returns:
        Carbon intensity in gCO2/kWh
    """
    db = EnergyMixDatabase()
    return db.get_carbon_intensity(region)


def compare_regional_impact(
    energy_kwh: float, regions: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compare carbon impact across different regions.

    Args:
        energy_kwh: Energy consumption in kWh
        regions: List of region codes to compare

    Returns:
        Dictionary with regional comparison data
    """
    comparison = {}

    for region in regions:
        footprint = calculate_carbon_footprint_from_energy(energy_kwh, region)
        comparison[region] = {
            "carbon_intensity_gco2_kwh": footprint.carbon_intensity_gco2_kwh,
            "total_emissions_kg": footprint.total_emissions_kg,
            "offset_cost_usd": footprint.offset_cost_usd,
        }

    return comparison
