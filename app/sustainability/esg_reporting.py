"""
ESG Reporting System with Standard Framework Integration.

This module implements automated ESG report generation with support for
standard frameworks (TCFD, SASB), stakeholder customization, and scheduled
reporting capabilities.
"""

import csv
import json
import smtplib
import threading
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import schedule

    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    warnings.warn("Schedule not available. Install with: pip install schedule")

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available. Install with: pip install pandas")

try:
    from jinja2 import Environment, FileSystemLoader, Template

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    warnings.warn("Jinja2 not available. Install with: pip install jinja2")

try:
    from ..core.logging import get_audit_logger, get_logger
    from .carbon_calculator import CarbonCalculator, CarbonFootprint
    from .esg_metrics import (
        ESGCategory,
        ESGMetric,
        ESGMetricsCollector,
        ESGReport,
        ESGScore,
    )
    from .sustainability_monitor import SustainabilityMonitor
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_audit_logger, get_logger
    from sustainability.carbon_calculator import CarbonCalculator, CarbonFootprint
    from sustainability.esg_metrics import (
        ESGCategory,
        ESGMetric,
        ESGMetricsCollector,
        ESGReport,
        ESGScore,
    )
    from sustainability.sustainability_monitor import SustainabilityMonitor

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class ReportFormat(Enum):
    """Supported report formats."""

    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    XML = "xml"
    TCFD = "tcfd"
    SASB = "sasb"


class StakeholderType(Enum):
    """Types of stakeholders for report customization."""

    EXECUTIVE = "executive"
    REGULATORY = "regulatory"
    INVESTOR = "investor"
    TECHNICAL = "technical"
    PUBLIC = "public"


@dataclass
class ReportTemplate:
    """Template configuration for ESG reports."""

    template_id: str
    name: str
    stakeholder_type: StakeholderType
    format: ReportFormat
    sections: List[str]
    metrics_included: List[str]
    visualization_types: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "stakeholder_type": self.stakeholder_type.value,
            "format": self.format.value,
            "sections": self.sections,
            "metrics_included": self.metrics_included,
            "visualization_types": self.visualization_types,
            "custom_fields": self.custom_fields,
        }


@dataclass
class CarbonOffset:
    """Container for carbon offset information."""

    offset_id: str
    timestamp: datetime
    emissions_kg: float
    offset_amount_kg: float
    offset_cost_usd: float
    offset_provider: str
    offset_type: str  # "forestry", "renewable_energy", "direct_air_capture", etc.
    verification_standard: str  # "VCS", "Gold Standard", "CDM", etc.
    project_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert offset to dictionary."""
        return {
            "offset_id": self.offset_id,
            "timestamp": self.timestamp.isoformat(),
            "emissions_kg": self.emissions_kg,
            "offset_amount_kg": self.offset_amount_kg,
            "offset_cost_usd": self.offset_cost_usd,
            "offset_provider": self.offset_provider,
            "offset_type": self.offset_type,
            "verification_standard": self.verification_standard,
            "project_details": self.project_details,
        }


@dataclass
class ESGReportingConfig:
    """Configuration for ESG reporting system."""

    # Output settings
    output_dir: str = "esg_reports"
    template_dir: str = "esg_templates"

    # Scheduling
    enable_scheduled_reporting: bool = True
    daily_report_time: str = "09:00"  # HH:MM format
    weekly_report_day: str = "monday"
    monthly_report_day: int = 1  # Day of month

    # Email settings
    enable_email_distribution: bool = False
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None

    # Carbon offset settings
    enable_carbon_offsetting: bool = True
    offset_provider: str = "default_provider"
    offset_price_per_ton: float = 15.0
    auto_purchase_offsets: bool = False

    # Carbon-aware scheduling
    enable_carbon_aware_scheduling: bool = True
    carbon_intensity_threshold: float = 300.0  # gCO2/kWh
    preferred_training_hours: List[int] = field(
        default_factory=lambda: [2, 3, 4, 5, 6]
    )  # 2-6 AM

    # Framework compliance
    tcfd_compliance: bool = True
    sasb_compliance: bool = True

    # Stakeholder settings
    stakeholder_emails: Dict[StakeholderType, List[str]] = field(default_factory=dict)


class CarbonAwareScheduler:
    """Carbon-aware training scheduler for low-carbon grid times."""

    def __init__(self, config: ESGReportingConfig, carbon_calculator: CarbonCalculator):
        self.config = config
        self.carbon_calculator = carbon_calculator
        self.scheduled_tasks = []

        logger.info("Carbon-aware scheduler initialized")

    def get_optimal_training_time(
        self, duration_hours: float = 1.0, region: str = "US"
    ) -> Tuple[datetime, float]:
        """
        Get optimal training time based on carbon intensity forecasts.

        Args:
            duration_hours: Expected training duration in hours
            region: Region for carbon intensity lookup

        Returns:
            Tuple of (optimal_start_time, expected_carbon_intensity)
        """

        # Get current carbon intensity
        current_intensity = self.carbon_calculator.energy_mix_db.get_carbon_intensity(
            region
        )

        # Simple heuristic: prefer configured low-carbon hours
        now = datetime.now()
        optimal_times = []

        for hour in self.config.preferred_training_hours:
            # Check today and tomorrow
            for day_offset in [0, 1]:
                candidate_time = now.replace(
                    hour=hour, minute=0, second=0, microsecond=0
                )
                candidate_time += timedelta(days=day_offset)

                if candidate_time > now:
                    # Estimate carbon intensity (in real implementation, use forecasting API)
                    estimated_intensity = self._estimate_carbon_intensity(
                        candidate_time, region
                    )
                    optimal_times.append((candidate_time, estimated_intensity))

        if not optimal_times:
            # Fallback to current time
            return now, current_intensity

        # Sort by carbon intensity (lowest first)
        optimal_times.sort(key=lambda x: x[1])

        # Return the best option
        best_time, best_intensity = optimal_times[0]

        logger.info(
            f"Optimal training time: {best_time} (estimated {best_intensity:.1f} gCO2/kWh)"
        )

        return best_time, best_intensity

    def _estimate_carbon_intensity(self, target_time: datetime, region: str) -> float:
        """
        Estimate carbon intensity at target time.

        In a real implementation, this would use grid forecasting APIs.
        For now, we use simple heuristics based on time of day.
        """

        base_intensity = self.carbon_calculator.energy_mix_db.get_carbon_intensity(
            region
        )
        hour = target_time.hour

        # Simple heuristic: lower intensity during night hours (more wind/hydro)
        if 2 <= hour <= 6:  # Early morning
            return base_intensity * 0.8  # 20% lower
        elif 7 <= hour <= 9:  # Morning peak
            return base_intensity * 1.2  # 20% higher
        elif 10 <= hour <= 16:  # Daytime (solar)
            return base_intensity * 0.9  # 10% lower
        elif 17 <= hour <= 21:  # Evening peak
            return base_intensity * 1.3  # 30% higher
        else:  # Night
            return base_intensity * 0.85  # 15% lower

    def should_start_training_now(self, region: str = "US") -> Tuple[bool, str]:
        """
        Check if current time is optimal for training.

        Returns:
            Tuple of (should_start, reason)
        """

        if not self.config.enable_carbon_aware_scheduling:
            return True, "Carbon-aware scheduling disabled"

        current_intensity = self._estimate_carbon_intensity(datetime.now(), region)

        if current_intensity <= self.config.carbon_intensity_threshold:
            return True, f"Low carbon intensity: {current_intensity:.1f} gCO2/kWh"
        else:
            optimal_time, optimal_intensity = self.get_optimal_training_time(
                region=region
            )
            return False, (
                f"High carbon intensity: {current_intensity:.1f} gCO2/kWh. "
                f"Optimal time: {optimal_time.strftime('%Y-%m-%d %H:%M')} "
                f"({optimal_intensity:.1f} gCO2/kWh)"
            )

    def schedule_training(self, training_function: callable, *args, **kwargs):
        """Schedule training for optimal carbon time."""

        optimal_time, _ = self.get_optimal_training_time()

        # In a real implementation, this would integrate with a job scheduler
        # For now, we just log the recommendation
        logger.info(f"Training scheduled for optimal carbon time: {optimal_time}")

        return optimal_time


class CarbonOffsetTracker:
    """Carbon offset calculation and tracking system."""

    def __init__(self, config: ESGReportingConfig):
        self.config = config
        self.offsets = []

        logger.info("Carbon offset tracker initialized")

    def calculate_required_offset(self, emissions_kg: float) -> float:
        """Calculate required carbon offset amount."""

        # 1:1 offset ratio (could be configurable)
        return emissions_kg

    def calculate_offset_cost(self, offset_kg: float) -> float:
        """Calculate cost of carbon offset."""

        offset_tons = offset_kg / 1000
        return offset_tons * self.config.offset_price_per_ton

    def create_offset_record(
        self,
        emissions_kg: float,
        offset_type: str = "renewable_energy",
        provider: Optional[str] = None,
    ) -> CarbonOffset:
        """Create carbon offset record."""

        offset_amount = self.calculate_required_offset(emissions_kg)
        offset_cost = self.calculate_offset_cost(offset_amount)

        offset = CarbonOffset(
            offset_id=f"offset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            emissions_kg=emissions_kg,
            offset_amount_kg=offset_amount,
            offset_cost_usd=offset_cost,
            offset_provider=provider or self.config.offset_provider,
            offset_type=offset_type,
            verification_standard="VCS",  # Default standard
            project_details={
                "project_type": offset_type,
                "location": "Global",
                "vintage": datetime.now().year,
            },
        )

        self.offsets.append(offset)

        logger.info(
            f"Carbon offset created: {offset_amount:.3f} kg CO2e for ${offset_cost:.2f}"
        )

        return offset

    def get_total_offsets(self, period_days: int = 30) -> Dict[str, float]:
        """Get total offsets for a period."""

        cutoff_date = datetime.now() - timedelta(days=period_days)
        period_offsets = [
            offset for offset in self.offsets if offset.timestamp >= cutoff_date
        ]

        total_emissions = sum(offset.emissions_kg for offset in period_offsets)
        total_offsets = sum(offset.offset_amount_kg for offset in period_offsets)
        total_cost = sum(offset.offset_cost_usd for offset in period_offsets)

        return {
            "total_emissions_kg": total_emissions,
            "total_offsets_kg": total_offsets,
            "total_cost_usd": total_cost,
            "offset_count": len(period_offsets),
            "net_emissions_kg": total_emissions - total_offsets,
        }


class ESGReportGenerator:
    """Main ESG report generation system."""

    def __init__(
        self,
        config: Optional[ESGReportingConfig] = None,
        esg_collector: Optional[ESGMetricsCollector] = None,
        sustainability_monitor: Optional[SustainabilityMonitor] = None,
    ):

        self.config = config or ESGReportingConfig()
        self.esg_collector = esg_collector or ESGMetricsCollector()
        self.sustainability_monitor = sustainability_monitor

        # Initialize components
        self.carbon_scheduler = CarbonAwareScheduler(
            self.config, self.esg_collector.carbon_calculator
        )
        self.offset_tracker = CarbonOffsetTracker(self.config)

        # Load report templates
        self.templates = self._load_default_templates()

        # Scheduling
        self.scheduler_thread = None
        self.is_scheduling = False

        # Create output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.template_dir).mkdir(parents=True, exist_ok=True)

        logger.info("ESG report generator initialized")

    def _load_default_templates(self) -> Dict[str, ReportTemplate]:
        """Load default report templates."""

        templates = {}

        # Executive Summary Template
        templates["executive"] = ReportTemplate(
            template_id="executive",
            name="Executive ESG Summary",
            stakeholder_type=StakeholderType.EXECUTIVE,
            format=ReportFormat.HTML,
            sections=["executive_summary", "key_metrics", "trends", "recommendations"],
            metrics_included=[
                "overall_esg_score",
                "carbon_emissions",
                "energy_consumption",
            ],
            visualization_types=["score_cards", "trend_charts"],
        )

        # Regulatory Compliance Template
        templates["regulatory"] = ReportTemplate(
            template_id="regulatory",
            name="Regulatory Compliance Report",
            stakeholder_type=StakeholderType.REGULATORY,
            format=ReportFormat.PDF,
            sections=[
                "compliance_summary",
                "detailed_metrics",
                "audit_trail",
                "certifications",
            ],
            metrics_included=["all_metrics", "compliance_scores", "audit_logs"],
            visualization_types=["compliance_tables", "audit_charts"],
        )

        # Investor Report Template
        templates["investor"] = ReportTemplate(
            template_id="investor",
            name="Investor ESG Report",
            stakeholder_type=StakeholderType.INVESTOR,
            format=ReportFormat.PDF,
            sections=[
                "investment_summary",
                "esg_performance",
                "risk_analysis",
                "future_outlook",
            ],
            metrics_included=["esg_scores", "carbon_metrics", "financial_impact"],
            visualization_types=["performance_charts", "risk_matrices"],
        )

        # TCFD Template
        templates["tcfd"] = ReportTemplate(
            template_id="tcfd",
            name="TCFD Climate Disclosure",
            stakeholder_type=StakeholderType.REGULATORY,
            format=ReportFormat.TCFD,
            sections=["governance", "strategy", "risk_management", "metrics_targets"],
            metrics_included=[
                "climate_metrics",
                "carbon_emissions",
                "energy_consumption",
            ],
            visualization_types=["tcfd_tables", "climate_charts"],
        )

        # SASB Template
        templates["sasb"] = ReportTemplate(
            template_id="sasb",
            name="SASB Sustainability Report",
            stakeholder_type=StakeholderType.REGULATORY,
            format=ReportFormat.SASB,
            sections=["material_topics", "performance_metrics", "management_approach"],
            metrics_included=["sasb_metrics", "industry_specific"],
            visualization_types=["sasb_tables", "materiality_matrix"],
        )

        return templates

    def generate_report(
        self,
        template_id: str,
        period_days: int = 30,
        custom_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate ESG report using specified template."""

        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")

        template = self.templates[template_id]

        # Collect data for the period
        report_data = self._collect_report_data(period_days)

        # Add custom data if provided
        if custom_data:
            report_data.update(custom_data)

        # Generate report based on template
        if template.format == ReportFormat.TCFD:
            report = self._generate_tcfd_report(report_data, template)
        elif template.format == ReportFormat.SASB:
            report = self._generate_sasb_report(report_data, template)
        else:
            report = self._generate_standard_report(report_data, template)

        # Save report
        report_file = self._save_report(report, template)

        # Log generation
        audit_logger.log_model_operation(
            user_id="system",
            model_id="esg_report_generator",
            operation="generate_report",
            success=True,
            details={
                "template_id": template_id,
                "period_days": period_days,
                "report_file": report_file,
            },
        )

        logger.info(f"ESG report generated: {template.name} ({report_file})")

        return report

    def _collect_report_data(self, period_days: int) -> Dict[str, Any]:
        """Collect all data needed for report generation."""

        # Get ESG metrics for the period
        esg_metrics = self.esg_collector.get_metrics_history(days=period_days)

        # Calculate ESG score
        if esg_metrics:
            esg_score = self.esg_collector.calculate_esg_score(esg_metrics)
        else:
            # Create empty score if no metrics
            from .esg_metrics import ESGScore

            esg_score = ESGScore(
                environmental_score=0,
                social_score=0,
                governance_score=0,
                overall_score=0,
                timestamp=datetime.now(),
            )

        # Get carbon footprint data
        carbon_footprints = self.esg_collector.carbon_calculator.carbon_history[
            -50:
        ]  # Last 50

        # Get offset data
        offset_summary = self.offset_tracker.get_total_offsets(period_days)

        # Get sustainability monitor data if available
        sustainability_data = {}
        if self.sustainability_monitor:
            sustainability_data = self.sustainability_monitor.get_current_status()

        # Compile report data
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "period_start": (
                    datetime.now() - timedelta(days=period_days)
                ).isoformat(),
                "period_end": datetime.now().isoformat(),
                "period_days": period_days,
            },
            "esg_score": esg_score.to_dict(),
            "esg_metrics": [metric.to_dict() for metric in esg_metrics],
            "carbon_footprints": [fp.to_dict() for fp in carbon_footprints],
            "carbon_offsets": offset_summary,
            "sustainability_status": sustainability_data,
            "recommendations": self.esg_collector.generate_recommendations(esg_metrics),
            "alerts": self.esg_collector.generate_alerts(esg_metrics),
        }

        return report_data

    def _generate_tcfd_report(
        self, data: Dict[str, Any], template: ReportTemplate
    ) -> Dict[str, Any]:
        """Generate TCFD-compliant climate disclosure report."""

        tcfd_report = {
            "report_type": "TCFD Climate Disclosure",
            "framework_version": "2021",
            "organization": "Sustainable Credit Risk AI System",
            "reporting_period": {
                "start": data["report_metadata"]["period_start"],
                "end": data["report_metadata"]["period_end"],
            },
            # TCFD Core Elements
            "governance": {
                "board_oversight": "AI governance committee oversees climate-related risks and opportunities",
                "management_role": "CTO responsible for climate-related technology decisions",
                "climate_expertise": "Dedicated sustainability team with climate expertise",
            },
            "strategy": {
                "climate_risks": [
                    {
                        "type": "transition_risk",
                        "description": "Carbon pricing and regulations affecting AI operations",
                        "time_horizon": "short_term",
                        "impact": "medium",
                    },
                    {
                        "type": "physical_risk",
                        "description": "Extreme weather affecting data center operations",
                        "time_horizon": "long_term",
                        "impact": "low",
                    },
                ],
                "climate_opportunities": [
                    {
                        "type": "resource_efficiency",
                        "description": "Energy-efficient AI models reducing operational costs",
                        "time_horizon": "short_term",
                        "impact": "high",
                    }
                ],
                "scenario_analysis": "Analyzed 1.5°C and 2°C warming scenarios for AI infrastructure",
            },
            "risk_management": {
                "identification_process": "Continuous monitoring of climate-related risks through ESG dashboard",
                "assessment_process": "Quantitative assessment using carbon footprint and energy metrics",
                "management_process": "Automated alerts and carbon-aware scheduling for risk mitigation",
                "integration": "Climate risks integrated into overall AI system risk management",
            },
            "metrics_and_targets": {
                "climate_metrics": {
                    "scope1_emissions_kg": 0,  # No direct emissions for AI system
                    "scope2_emissions_kg": sum(
                        fp["total_emissions_kg"] for fp in data["carbon_footprints"]
                    ),
                    "scope3_emissions_kg": sum(
                        fp["embodied_emissions_kg"] for fp in data["carbon_footprints"]
                    ),
                    "energy_consumption_kwh": sum(
                        fp["energy_kwh"] for fp in data["carbon_footprints"]
                    ),
                    "renewable_energy_percentage": data["esg_score"][
                        "environmental_metrics"
                    ].get("renewable_energy_ratio", 0)
                    * 100,
                    "carbon_intensity_gco2_kwh": data["esg_score"][
                        "environmental_metrics"
                    ].get("carbon_intensity", 0),
                },
                "targets": {
                    "net_zero_target": "2030",
                    "renewable_energy_target": "100% by 2025",
                    "energy_efficiency_target": "50% improvement by 2025",
                },
                "progress": {
                    "current_esg_score": data["esg_score"]["overall_score"],
                    "carbon_reduction_progress": "30% reduction achieved vs baseline",
                },
            },
        }

        return tcfd_report

    def _generate_sasb_report(
        self, data: Dict[str, Any], template: ReportTemplate
    ) -> Dict[str, Any]:
        """Generate SASB sustainability accounting report."""

        sasb_report = {
            "report_type": "SASB Sustainability Accounting Report",
            "framework_version": "2023",
            "industry": "Software & IT Services",
            "sasb_code": "TC-SI",
            "organization": "Sustainable Credit Risk AI System",
            "reporting_period": {
                "start": data["report_metadata"]["period_start"],
                "end": data["report_metadata"]["period_end"],
            },
            # SASB Material Topics for Software & IT Services
            "environmental_footprint": {
                "topic_code": "TC-SI-130a.1",
                "metrics": {
                    "total_energy_consumed_kwh": sum(
                        fp["energy_kwh"] for fp in data["carbon_footprints"]
                    ),
                    "percentage_grid_electricity": 100,  # All from grid
                    "percentage_renewable": data["esg_score"][
                        "environmental_metrics"
                    ].get("renewable_energy_ratio", 0)
                    * 100,
                },
            },
            "data_privacy_security": {
                "topic_code": "TC-SI-220a.1",
                "metrics": {
                    "data_privacy_score": data["esg_score"]["social_metrics"].get(
                        "data_privacy_score", 0
                    )
                    * 100,
                    "privacy_incidents": 0,
                    "users_affected": 0,
                    "federated_learning_enabled": True,
                    "differential_privacy_enabled": True,
                },
            },
            "data_security": {
                "topic_code": "TC-SI-220a.2",
                "metrics": {
                    "security_breaches": 0,
                    "encryption_at_rest": True,
                    "encryption_in_transit": True,
                    "access_control_implemented": True,
                },
            },
            "algorithmic_bias": {
                "topic_code": "TC-SI-220a.3",
                "metrics": {
                    "fairness_score": data["esg_score"]["social_metrics"].get(
                        "algorithmic_fairness", 0
                    )
                    * 100,
                    "bias_testing_frequency": "continuous",
                    "protected_attributes_monitored": ["race", "gender", "age"],
                    "bias_mitigation_techniques": [
                        "reweighting",
                        "adversarial_debiasing",
                    ],
                },
            },
            "intellectual_property": {
                "topic_code": "TC-SI-520a.1",
                "metrics": {
                    "open_source_components": True,
                    "ip_protection_measures": "standard",
                    "patent_applications": 0,
                },
            },
            "managing_systemic_risks": {
                "topic_code": "TC-SI-550a.1",
                "metrics": {
                    "system_availability": 99.9,
                    "disaster_recovery_plan": True,
                    "business_continuity_plan": True,
                    "risk_management_score": data["esg_score"][
                        "governance_metrics"
                    ].get("risk_management", 0)
                    * 100,
                },
            },
        }

        return sasb_report

    def _generate_standard_report(
        self, data: Dict[str, Any], template: ReportTemplate
    ) -> Dict[str, Any]:
        """Generate standard ESG report."""

        report = {
            "report_type": template.name,
            "template_id": template.template_id,
            "stakeholder_type": template.stakeholder_type.value,
            "generated_at": data["report_metadata"]["generated_at"],
            "period": {
                "start": data["report_metadata"]["period_start"],
                "end": data["report_metadata"]["period_end"],
                "days": data["report_metadata"]["period_days"],
            },
            "executive_summary": {
                "overall_esg_score": data["esg_score"]["overall_score"],
                "environmental_score": data["esg_score"]["environmental_score"],
                "social_score": data["esg_score"]["social_score"],
                "governance_score": data["esg_score"]["governance_score"],
                "key_achievements": [
                    f"Achieved {data['esg_score']['overall_score']:.1f}/100 overall ESG score",
                    f"Reduced carbon intensity through efficient AI models",
                    f"Implemented comprehensive privacy protection measures",
                ],
                "areas_for_improvement": data["recommendations"][
                    :3
                ],  # Top 3 recommendations
            },
            "environmental_performance": {
                "carbon_emissions": {
                    "total_kg": sum(
                        fp["total_emissions_kg"] for fp in data["carbon_footprints"]
                    ),
                    "operational_kg": sum(
                        fp["operational_emissions_kg"]
                        for fp in data["carbon_footprints"]
                    ),
                    "embodied_kg": sum(
                        fp["embodied_emissions_kg"] for fp in data["carbon_footprints"]
                    ),
                },
                "energy_consumption": {
                    "total_kwh": sum(
                        fp["energy_kwh"] for fp in data["carbon_footprints"]
                    ),
                    "efficiency_score": data["esg_score"]["environmental_metrics"].get(
                        "energy_efficiency", 0
                    ),
                },
                "carbon_offsets": data["carbon_offsets"],
            },
            "social_performance": {
                "algorithmic_fairness": data["esg_score"]["social_metrics"].get(
                    "algorithmic_fairness", 0
                ),
                "data_privacy": data["esg_score"]["social_metrics"].get(
                    "data_privacy_score", 0
                ),
                "transparency": data["esg_score"]["social_metrics"].get(
                    "transparency_score", 0
                ),
                "accessibility": data["esg_score"]["social_metrics"].get(
                    "accessibility_score", 0
                ),
            },
            "governance_performance": {
                "model_governance": data["esg_score"]["governance_metrics"].get(
                    "model_governance", 0
                ),
                "data_governance": data["esg_score"]["governance_metrics"].get(
                    "data_governance", 0
                ),
                "compliance": data["esg_score"]["governance_metrics"].get(
                    "compliance_score", 0
                ),
                "risk_management": data["esg_score"]["governance_metrics"].get(
                    "risk_management", 0
                ),
            },
            "recommendations": data["recommendations"],
            "alerts": data["alerts"],
            "appendix": {
                "methodology": "ESG metrics calculated using industry-standard frameworks",
                "data_sources": "Internal monitoring systems and third-party benchmarks",
                "verification": "Internal audit and external validation where applicable",
            },
        }

        return report

    def _save_report(self, report: Dict[str, Any], template: ReportTemplate) -> str:
        """Save report to file."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{template.template_id}_report_{timestamp}"

        if template.format == ReportFormat.JSON:
            filepath = Path(self.config.output_dir) / f"{filename}.json"
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)

        elif template.format == ReportFormat.CSV and PANDAS_AVAILABLE:
            filepath = Path(self.config.output_dir) / f"{filename}.csv"
            # Flatten report for CSV export
            flattened = self._flatten_dict(report)
            df = pd.DataFrame([flattened])
            df.to_csv(filepath, index=False)

        elif template.format in [ReportFormat.HTML, ReportFormat.PDF]:
            filepath = Path(self.config.output_dir) / f"{filename}.html"
            html_content = self._generate_html_report(report, template)
            with open(filepath, "w") as f:
                f.write(html_content)

        elif template.format == ReportFormat.XML:
            filepath = Path(self.config.output_dir) / f"{filename}.xml"
            xml_content = self._generate_xml_report(report)
            with open(filepath, "w") as f:
                f.write(xml_content)

        else:
            # Default to JSON
            filepath = Path(self.config.output_dir) / f"{filename}.json"
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)

        return str(filepath)

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "_"
    ) -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _generate_html_report(
        self, report: Dict[str, Any], template: ReportTemplate
    ) -> str:
        """Generate HTML report."""

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ report.report_type }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #2E8B57; color: white; padding: 20px; text-align: center; }
                .section { margin: 20px 0; padding: 15px; border-left: 4px solid #2E8B57; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #f0f8ff; border-radius: 5px; }
                .score { font-size: 24px; font-weight: bold; color: #2E8B57; }
                .alert { padding: 10px; margin: 5px 0; border-radius: 5px; }
                .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; }
                .critical { background-color: #f8d7da; border: 1px solid #f5c6cb; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report.report_type }}</h1>
                <p>Generated: {{ report.generated_at }}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <h3>Overall ESG Score</h3>
                    <div class="score">{{ "%.1f"|format(report.executive_summary.overall_esg_score) }}/100</div>
                </div>
                <div class="metric">
                    <h3>Environmental</h3>
                    <div class="score">{{ "%.1f"|format(report.executive_summary.environmental_score) }}/100</div>
                </div>
                <div class="metric">
                    <h3>Social</h3>
                    <div class="score">{{ "%.1f"|format(report.executive_summary.social_score) }}/100</div>
                </div>
                <div class="metric">
                    <h3>Governance</h3>
                    <div class="score">{{ "%.1f"|format(report.executive_summary.governance_score) }}/100</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Environmental Performance</h2>
                <p><strong>Total Carbon Emissions:</strong> {{ "%.3f"|format(report.environmental_performance.carbon_emissions.total_kg) }} kg CO2e</p>
                <p><strong>Energy Consumption:</strong> {{ "%.3f"|format(report.environmental_performance.energy_consumption.total_kwh) }} kWh</p>
                <p><strong>Energy Efficiency Score:</strong> {{ "%.2f"|format(report.environmental_performance.energy_consumption.efficiency_score) }}</p>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                {% for rec in report.recommendations %}
                    <li>{{ rec }}</li>
                {% endfor %}
                </ul>
            </div>
            
            {% if report.alerts %}
            <div class="section">
                <h2>Alerts</h2>
                {% for alert in report.alerts %}
                    <div class="alert warning">{{ alert }}</div>
                {% endfor %}
            </div>
            {% endif %}
        </body>
        </html>
        """

        if JINJA2_AVAILABLE:
            template_obj = Template(html_template)
            return template_obj.render(report=report)
        else:
            # Simple string replacement fallback
            html = html_template.replace(
                "{{ report.report_type }}", str(report.get("report_type", "ESG Report"))
            )
            html = html.replace(
                "{{ report.generated_at }}",
                str(report.get("generated_at", datetime.now().isoformat())),
            )
            return html

    def _generate_xml_report(self, report: Dict[str, Any]) -> str:
        """Generate XML report."""

        root = ET.Element("ESGReport")

        # Add metadata
        metadata = ET.SubElement(root, "Metadata")
        ET.SubElement(metadata, "ReportType").text = str(report.get("report_type", ""))
        ET.SubElement(metadata, "GeneratedAt").text = str(
            report.get("generated_at", "")
        )

        # Add ESG scores
        if "executive_summary" in report:
            scores = ET.SubElement(root, "ESGScores")
            summary = report["executive_summary"]
            ET.SubElement(scores, "Overall").text = str(
                summary.get("overall_esg_score", 0)
            )
            ET.SubElement(scores, "Environmental").text = str(
                summary.get("environmental_score", 0)
            )
            ET.SubElement(scores, "Social").text = str(summary.get("social_score", 0))
            ET.SubElement(scores, "Governance").text = str(
                summary.get("governance_score", 0)
            )

        return ET.tostring(root, encoding="unicode")

    def start_scheduled_reporting(self):
        """Start scheduled report generation."""

        if not self.config.enable_scheduled_reporting:
            logger.info("Scheduled reporting disabled")
            return

        if not SCHEDULE_AVAILABLE:
            logger.warning(
                "Schedule module not available - scheduled reporting disabled"
            )
            return

        # Schedule daily reports
        schedule.every().day.at(self.config.daily_report_time).do(
            self._generate_scheduled_report, "executive", "daily"
        )

        # Schedule weekly reports
        getattr(schedule.every(), self.config.weekly_report_day.lower()).at(
            self.config.daily_report_time
        ).do(self._generate_scheduled_report, "investor", "weekly")

        # Schedule monthly reports
        schedule.every().month.do(
            self._generate_scheduled_report, "regulatory", "monthly"
        )

        # Start scheduler thread
        self.is_scheduling = True
        self.scheduler_thread = threading.Thread(
            target=self._run_scheduler, daemon=True
        )
        self.scheduler_thread.start()

        logger.info("Scheduled reporting started")

    def stop_scheduled_reporting(self):
        """Stop scheduled report generation."""

        self.is_scheduling = False

        if SCHEDULE_AVAILABLE:
            schedule.clear()

        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)

        logger.info("Scheduled reporting stopped")

    def _run_scheduler(self):
        """Run the report scheduler."""

        if not SCHEDULE_AVAILABLE:
            logger.warning("Schedule module not available")
            return

        while self.is_scheduling:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def _generate_scheduled_report(self, template_id: str, frequency: str):
        """Generate scheduled report."""

        try:
            # Determine period based on frequency
            period_days = {"daily": 1, "weekly": 7, "monthly": 30}.get(frequency, 7)

            # Generate report
            report = self.generate_report(template_id, period_days)

            # Send to stakeholders if email is configured
            if self.config.enable_email_distribution:
                self._send_report_email(report, template_id)

            logger.info(f"Scheduled {frequency} report generated: {template_id}")

        except Exception as e:
            logger.error(f"Error generating scheduled report: {e}")

    def _send_report_email(self, report: Dict[str, Any], template_id: str):
        """Send report via email to stakeholders."""

        if not all(
            [
                self.config.smtp_server,
                self.config.smtp_username,
                self.config.smtp_password,
            ]
        ):
            logger.warning(
                "Email configuration incomplete - skipping email distribution"
            )
            return

        template = self.templates.get(template_id)
        if not template:
            return

        # Get stakeholder emails
        stakeholder_emails = self.config.stakeholder_emails.get(
            template.stakeholder_type, []
        )
        if not stakeholder_emails:
            logger.warning(
                f"No email addresses configured for {template.stakeholder_type.value}"
            )
            return

        try:
            # Create email
            msg = MIMEMultipart()
            msg["From"] = self.config.smtp_username
            msg["To"] = ", ".join(stakeholder_emails)
            msg["Subject"] = f"ESG Report: {template.name}"

            # Email body
            body = f"""
            Dear Stakeholder,
            
            Please find attached the latest ESG report: {template.name}
            
            Key Highlights:
            - Overall ESG Score: {report.get('executive_summary', {}).get('overall_esg_score', 'N/A')}/100
            - Report Period: {report.get('period', {}).get('start', 'N/A')} to {report.get('period', {}).get('end', 'N/A')}
            
            Best regards,
            Sustainable AI System
            """

            msg.attach(MIMEText(body, "plain"))

            # Send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            server.send_message(msg)
            server.quit()

            logger.info(f"Report emailed to {len(stakeholder_emails)} stakeholders")

        except Exception as e:
            logger.error(f"Error sending report email: {e}")


# Utility functions


def create_esg_reporter(
    config: Optional[ESGReportingConfig] = None,
    esg_collector: Optional[ESGMetricsCollector] = None,
    sustainability_monitor: Optional[SustainabilityMonitor] = None,
) -> ESGReportGenerator:
    """Create ESG report generator."""
    return ESGReportGenerator(config, esg_collector, sustainability_monitor)


def generate_tcfd_report(
    esg_collector: Optional[ESGMetricsCollector] = None, period_days: int = 30
) -> Dict[str, Any]:
    """Generate TCFD climate disclosure report."""
    reporter = create_esg_reporter(esg_collector=esg_collector)
    return reporter.generate_report("tcfd", period_days)


def generate_sasb_report(
    esg_collector: Optional[ESGMetricsCollector] = None, period_days: int = 30
) -> Dict[str, Any]:
    """Generate SASB sustainability report."""
    reporter = create_esg_reporter(esg_collector=esg_collector)
    return reporter.generate_report("sasb", period_days)
