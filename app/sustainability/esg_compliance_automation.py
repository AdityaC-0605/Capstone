"""
ESG Compliance Automation System

This module implements automated ESG (Environmental, Social, Governance) compliance
reporting with TCFD (Task Force on Climate-related Financial Disclosures) and 
SASB (Sustainability Accounting Standards Board) standards.
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
import aiohttp
from jinja2 import Template

try:
    from ..core.logging import get_logger
    from ..sustainability.energy_tracker import EnergyTracker, EnergyReport
    from ..sustainability.carbon_calculator import CarbonCalculator, CarbonFootprint
    from ..sustainability.sustainability_monitor import SustainabilityMonitor
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.core.logging import get_logger
    from src.sustainability.energy_tracker import EnergyTracker, EnergyReport
    from src.sustainability.carbon_calculator import CarbonCalculator, CarbonFootprint
    from src.sustainability.sustainability_monitor import SustainabilityMonitor

logger = get_logger(__name__)


class ESGStandard(Enum):
    """ESG reporting standards."""
    TCFD = "tcfd"  # Task Force on Climate-related Financial Disclosures
    SASB = "sasb"  # Sustainability Accounting Standards Board
    GRI = "gri"    # Global Reporting Initiative
    CDP = "cdp"    # Carbon Disclosure Project


class ESGCategory(Enum):
    """ESG categories."""
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"


class ReportFrequency(Enum):
    """Report generation frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


@dataclass
class ESGComplianceConfig:
    """Configuration for ESG compliance automation."""
    # Reporting settings
    enable_automated_reporting: bool = True
    report_frequency: ReportFrequency = ReportFrequency.MONTHLY
    standards: List[ESGStandard] = field(default_factory=lambda: [ESGStandard.TCFD, ESGStandard.SASB])
    
    # Data collection settings
    data_retention_days: int = 365
    data_validation_enabled: bool = True
    data_quality_threshold: float = 0.95  # 95% data quality required
    
    # Alert settings
    enable_compliance_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "carbon_intensity_increase": 0.1,  # 10% increase triggers alert
        "energy_efficiency_decrease": 0.05,  # 5% decrease triggers alert
        "data_quality_below": 0.9  # Below 90% triggers alert
    })
    
    # Integration settings
    enable_external_verification: bool = True
    verification_providers: List[str] = field(default_factory=lambda: ["sustainalytics", "msci", "refinitiv"])
    
    # Output settings
    output_formats: List[str] = field(default_factory=lambda: ["json", "pdf", "excel"])
    output_directory: str = "esg_reports"


@dataclass
class ESGIndicator:
    """ESG indicator definition."""
    indicator_id: str
    name: str
    description: str
    category: ESGCategory
    standard: ESGStandard
    unit: str
    target_value: Optional[float] = None
    baseline_value: Optional[float] = None
    weight: float = 1.0  # Weight in ESG score calculation
    is_mandatory: bool = True
    data_sources: List[str] = field(default_factory=list)


@dataclass
class ESGDataPoint:
    """ESG data point."""
    indicator_id: str
    value: float
    timestamp: datetime
    source: str
    quality_score: float = 1.0
    verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ESGReport:
    """ESG compliance report."""
    report_id: str
    report_type: ESGStandard
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    company_info: Dict[str, Any]
    executive_summary: Dict[str, Any]
    indicators: Dict[str, ESGDataPoint]
    compliance_score: float
    recommendations: List[str]
    verification_status: str
    file_path: Optional[str] = None


class TCFDReportGenerator:
    """TCFD (Task Force on Climate-related Financial Disclosures) report generator."""
    
    def __init__(self, config: ESGComplianceConfig):
        self.config = config
        self.tcfd_indicators = self._initialize_tcfd_indicators()
        logger.info("TCFD report generator initialized")
    
    def _initialize_tcfd_indicators(self) -> List[ESGIndicator]:
        """Initialize TCFD-specific indicators."""
        return [
            # Governance
            ESGIndicator(
                indicator_id="tcfd_gov_01",
                name="Board Oversight of Climate Risks",
                description="Board-level oversight and management of climate-related risks",
                category=ESGCategory.GOVERNANCE,
                standard=ESGStandard.TCFD,
                unit="score",
                target_value=8.0,
                weight=1.0,
                is_mandatory=True
            ),
            ESGIndicator(
                indicator_id="tcfd_gov_02",
                name="Management Role in Climate Risk Assessment",
                description="Management's role in assessing and managing climate-related risks",
                category=ESGCategory.GOVERNANCE,
                standard=ESGStandard.TCFD,
                unit="score",
                target_value=8.0,
                weight=1.0,
                is_mandatory=True
            ),
            
            # Strategy
            ESGIndicator(
                indicator_id="tcfd_str_01",
                name="Climate-Related Risks and Opportunities",
                description="Climate-related risks and opportunities identified by the organization",
                category=ESGCategory.ENVIRONMENTAL,
                standard=ESGStandard.TCFD,
                unit="count",
                target_value=10.0,
                weight=1.0,
                is_mandatory=True
            ),
            ESGIndicator(
                indicator_id="tcfd_str_02",
                name="Impact on Business Strategy",
                description="Impact of climate-related risks and opportunities on business strategy",
                category=ESGCategory.ENVIRONMENTAL,
                standard=ESGStandard.TCFD,
                unit="score",
                target_value=7.0,
                weight=1.0,
                is_mandatory=True
            ),
            
            # Risk Management
            ESGIndicator(
                indicator_id="tcfd_risk_01",
                name="Climate Risk Assessment Process",
                description="Process for identifying, assessing, and managing climate-related risks",
                category=ESGCategory.GOVERNANCE,
                standard=ESGStandard.TCFD,
                unit="score",
                target_value=8.0,
                weight=1.0,
                is_mandatory=True
            ),
            ESGIndicator(
                indicator_id="tcfd_risk_02",
                name="Climate Risk Integration",
                description="Integration of climate-related risks into overall risk management",
                category=ESGCategory.GOVERNANCE,
                standard=ESGStandard.TCFD,
                unit="score",
                target_value=8.0,
                weight=1.0,
                is_mandatory=True
            ),
            
            # Metrics and Targets
            ESGIndicator(
                indicator_id="tcfd_met_01",
                name="Scope 1 Emissions",
                description="Direct greenhouse gas emissions",
                category=ESGCategory.ENVIRONMENTAL,
                standard=ESGStandard.TCFD,
                unit="tCO2e",
                target_value=100.0,
                weight=1.0,
                is_mandatory=True
            ),
            ESGIndicator(
                indicator_id="tcfd_met_02",
                name="Scope 2 Emissions",
                description="Indirect greenhouse gas emissions from purchased energy",
                category=ESGCategory.ENVIRONMENTAL,
                standard=ESGStandard.TCFD,
                unit="tCO2e",
                target_value=200.0,
                weight=1.0,
                is_mandatory=True
            ),
            ESGIndicator(
                indicator_id="tcfd_met_03",
                name="Scope 3 Emissions",
                description="Other indirect greenhouse gas emissions",
                category=ESGCategory.ENVIRONMENTAL,
                standard=ESGStandard.TCFD,
                unit="tCO2e",
                target_value=500.0,
                weight=1.0,
                is_mandatory=True
            ),
            ESGIndicator(
                indicator_id="tcfd_met_04",
                name="Energy Consumption",
                description="Total energy consumption",
                category=ESGCategory.ENVIRONMENTAL,
                standard=ESGStandard.TCFD,
                unit="MWh",
                target_value=1000.0,
                weight=1.0,
                is_mandatory=True
            )
        ]
    
    def generate_report(self, data_points: Dict[str, ESGDataPoint], 
                       company_info: Dict[str, Any]) -> ESGReport:
        """Generate TCFD compliance report."""
        try:
            report_id = f"tcfd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            period_end = datetime.now()
            period_start = period_end - timedelta(days=30)  # Monthly report
            
            # Calculate compliance score
            compliance_score = self._calculate_tcfd_compliance_score(data_points)
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(data_points, compliance_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(data_points, compliance_score)
            
            # Determine verification status
            verification_status = self._determine_verification_status(data_points)
            
            report = ESGReport(
                report_id=report_id,
                report_type=ESGStandard.TCFD,
                period_start=period_start,
                period_end=period_end,
                generated_at=datetime.now(),
                company_info=company_info,
                executive_summary=executive_summary,
                indicators=data_points,
                compliance_score=compliance_score,
                recommendations=recommendations,
                verification_status=verification_status
            )
            
            logger.info(f"TCFD report generated: {report_id}, compliance score: {compliance_score:.2f}")
            return report
            
        except Exception as e:
            logger.error(f"TCFD report generation failed: {e}")
            raise
    
    def _calculate_tcfd_compliance_score(self, data_points: Dict[str, ESGDataPoint]) -> float:
        """Calculate TCFD compliance score."""
        total_score = 0.0
        total_weight = 0.0
        
        for indicator in self.tcfd_indicators:
            if indicator.indicator_id in data_points:
                data_point = data_points[indicator.indicator_id]
                
                # Calculate score based on target achievement
                if indicator.target_value is not None:
                    if indicator.indicator_id.startswith("tcfd_met"):  # Metrics - lower is better
                        score = max(0, 1 - (data_point.value / indicator.target_value))
                    else:  # Other indicators - higher is better
                        score = min(1, data_point.value / indicator.target_value)
                else:
                    score = 0.5  # Default score if no target
                
                # Apply quality score
                score *= data_point.quality_score
                
                total_score += score * indicator.weight
                total_weight += indicator.weight
        
        return total_score / max(1, total_weight) * 100  # Convert to percentage
    
    def _generate_executive_summary(self, data_points: Dict[str, ESGDataPoint], 
                                  compliance_score: float) -> Dict[str, Any]:
        """Generate executive summary for TCFD report."""
        # Calculate key metrics
        scope1_emissions = data_points.get("tcfd_met_01", ESGDataPoint("", 0, datetime.now(), "")).value
        scope2_emissions = data_points.get("tcfd_met_02", ESGDataPoint("", 0, datetime.now(), "")).value
        scope3_emissions = data_points.get("tcfd_met_03", ESGDataPoint("", 0, datetime.now(), "")).value
        total_emissions = scope1_emissions + scope2_emissions + scope3_emissions
        
        energy_consumption = data_points.get("tcfd_met_04", ESGDataPoint("", 0, datetime.now(), "")).value
        
        return {
            "compliance_score": compliance_score,
            "compliance_level": self._get_compliance_level(compliance_score),
            "total_emissions_tco2e": total_emissions,
            "energy_consumption_mwh": energy_consumption,
            "emissions_intensity": total_emissions / max(1, energy_consumption),
            "key_achievements": [
                f"TCFD compliance score: {compliance_score:.1f}%",
                f"Total emissions: {total_emissions:.1f} tCO2e",
                f"Energy consumption: {energy_consumption:.1f} MWh"
            ],
            "areas_for_improvement": self._identify_improvement_areas(data_points)
        }
    
    def _generate_recommendations(self, data_points: Dict[str, ESGDataPoint], 
                                compliance_score: float) -> List[str]:
        """Generate recommendations based on TCFD compliance."""
        recommendations = []
        
        if compliance_score < 70:
            recommendations.append("Implement comprehensive climate risk assessment framework")
            recommendations.append("Establish board-level climate governance structure")
        
        # Check specific indicators
        scope1_emissions = data_points.get("tcfd_met_01", ESGDataPoint("", 0, datetime.now(), "")).value
        if scope1_emissions > 150:  # Above target
            recommendations.append("Reduce direct emissions through energy efficiency measures")
        
        scope2_emissions = data_points.get("tcfd_met_02", ESGDataPoint("", 0, datetime.now(), "")).value
        if scope2_emissions > 250:  # Above target
            recommendations.append("Transition to renewable energy sources")
        
        scope3_emissions = data_points.get("tcfd_met_03", ESGDataPoint("", 0, datetime.now(), "")).value
        if scope3_emissions > 600:  # Above target
            recommendations.append("Engage with supply chain partners to reduce indirect emissions")
        
        return recommendations
    
    def _get_compliance_level(self, score: float) -> str:
        """Get compliance level based on score."""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Satisfactory"
        elif score >= 60:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def _identify_improvement_areas(self, data_points: Dict[str, ESGDataPoint]) -> List[str]:
        """Identify areas for improvement."""
        improvement_areas = []
        
        for indicator in self.tcfd_indicators:
            if indicator.indicator_id in data_points:
                data_point = data_points[indicator.indicator_id]
                if data_point.quality_score < 0.8:
                    improvement_areas.append(f"Improve data quality for {indicator.name}")
        
        return improvement_areas
    
    def _determine_verification_status(self, data_points: Dict[str, ESGDataPoint]) -> str:
        """Determine verification status based on data quality."""
        verified_count = sum(1 for dp in data_points.values() if dp.verified)
        total_count = len(data_points)
        
        if verified_count == total_count:
            return "Fully Verified"
        elif verified_count >= total_count * 0.8:
            return "Partially Verified"
        else:
            return "Unverified"


class SASBReportGenerator:
    """SASB (Sustainability Accounting Standards Board) report generator."""
    
    def __init__(self, config: ESGComplianceConfig):
        self.config = config
        self.sasb_indicators = self._initialize_sasb_indicators()
        logger.info("SASB report generator initialized")
    
    def _initialize_sasb_indicators(self) -> List[ESGIndicator]:
        """Initialize SASB-specific indicators for financial services."""
        return [
            # Environmental
            ESGIndicator(
                indicator_id="sasb_env_01",
                name="Greenhouse Gas Emissions",
                description="Total greenhouse gas emissions from operations",
                category=ESGCategory.ENVIRONMENTAL,
                standard=ESGStandard.SASB,
                unit="tCO2e",
                target_value=300.0,
                weight=1.0,
                is_mandatory=True
            ),
            ESGIndicator(
                indicator_id="sasb_env_02",
                name="Energy Management",
                description="Energy consumption and renewable energy usage",
                category=ESGCategory.ENVIRONMENTAL,
                standard=ESGStandard.SASB,
                unit="MWh",
                target_value=1200.0,
                weight=1.0,
                is_mandatory=True
            ),
            ESGIndicator(
                indicator_id="sasb_env_03",
                name="Water Management",
                description="Water consumption and efficiency",
                category=ESGCategory.ENVIRONMENTAL,
                standard=ESGStandard.SASB,
                unit="m3",
                target_value=5000.0,
                weight=0.8,
                is_mandatory=False
            ),
            
            # Social
            ESGIndicator(
                indicator_id="sasb_soc_01",
                name="Employee Diversity",
                description="Workforce diversity and inclusion metrics",
                category=ESGCategory.SOCIAL,
                standard=ESGStandard.SASB,
                unit="percentage",
                target_value=40.0,  # 40% diverse representation
                weight=1.0,
                is_mandatory=True
            ),
            ESGIndicator(
                indicator_id="sasb_soc_02",
                name="Employee Training",
                description="Employee training and development hours",
                category=ESGCategory.SOCIAL,
                standard=ESGStandard.SASB,
                unit="hours_per_employee",
                target_value=40.0,
                weight=0.8,
                is_mandatory=False
            ),
            ESGIndicator(
                indicator_id="sasb_soc_03",
                name="Customer Privacy",
                description="Data privacy and security incidents",
                category=ESGCategory.SOCIAL,
                standard=ESGStandard.SASB,
                unit="incidents",
                target_value=0.0,  # Zero incidents target
                weight=1.0,
                is_mandatory=True
            ),
            
            # Governance
            ESGIndicator(
                indicator_id="sasb_gov_01",
                name="Board Independence",
                description="Percentage of independent board members",
                category=ESGCategory.GOVERNANCE,
                standard=ESGStandard.SASB,
                unit="percentage",
                target_value=75.0,
                weight=1.0,
                is_mandatory=True
            ),
            ESGIndicator(
                indicator_id="sasb_gov_02",
                name="Ethics and Compliance",
                description="Ethics and compliance training completion rate",
                category=ESGCategory.GOVERNANCE,
                standard=ESGStandard.SASB,
                unit="percentage",
                target_value=100.0,
                weight=1.0,
                is_mandatory=True
            ),
            ESGIndicator(
                indicator_id="sasb_gov_03",
                name="Risk Management",
                description="Risk management framework effectiveness",
                category=ESGCategory.GOVERNANCE,
                standard=ESGStandard.SASB,
                unit="score",
                target_value=8.0,
                weight=1.0,
                is_mandatory=True
            )
        ]
    
    def generate_report(self, data_points: Dict[str, ESGDataPoint], 
                       company_info: Dict[str, Any]) -> ESGReport:
        """Generate SASB compliance report."""
        try:
            report_id = f"sasb_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            period_end = datetime.now()
            period_start = period_end - timedelta(days=30)  # Monthly report
            
            # Calculate compliance score
            compliance_score = self._calculate_sasb_compliance_score(data_points)
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(data_points, compliance_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(data_points, compliance_score)
            
            # Determine verification status
            verification_status = self._determine_verification_status(data_points)
            
            report = ESGReport(
                report_id=report_id,
                report_type=ESGStandard.SASB,
                period_start=period_start,
                period_end=period_end,
                generated_at=datetime.now(),
                company_info=company_info,
                executive_summary=executive_summary,
                indicators=data_points,
                compliance_score=compliance_score,
                recommendations=recommendations,
                verification_status=verification_status
            )
            
            logger.info(f"SASB report generated: {report_id}, compliance score: {compliance_score:.2f}")
            return report
            
        except Exception as e:
            logger.error(f"SASB report generation failed: {e}")
            raise
    
    def _calculate_sasb_compliance_score(self, data_points: Dict[str, ESGDataPoint]) -> float:
        """Calculate SASB compliance score."""
        total_score = 0.0
        total_weight = 0.0
        
        for indicator in self.sasb_indicators:
            if indicator.indicator_id in data_points:
                data_point = data_points[indicator.indicator_id]
                
                # Calculate score based on target achievement
                if indicator.target_value is not None:
                    if indicator.indicator_id in ["sasb_soc_03"]:  # Incidents - lower is better
                        score = max(0, 1 - (data_point.value / max(1, indicator.target_value)))
                    else:  # Other indicators - higher is better
                        score = min(1, data_point.value / max(1, indicator.target_value))
                else:
                    score = 0.5  # Default score if no target
                
                # Apply quality score
                score *= data_point.quality_score
                
                total_score += score * indicator.weight
                total_weight += indicator.weight
        
        return total_score / max(1, total_weight) * 100  # Convert to percentage
    
    def _generate_executive_summary(self, data_points: Dict[str, ESGDataPoint], 
                                  compliance_score: float) -> Dict[str, Any]:
        """Generate executive summary for SASB report."""
        # Calculate key metrics
        ghg_emissions = data_points.get("sasb_env_01", ESGDataPoint("", 0, datetime.now(), "")).value
        energy_consumption = data_points.get("sasb_env_02", ESGDataPoint("", 0, datetime.now(), "")).value
        employee_diversity = data_points.get("sasb_soc_01", ESGDataPoint("", 0, datetime.now(), "")).value
        board_independence = data_points.get("sasb_gov_01", ESGDataPoint("", 0, datetime.now(), "")).value
        
        return {
            "compliance_score": compliance_score,
            "compliance_level": self._get_compliance_level(compliance_score),
            "ghg_emissions_tco2e": ghg_emissions,
            "energy_consumption_mwh": energy_consumption,
            "employee_diversity_percent": employee_diversity,
            "board_independence_percent": board_independence,
            "key_achievements": [
                f"SASB compliance score: {compliance_score:.1f}%",
                f"GHG emissions: {ghg_emissions:.1f} tCO2e",
                f"Employee diversity: {employee_diversity:.1f}%",
                f"Board independence: {board_independence:.1f}%"
            ],
            "areas_for_improvement": self._identify_improvement_areas(data_points)
        }
    
    def _generate_recommendations(self, data_points: Dict[str, ESGDataPoint], 
                                compliance_score: float) -> List[str]:
        """Generate recommendations based on SASB compliance."""
        recommendations = []
        
        if compliance_score < 70:
            recommendations.append("Strengthen ESG governance framework")
            recommendations.append("Implement comprehensive ESG risk management")
        
        # Check specific indicators
        ghg_emissions = data_points.get("sasb_env_01", ESGDataPoint("", 0, datetime.now(), "")).value
        if ghg_emissions > 350:  # Above target
            recommendations.append("Implement carbon reduction initiatives")
        
        employee_diversity = data_points.get("sasb_soc_01", ESGDataPoint("", 0, datetime.now(), "")).value
        if employee_diversity < 35:  # Below target
            recommendations.append("Enhance diversity and inclusion programs")
        
        board_independence = data_points.get("sasb_gov_01", ESGDataPoint("", 0, datetime.now(), "")).value
        if board_independence < 70:  # Below target
            recommendations.append("Increase board independence")
        
        return recommendations
    
    def _get_compliance_level(self, score: float) -> str:
        """Get compliance level based on score."""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Satisfactory"
        elif score >= 60:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def _identify_improvement_areas(self, data_points: Dict[str, ESGDataPoint]) -> List[str]:
        """Identify areas for improvement."""
        improvement_areas = []
        
        for indicator in self.sasb_indicators:
            if indicator.indicator_id in data_points:
                data_point = data_points[indicator.indicator_id]
                if data_point.quality_score < 0.8:
                    improvement_areas.append(f"Improve data quality for {indicator.name}")
        
        return improvement_areas
    
    def _determine_verification_status(self, data_points: Dict[str, ESGDataPoint]) -> str:
        """Determine verification status based on data quality."""
        verified_count = sum(1 for dp in data_points.values() if dp.verified)
        total_count = len(data_points)
        
        if verified_count == total_count:
            return "Fully Verified"
        elif verified_count >= total_count * 0.8:
            return "Partially Verified"
        else:
            return "Unverified"


class ESGComplianceAutomation:
    """Main ESG compliance automation system."""
    
    def __init__(self, config: Optional[ESGComplianceConfig] = None):
        self.config = config or ESGComplianceConfig()
        self.energy_tracker = EnergyTracker()
        self.carbon_calculator = CarbonCalculator()
        self.sustainability_monitor = SustainabilityMonitor()
        
        # Initialize report generators
        self.tcfd_generator = TCFDReportGenerator(self.config)
        self.sasb_generator = SASBReportGenerator(self.config)
        
        # Data storage
        self.esg_data: Dict[str, List[ESGDataPoint]] = {}
        self.generated_reports: List[ESGReport] = []
        
        # Create output directory
        Path(self.config.output_directory).mkdir(exist_ok=True)
        
        logger.info("ESG compliance automation system initialized")
    
    def collect_esg_data(self, indicator_id: str, value: float, 
                        source: str, quality_score: float = 1.0,
                        verified: bool = False, metadata: Dict[str, Any] = None) -> bool:
        """Collect ESG data point."""
        try:
            data_point = ESGDataPoint(
                indicator_id=indicator_id,
                value=value,
                timestamp=datetime.now(),
                source=source,
                quality_score=quality_score,
                verified=verified,
                metadata=metadata or {}
            )
            
            if indicator_id not in self.esg_data:
                self.esg_data[indicator_id] = []
            
            self.esg_data[indicator_id].append(data_point)
            
            # Validate data quality
            if self.config.data_validation_enabled:
                if quality_score < self.config.data_quality_threshold:
                    logger.warning(f"Low data quality for {indicator_id}: {quality_score:.2f}")
            
            logger.debug(f"ESG data collected: {indicator_id} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to collect ESG data for {indicator_id}: {e}")
            return False
    
    def generate_compliance_reports(self, company_info: Dict[str, Any]) -> List[ESGReport]:
        """Generate compliance reports for all enabled standards."""
        reports = []
        
        try:
            # Get latest data points for each indicator
            latest_data = self._get_latest_data_points()
            
            # Generate TCFD report
            if ESGStandard.TCFD in self.config.standards:
                tcfd_report = self.tcfd_generator.generate_report(latest_data, company_info)
                reports.append(tcfd_report)
                self._save_report(tcfd_report)
            
            # Generate SASB report
            if ESGStandard.SASB in self.config.standards:
                sasb_report = self.sasb_generator.generate_report(latest_data, company_info)
                reports.append(sasb_report)
                self._save_report(sasb_report)
            
            self.generated_reports.extend(reports)
            logger.info(f"Generated {len(reports)} compliance reports")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
        
        return reports
    
    def _get_latest_data_points(self) -> Dict[str, ESGDataPoint]:
        """Get the latest data point for each indicator."""
        latest_data = {}
        
        for indicator_id, data_points in self.esg_data.items():
            if data_points:
                # Get the most recent data point
                latest_data_point = max(data_points, key=lambda dp: dp.timestamp)
                latest_data[indicator_id] = latest_data_point
        
        return latest_data
    
    def _save_report(self, report: ESGReport) -> bool:
        """Save report to file."""
        try:
            # Save as JSON
            if "json" in self.config.output_formats:
                json_path = Path(self.config.output_directory) / f"{report.report_id}.json"
                with open(json_path, 'w') as f:
                    json.dump(self._report_to_dict(report), f, indent=2, default=str)
                report.file_path = str(json_path)
            
            logger.info(f"Report saved: {report.report_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save report {report.report_id}: {e}")
            return False
    
    def _report_to_dict(self, report: ESGReport) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "report_id": report.report_id,
            "report_type": report.report_type.value,
            "period_start": report.period_start.isoformat(),
            "period_end": report.period_end.isoformat(),
            "generated_at": report.generated_at.isoformat(),
            "company_info": report.company_info,
            "executive_summary": report.executive_summary,
            "indicators": {
                indicator_id: {
                    "value": dp.value,
                    "timestamp": dp.timestamp.isoformat(),
                    "source": dp.source,
                    "quality_score": dp.quality_score,
                    "verified": dp.verified,
                    "metadata": dp.metadata
                }
                for indicator_id, dp in report.indicators.items()
            },
            "compliance_score": report.compliance_score,
            "recommendations": report.recommendations,
            "verification_status": report.verification_status
        }
    
    def get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for ESG compliance dashboard."""
        try:
            latest_data = self._get_latest_data_points()
            
            # Calculate overall ESG score
            total_score = 0.0
            total_weight = 0.0
            
            for data_point in latest_data.values():
                # Simple scoring based on data quality and recency
                score = data_point.quality_score * 100
                weight = 1.0
                
                total_score += score * weight
                total_weight += weight
            
            overall_esg_score = total_score / max(1, total_weight)
            
            # Categorize by ESG pillars
            environmental_score = self._calculate_pillar_score(latest_data, ESGCategory.ENVIRONMENTAL)
            social_score = self._calculate_pillar_score(latest_data, ESGCategory.SOCIAL)
            governance_score = self._calculate_pillar_score(latest_data, ESGCategory.GOVERNANCE)
            
            return {
                "overall_esg_score": overall_esg_score,
                "environmental_score": environmental_score,
                "social_score": social_score,
                "governance_score": governance_score,
                "total_indicators": len(latest_data),
                "verified_indicators": sum(1 for dp in latest_data.values() if dp.verified),
                "data_quality_average": np.mean([dp.quality_score for dp in latest_data.values()]) if latest_data else 0,
                "latest_reports": [
                    {
                        "report_id": report.report_id,
                        "report_type": report.report_type.value,
                        "compliance_score": report.compliance_score,
                        "generated_at": report.generated_at.isoformat()
                    }
                    for report in self.generated_reports[-5:]  # Last 5 reports
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {"error": str(e)}
    
    def _calculate_pillar_score(self, data_points: Dict[str, ESGDataPoint], 
                               category: ESGCategory) -> float:
        """Calculate score for a specific ESG pillar."""
        pillar_data = [dp for dp in data_points.values() 
                      if any(indicator.category == category 
                            for indicator in self.tcfd_generator.tcfd_indicators + 
                            self.sasb_generator.sasb_indicators
                            if indicator.indicator_id == dp.indicator_id)]
        
        if not pillar_data:
            return 0.0
        
        return np.mean([dp.quality_score * 100 for dp in pillar_data])
    
    def check_compliance_alerts(self) -> List[Dict[str, Any]]:
        """Check for compliance alerts based on thresholds."""
        alerts = []
        
        try:
            latest_data = self._get_latest_data_points()
            
            # Check carbon intensity increase
            if "tcfd_met_01" in latest_data:
                current_emissions = latest_data["tcfd_met_01"].value
                # Compare with historical average (simplified)
                historical_avg = 200.0  # This would be calculated from historical data
                increase = (current_emissions - historical_avg) / max(1, historical_avg)
                
                if increase > self.config.alert_thresholds["carbon_intensity_increase"]:
                    alerts.append({
                        "type": "carbon_intensity_increase",
                        "severity": "high" if increase > 0.2 else "medium",
                        "message": f"Carbon emissions increased by {increase:.1%}",
                        "current_value": current_emissions,
                        "threshold": self.config.alert_thresholds["carbon_intensity_increase"],
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Check data quality
            avg_quality = np.mean([dp.quality_score for dp in latest_data.values()]) if latest_data else 1.0
            if avg_quality < self.config.alert_thresholds["data_quality_below"]:
                alerts.append({
                    "type": "data_quality_below",
                    "severity": "medium",
                    "message": f"Average data quality below threshold: {avg_quality:.2f}",
                    "current_value": avg_quality,
                    "threshold": self.config.alert_thresholds["data_quality_below"],
                    "timestamp": datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
        
        return alerts


# Utility functions
def create_esg_compliance_automation(config_dict: Optional[Dict[str, Any]] = None) -> ESGComplianceAutomation:
    """Create ESG compliance automation system with configuration."""
    if config_dict:
        config = ESGComplianceConfig(**config_dict)
    else:
        config = ESGComplianceConfig()
    return ESGComplianceAutomation(config)


def demo_esg_compliance_automation() -> Dict[str, Any]:
    """Demonstrate ESG compliance automation system."""
    logger.info("Starting ESG compliance automation demo")
    
    # Create ESG compliance system
    esg_system = create_esg_compliance_automation()
    
    # Sample company information
    company_info = {
        "name": "Sustainable AI Credit Corp",
        "sector": "Financial Services",
        "size": "Large",
        "location": "United States",
        "employees": 1000,
        "revenue": 1000000000  # $1B
    }
    
    # Collect sample ESG data
    sample_data = [
        # TCFD indicators
        ("tcfd_gov_01", 8.5, "board_assessment", 0.95, True),
        ("tcfd_gov_02", 7.8, "management_review", 0.90, True),
        ("tcfd_str_01", 12, "risk_assessment", 0.88, True),
        ("tcfd_str_02", 7.2, "strategy_analysis", 0.85, False),
        ("tcfd_risk_01", 8.0, "risk_framework", 0.92, True),
        ("tcfd_risk_02", 7.5, "integration_review", 0.87, True),
        ("tcfd_met_01", 95.5, "emissions_tracking", 0.94, True),
        ("tcfd_met_02", 185.2, "energy_procurement", 0.91, True),
        ("tcfd_met_03", 420.8, "supply_chain", 0.83, False),
        ("tcfd_met_04", 850.0, "energy_monitoring", 0.89, True),
        
        # SASB indicators
        ("sasb_env_01", 280.5, "emissions_inventory", 0.93, True),
        ("sasb_env_02", 1100.0, "energy_management", 0.90, True),
        ("sasb_env_03", 4200.0, "water_tracking", 0.85, False),
        ("sasb_soc_01", 42.5, "hr_analytics", 0.88, True),
        ("sasb_soc_02", 38.0, "training_records", 0.82, False),
        ("sasb_soc_03", 0, "security_monitoring", 0.95, True),
        ("sasb_gov_01", 78.0, "board_composition", 0.92, True),
        ("sasb_gov_02", 98.5, "compliance_tracking", 0.94, True),
        ("sasb_gov_03", 8.2, "risk_assessment", 0.89, True)
    ]
    
    # Collect all sample data
    for indicator_id, value, source, quality, verified in sample_data:
        esg_system.collect_esg_data(indicator_id, value, source, quality, verified)
    
    # Generate compliance reports
    reports = esg_system.generate_compliance_reports(company_info)
    
    # Get dashboard data
    dashboard_data = esg_system.get_compliance_dashboard_data()
    
    # Check for alerts
    alerts = esg_system.check_compliance_alerts()
    
    return {
        "company_info": company_info,
        "data_points_collected": len(sample_data),
        "reports_generated": len(reports),
        "reports": [
            {
                "report_id": report.report_id,
                "report_type": report.report_type.value,
                "compliance_score": report.compliance_score,
                "verification_status": report.verification_status,
                "recommendations_count": len(report.recommendations)
            }
            for report in reports
        ],
        "dashboard_data": dashboard_data,
        "alerts": alerts,
        "system_status": "operational"
    }
