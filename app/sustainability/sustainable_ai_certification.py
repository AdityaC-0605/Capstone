"""
Sustainable AI Certification and Validation Framework

This module implements a comprehensive certification system for sustainable AI models,
including validation criteria, scoring algorithms, and automated certification processes.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import uuid

try:
    from ..core.logging import get_logger
    from ..sustainability.energy_tracker import EnergyTracker, EnergyReport
    from ..sustainability.carbon_calculator import CarbonCalculator, CarbonFootprint
    from ..sustainability.sustainability_monitor import SustainabilityMonitor
    from ..sustainability.sustainable_model_lifecycle import (
        SustainableModelLifecycleManager,
    )
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.core.logging import get_logger
    from src.sustainability.energy_tracker import EnergyTracker, EnergyReport
    from src.sustainability.carbon_calculator import CarbonCalculator, CarbonFootprint
    from src.sustainability.sustainability_monitor import SustainabilityMonitor
    from src.sustainability.sustainable_model_lifecycle import (
        SustainableModelLifecycleManager,
    )

logger = get_logger(__name__)


class CertificationLevel(Enum):
    """AI sustainability certification levels."""

    BRONZE = "bronze"  # Basic sustainability compliance
    SILVER = "silver"  # Good sustainability practices
    GOLD = "gold"  # Excellent sustainability practices
    PLATINUM = "platinum"  # Industry-leading sustainability
    CARBON_NEUTRAL = "carbon_neutral"  # Carbon neutral operations


class ValidationCriteria(Enum):
    """Validation criteria for sustainable AI certification."""

    # Environmental Criteria
    CARBON_EFFICIENCY = "carbon_efficiency"
    ENERGY_EFFICIENCY = "energy_efficiency"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    CARBON_NEUTRALITY = "carbon_neutrality"

    # Performance Criteria
    MODEL_EFFICIENCY = "model_efficiency"
    INFERENCE_SPEED = "inference_speed"
    MEMORY_USAGE = "memory_usage"
    ACCURACY_MAINTENANCE = "accuracy_maintenance"

    # Governance Criteria
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    ETHICAL_AI = "ethical_ai"
    BIAS_MITIGATION = "bias_mitigation"

    # Lifecycle Criteria
    SUSTAINABLE_DEVELOPMENT = "sustainable_development"
    END_OF_LIFE_MANAGEMENT = "end_of_life_management"
    CONTINUOUS_IMPROVEMENT = "continuous_improvement"


class ValidationStatus(Enum):
    """Validation status for certification criteria."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ValidationResult:
    """Result of a validation criterion."""

    criterion: ValidationCriteria
    status: ValidationStatus
    score: float  # 0-100
    details: str
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.now)


@dataclass
class CertificationConfig:
    """Configuration for sustainable AI certification."""

    # Certification thresholds
    bronze_threshold: float = 60.0
    silver_threshold: float = 75.0
    gold_threshold: float = 85.0
    platinum_threshold: float = 95.0
    carbon_neutral_threshold: float = 100.0

    # Validation settings
    enable_automated_validation: bool = True
    validation_timeout_seconds: int = 300
    require_evidence: bool = True
    evidence_retention_days: int = 365

    # Scoring weights
    environmental_weight: float = 0.4
    performance_weight: float = 0.3
    governance_weight: float = 0.2
    lifecycle_weight: float = 0.1

    # Certification validity
    certification_validity_days: int = 365
    require_renewal: bool = True
    renewal_grace_period_days: int = 30

    # External validation
    enable_external_audit: bool = True
    audit_providers: List[str] = field(
        default_factory=lambda: ["sustainalytics", "msci", "cdp"]
    )

    # Output settings
    generate_certificate: bool = True
    certificate_format: str = "pdf"
    output_directory: str = "certifications"


@dataclass
class SustainableAICertificate:
    """Sustainable AI certification certificate."""

    certificate_id: str
    model_id: str
    model_name: str
    organization: str
    certification_level: CertificationLevel
    overall_score: float
    validation_results: List[ValidationResult]
    issued_at: datetime
    valid_until: datetime
    issued_by: str
    certificate_hash: str
    verification_url: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnvironmentalValidator:
    """Validates environmental sustainability criteria."""

    def __init__(self, config: CertificationConfig):
        self.config = config
        self.energy_tracker = EnergyTracker()
        self.carbon_calculator = CarbonCalculator()
        logger.info("Environmental validator initialized")

    def validate_carbon_efficiency(
        self, model_metrics: Dict[str, Any]
    ) -> ValidationResult:
        """Validate carbon efficiency of the AI model."""
        try:
            # Get carbon metrics
            carbon_footprint = model_metrics.get("carbon_footprint_kg", 0.0)
            model_accuracy = model_metrics.get("accuracy", 0.0)
            inference_count = model_metrics.get("inference_count", 1)

            # Calculate carbon efficiency (accuracy per kg CO2)
            if carbon_footprint > 0:
                carbon_efficiency = (
                    model_accuracy * inference_count
                ) / carbon_footprint
            else:
                carbon_efficiency = float("inf")

            # Industry benchmark: >1000 accuracy-inferences per kg CO2
            benchmark = 1000.0
            score = min(100, (carbon_efficiency / benchmark) * 100)

            if score >= 80:
                status = ValidationStatus.PASSED
                details = f"Excellent carbon efficiency: {carbon_efficiency:.1f} accuracy-inferences/kg CO2"
            elif score >= 60:
                status = ValidationStatus.WARNING
                details = f"Good carbon efficiency: {carbon_efficiency:.1f} accuracy-inferences/kg CO2"
            else:
                status = ValidationStatus.FAILED
                details = f"Poor carbon efficiency: {carbon_efficiency:.1f} accuracy-inferences/kg CO2"

            evidence = [
                f"Carbon footprint: {carbon_footprint:.3f} kg CO2",
                f"Model accuracy: {model_accuracy:.3f}",
                f"Inference count: {inference_count:,}",
                f"Carbon efficiency: {carbon_efficiency:.1f} accuracy-inferences/kg CO2",
            ]

            recommendations = []
            if score < 80:
                recommendations.extend(
                    [
                        "Implement model compression techniques",
                        "Use energy-efficient hardware",
                        "Optimize inference algorithms",
                        "Consider carbon-aware training scheduling",
                    ]
                )

            return ValidationResult(
                criterion=ValidationCriteria.CARBON_EFFICIENCY,
                status=status,
                score=score,
                details=details,
                evidence=evidence,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Carbon efficiency validation failed: {e}")
            return ValidationResult(
                criterion=ValidationCriteria.CARBON_EFFICIENCY,
                status=ValidationStatus.FAILED,
                score=0.0,
                details=f"Validation failed: {str(e)}",
                recommendations=["Fix validation errors and retry"],
            )

    def validate_energy_efficiency(
        self, model_metrics: Dict[str, Any]
    ) -> ValidationResult:
        """Validate energy efficiency of the AI model."""
        try:
            # Get energy metrics
            energy_consumption = model_metrics.get("energy_consumption_kwh", 0.0)
            model_accuracy = model_metrics.get("accuracy", 0.0)
            inference_count = model_metrics.get("inference_count", 1)

            # Calculate energy efficiency (accuracy per kWh)
            if energy_consumption > 0:
                energy_efficiency = (
                    model_accuracy * inference_count
                ) / energy_consumption
            else:
                energy_efficiency = float("inf")

            # Industry benchmark: >5000 accuracy-inferences per kWh
            benchmark = 5000.0
            score = min(100, (energy_efficiency / benchmark) * 100)

            if score >= 80:
                status = ValidationStatus.PASSED
                details = f"Excellent energy efficiency: {energy_efficiency:.1f} accuracy-inferences/kWh"
            elif score >= 60:
                status = ValidationStatus.WARNING
                details = f"Good energy efficiency: {energy_efficiency:.1f} accuracy-inferences/kWh"
            else:
                status = ValidationStatus.FAILED
                details = f"Poor energy efficiency: {energy_efficiency:.1f} accuracy-inferences/kWh"

            evidence = [
                f"Energy consumption: {energy_consumption:.3f} kWh",
                f"Model accuracy: {model_accuracy:.3f}",
                f"Inference count: {inference_count:,}",
                f"Energy efficiency: {energy_efficiency:.1f} accuracy-inferences/kWh",
            ]

            recommendations = []
            if score < 80:
                recommendations.extend(
                    [
                        "Use energy-efficient processors (ARM, RISC-V)",
                        "Implement dynamic voltage and frequency scaling",
                        "Optimize model architecture for energy efficiency",
                        "Use renewable energy sources for training",
                    ]
                )

            return ValidationResult(
                criterion=ValidationCriteria.ENERGY_EFFICIENCY,
                status=status,
                score=score,
                details=details,
                evidence=evidence,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Energy efficiency validation failed: {e}")
            return ValidationResult(
                criterion=ValidationCriteria.ENERGY_EFFICIENCY,
                status=ValidationStatus.FAILED,
                score=0.0,
                details=f"Validation failed: {str(e)}",
                recommendations=["Fix validation errors and retry"],
            )

    def validate_carbon_neutrality(
        self, model_metrics: Dict[str, Any]
    ) -> ValidationResult:
        """Validate carbon neutrality of the AI model."""
        try:
            # Get carbon metrics
            carbon_footprint = model_metrics.get("carbon_footprint_kg", 0.0)
            carbon_offsets = model_metrics.get("carbon_offsets_kg", 0.0)

            # Calculate carbon neutrality
            net_carbon = carbon_footprint - carbon_offsets
            neutrality_ratio = carbon_offsets / max(1, carbon_footprint)

            if net_carbon <= 0 and neutrality_ratio >= 1.0:
                status = ValidationStatus.PASSED
                score = 100.0
                details = f"Carbon neutral: {net_carbon:.3f} kg CO2 net emissions"
            elif neutrality_ratio >= 0.8:
                status = ValidationStatus.WARNING
                score = neutrality_ratio * 100
                details = f"Near carbon neutral: {net_carbon:.3f} kg CO2 net emissions"
            else:
                status = ValidationStatus.FAILED
                score = neutrality_ratio * 100
                details = f"Not carbon neutral: {net_carbon:.3f} kg CO2 net emissions"

            evidence = [
                f"Carbon footprint: {carbon_footprint:.3f} kg CO2",
                f"Carbon offsets: {carbon_offsets:.3f} kg CO2",
                f"Net emissions: {net_carbon:.3f} kg CO2",
                f"Neutrality ratio: {neutrality_ratio:.2f}",
            ]

            recommendations = []
            if status != ValidationStatus.PASSED:
                recommendations.extend(
                    [
                        "Purchase additional carbon offsets",
                        "Implement carbon reduction strategies",
                        "Use renewable energy sources",
                        "Optimize model efficiency",
                    ]
                )

            return ValidationResult(
                criterion=ValidationCriteria.CARBON_NEUTRALITY,
                status=status,
                score=score,
                details=details,
                evidence=evidence,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Carbon neutrality validation failed: {e}")
            return ValidationResult(
                criterion=ValidationCriteria.CARBON_NEUTRALITY,
                status=ValidationStatus.FAILED,
                score=0.0,
                details=f"Validation failed: {str(e)}",
                recommendations=["Fix validation errors and retry"],
            )


class PerformanceValidator:
    """Validates performance sustainability criteria."""

    def __init__(self, config: CertificationConfig):
        self.config = config
        logger.info("Performance validator initialized")

    def validate_model_efficiency(
        self, model_metrics: Dict[str, Any]
    ) -> ValidationResult:
        """Validate model efficiency metrics."""
        try:
            # Get model metrics
            model_size_mb = model_metrics.get("model_size_mb", 0.0)
            accuracy = model_metrics.get("accuracy", 0.0)
            inference_time_ms = model_metrics.get("inference_time_ms", 0.0)

            # Calculate efficiency score
            if model_size_mb > 0 and inference_time_ms > 0:
                efficiency_score = (accuracy * 1000) / (
                    model_size_mb * inference_time_ms
                )
            else:
                efficiency_score = 0.0

            # Industry benchmark: >0.1 accuracy/(MB*ms)
            benchmark = 0.1
            score = min(100, (efficiency_score / benchmark) * 100)

            if score >= 80:
                status = ValidationStatus.PASSED
                details = f"Excellent model efficiency: {efficiency_score:.3f} accuracy/(MB*ms)"
            elif score >= 60:
                status = ValidationStatus.WARNING
                details = (
                    f"Good model efficiency: {efficiency_score:.3f} accuracy/(MB*ms)"
                )
            else:
                status = ValidationStatus.FAILED
                details = (
                    f"Poor model efficiency: {efficiency_score:.3f} accuracy/(MB*ms)"
                )

            evidence = [
                f"Model size: {model_size_mb:.1f} MB",
                f"Accuracy: {accuracy:.3f}",
                f"Inference time: {inference_time_ms:.1f} ms",
                f"Efficiency score: {efficiency_score:.3f} accuracy/(MB*ms)",
            ]

            recommendations = []
            if score < 80:
                recommendations.extend(
                    [
                        "Implement model compression",
                        "Use quantization techniques",
                        "Optimize model architecture",
                        "Consider knowledge distillation",
                    ]
                )

            return ValidationResult(
                criterion=ValidationCriteria.MODEL_EFFICIENCY,
                status=status,
                score=score,
                details=details,
                evidence=evidence,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Model efficiency validation failed: {e}")
            return ValidationResult(
                criterion=ValidationCriteria.MODEL_EFFICIENCY,
                status=ValidationStatus.FAILED,
                score=0.0,
                details=f"Validation failed: {str(e)}",
                recommendations=["Fix validation errors and retry"],
            )

    def validate_inference_speed(
        self, model_metrics: Dict[str, Any]
    ) -> ValidationResult:
        """Validate inference speed performance."""
        try:
            # Get inference metrics
            inference_time_ms = model_metrics.get("inference_time_ms", 0.0)
            batch_size = model_metrics.get("batch_size", 1)

            # Calculate throughput
            if inference_time_ms > 0:
                throughput = (
                    batch_size * 1000
                ) / inference_time_ms  # inferences per second
            else:
                throughput = 0.0

            # Industry benchmark: >1000 inferences/second
            benchmark = 1000.0
            score = min(100, (throughput / benchmark) * 100)

            if score >= 80:
                status = ValidationStatus.PASSED
                details = (
                    f"Excellent inference speed: {throughput:.1f} inferences/second"
                )
            elif score >= 60:
                status = ValidationStatus.WARNING
                details = f"Good inference speed: {throughput:.1f} inferences/second"
            else:
                status = ValidationStatus.FAILED
                details = f"Poor inference speed: {throughput:.1f} inferences/second"

            evidence = [
                f"Inference time: {inference_time_ms:.1f} ms",
                f"Batch size: {batch_size}",
                f"Throughput: {throughput:.1f} inferences/second",
            ]

            recommendations = []
            if score < 80:
                recommendations.extend(
                    [
                        "Optimize model architecture",
                        "Use faster hardware (GPU, TPU)",
                        "Implement model optimization",
                        "Consider edge deployment",
                    ]
                )

            return ValidationResult(
                criterion=ValidationCriteria.INFERENCE_SPEED,
                status=status,
                score=score,
                details=details,
                evidence=evidence,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Inference speed validation failed: {e}")
            return ValidationResult(
                criterion=ValidationCriteria.INFERENCE_SPEED,
                status=ValidationStatus.FAILED,
                score=0.0,
                details=f"Validation failed: {str(e)}",
                recommendations=["Fix validation errors and retry"],
            )


class GovernanceValidator:
    """Validates governance and ethical AI criteria."""

    def __init__(self, config: CertificationConfig):
        self.config = config
        logger.info("Governance validator initialized")

    def validate_transparency(self, model_metrics: Dict[str, Any]) -> ValidationResult:
        """Validate model transparency and explainability."""
        try:
            # Get transparency metrics
            explainability_score = model_metrics.get("explainability_score", 0.0)
            documentation_quality = model_metrics.get("documentation_quality", 0.0)
            model_interpretability = model_metrics.get("model_interpretability", 0.0)

            # Calculate overall transparency score
            transparency_score = (
                explainability_score + documentation_quality + model_interpretability
            ) / 3

            # Industry benchmark: >0.8 transparency score
            benchmark = 0.8
            score = min(100, (transparency_score / benchmark) * 100)

            if score >= 80:
                status = ValidationStatus.PASSED
                details = (
                    f"Excellent transparency: {transparency_score:.2f} overall score"
                )
            elif score >= 60:
                status = ValidationStatus.WARNING
                details = f"Good transparency: {transparency_score:.2f} overall score"
            else:
                status = ValidationStatus.FAILED
                details = f"Poor transparency: {transparency_score:.2f} overall score"

            evidence = [
                f"Explainability score: {explainability_score:.2f}",
                f"Documentation quality: {documentation_quality:.2f}",
                f"Model interpretability: {model_interpretability:.2f}",
                f"Overall transparency: {transparency_score:.2f}",
            ]

            recommendations = []
            if score < 80:
                recommendations.extend(
                    [
                        "Implement explainable AI techniques (SHAP, LIME)",
                        "Improve model documentation",
                        "Add model interpretability features",
                        "Provide decision explanations",
                    ]
                )

            return ValidationResult(
                criterion=ValidationCriteria.TRANSPARENCY,
                status=status,
                score=score,
                details=details,
                evidence=evidence,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Transparency validation failed: {e}")
            return ValidationResult(
                criterion=ValidationCriteria.TRANSPARENCY,
                status=ValidationStatus.FAILED,
                score=0.0,
                details=f"Validation failed: {str(e)}",
                recommendations=["Fix validation errors and retry"],
            )

    def validate_bias_mitigation(
        self, model_metrics: Dict[str, Any]
    ) -> ValidationResult:
        """Validate bias mitigation measures."""
        try:
            # Get bias metrics
            bias_score = model_metrics.get("bias_score", 0.0)  # Lower is better
            fairness_metrics = model_metrics.get("fairness_metrics", {})

            # Calculate fairness score (inverse of bias)
            fairness_score = max(0, 1 - bias_score)

            # Industry benchmark: <0.1 bias score
            benchmark = 0.1
            score = min(100, (fairness_score / (1 - benchmark)) * 100)

            if score >= 80:
                status = ValidationStatus.PASSED
                details = f"Excellent bias mitigation: {bias_score:.3f} bias score"
            elif score >= 60:
                status = ValidationStatus.WARNING
                details = f"Good bias mitigation: {bias_score:.3f} bias score"
            else:
                status = ValidationStatus.FAILED
                details = f"Poor bias mitigation: {bias_score:.3f} bias score"

            evidence = [
                f"Bias score: {bias_score:.3f}",
                f"Fairness score: {fairness_score:.3f}",
                f"Fairness metrics: {fairness_metrics}",
            ]

            recommendations = []
            if score < 80:
                recommendations.extend(
                    [
                        "Implement bias detection algorithms",
                        "Use diverse training datasets",
                        "Apply fairness constraints",
                        "Regular bias auditing",
                    ]
                )

            return ValidationResult(
                criterion=ValidationCriteria.BIAS_MITIGATION,
                status=status,
                score=score,
                details=details,
                evidence=evidence,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Bias mitigation validation failed: {e}")
            return ValidationResult(
                criterion=ValidationCriteria.BIAS_MITIGATION,
                status=ValidationStatus.FAILED,
                score=0.0,
                details=f"Validation failed: {str(e)}",
                recommendations=["Fix validation errors and retry"],
            )


class SustainableAICertificationFramework:
    """Main sustainable AI certification framework."""

    def __init__(self, config: Optional[CertificationConfig] = None):
        self.config = config or CertificationConfig()

        # Initialize validators
        self.environmental_validator = EnvironmentalValidator(self.config)
        self.performance_validator = PerformanceValidator(self.config)
        self.governance_validator = GovernanceValidator(self.config)

        # Certificate storage
        self.certificates: Dict[str, SustainableAICertificate] = {}

        # Create output directory
        Path(self.config.output_directory).mkdir(exist_ok=True)

        logger.info("Sustainable AI certification framework initialized")

    def validate_model(
        self, model_id: str, model_metrics: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate a model against all certification criteria."""
        validation_results = []

        try:
            logger.info(f"Starting validation for model {model_id}")

            # Environmental validations
            validation_results.append(
                self.environmental_validator.validate_carbon_efficiency(model_metrics)
            )
            validation_results.append(
                self.environmental_validator.validate_energy_efficiency(model_metrics)
            )
            validation_results.append(
                self.environmental_validator.validate_carbon_neutrality(model_metrics)
            )

            # Performance validations
            validation_results.append(
                self.performance_validator.validate_model_efficiency(model_metrics)
            )
            validation_results.append(
                self.performance_validator.validate_inference_speed(model_metrics)
            )

            # Governance validations
            validation_results.append(
                self.governance_validator.validate_transparency(model_metrics)
            )
            validation_results.append(
                self.governance_validator.validate_bias_mitigation(model_metrics)
            )

            logger.info(
                f"Validation completed for model {model_id}: {len(validation_results)} criteria"
            )

        except Exception as e:
            logger.error(f"Model validation failed for {model_id}: {e}")

        return validation_results

    def calculate_overall_score(
        self, validation_results: List[ValidationResult]
    ) -> float:
        """Calculate overall certification score."""
        if not validation_results:
            return 0.0

        # Categorize results by criteria type
        environmental_results = [
            r
            for r in validation_results
            if r.criterion
            in [
                ValidationCriteria.CARBON_EFFICIENCY,
                ValidationCriteria.ENERGY_EFFICIENCY,
                ValidationCriteria.CARBON_NEUTRALITY,
            ]
        ]

        performance_results = [
            r
            for r in validation_results
            if r.criterion
            in [ValidationCriteria.MODEL_EFFICIENCY, ValidationCriteria.INFERENCE_SPEED]
        ]

        governance_results = [
            r
            for r in validation_results
            if r.criterion
            in [ValidationCriteria.TRANSPARENCY, ValidationCriteria.BIAS_MITIGATION]
        ]

        # Calculate weighted scores
        environmental_score = (
            np.mean([r.score for r in environmental_results])
            if environmental_results
            else 0.0
        )
        performance_score = (
            np.mean([r.score for r in performance_results])
            if performance_results
            else 0.0
        )
        governance_score = (
            np.mean([r.score for r in governance_results])
            if governance_results
            else 0.0
        )

        # Calculate overall weighted score
        overall_score = (
            environmental_score * self.config.environmental_weight
            + performance_score * self.config.performance_weight
            + governance_score * self.config.governance_weight
        )

        return overall_score

    def determine_certification_level(self, overall_score: float) -> CertificationLevel:
        """Determine certification level based on overall score."""
        if overall_score >= self.config.carbon_neutral_threshold:
            return CertificationLevel.CARBON_NEUTRAL
        elif overall_score >= self.config.platinum_threshold:
            return CertificationLevel.PLATINUM
        elif overall_score >= self.config.gold_threshold:
            return CertificationLevel.GOLD
        elif overall_score >= self.config.silver_threshold:
            return CertificationLevel.SILVER
        elif overall_score >= self.config.bronze_threshold:
            return CertificationLevel.BRONZE
        else:
            return None  # No certification

    def issue_certificate(
        self,
        model_id: str,
        model_name: str,
        organization: str,
        validation_results: List[ValidationResult],
    ) -> Optional[SustainableAICertificate]:
        """Issue a sustainable AI certificate."""
        try:
            # Calculate overall score
            overall_score = self.calculate_overall_score(validation_results)

            # Determine certification level
            certification_level = self.determine_certification_level(overall_score)

            if certification_level is None:
                logger.warning(
                    f"Model {model_id} does not meet minimum certification requirements"
                )
                return None

            # Generate certificate
            certificate_id = str(uuid.uuid4())
            issued_at = datetime.now()
            valid_until = issued_at + timedelta(
                days=self.config.certification_validity_days
            )

            # Create certificate hash for verification
            certificate_data = f"{certificate_id}{model_id}{organization}{overall_score}{issued_at.isoformat()}"
            certificate_hash = hashlib.sha256(certificate_data.encode()).hexdigest()

            # Generate verification URL
            verification_url = (
                f"https://sustainable-ai-cert.org/verify/{certificate_id}"
            )

            certificate = SustainableAICertificate(
                certificate_id=certificate_id,
                model_id=model_id,
                model_name=model_name,
                organization=organization,
                certification_level=certification_level,
                overall_score=overall_score,
                validation_results=validation_results,
                issued_at=issued_at,
                valid_until=valid_until,
                issued_by="Sustainable AI Certification Authority",
                certificate_hash=certificate_hash,
                verification_url=verification_url,
                metadata={
                    "validation_criteria_count": len(validation_results),
                    "passed_criteria": sum(
                        1
                        for r in validation_results
                        if r.status == ValidationStatus.PASSED
                    ),
                    "environmental_score": np.mean(
                        [
                            r.score
                            for r in validation_results
                            if r.criterion
                            in [
                                ValidationCriteria.CARBON_EFFICIENCY,
                                ValidationCriteria.ENERGY_EFFICIENCY,
                                ValidationCriteria.CARBON_NEUTRALITY,
                            ]
                        ]
                    ),
                    "performance_score": np.mean(
                        [
                            r.score
                            for r in validation_results
                            if r.criterion
                            in [
                                ValidationCriteria.MODEL_EFFICIENCY,
                                ValidationCriteria.INFERENCE_SPEED,
                            ]
                        ]
                    ),
                    "governance_score": np.mean(
                        [
                            r.score
                            for r in validation_results
                            if r.criterion
                            in [
                                ValidationCriteria.TRANSPARENCY,
                                ValidationCriteria.BIAS_MITIGATION,
                            ]
                        ]
                    ),
                },
            )

            # Store certificate
            self.certificates[certificate_id] = certificate

            # Save certificate to file
            if self.config.generate_certificate:
                self._save_certificate(certificate)

            logger.info(
                f"Certificate issued: {certificate_id} - {certification_level.value.upper()}"
            )
            return certificate

        except Exception as e:
            logger.error(f"Certificate issuance failed for model {model_id}: {e}")
            return None

    def _save_certificate(self, certificate: SustainableAICertificate) -> bool:
        """Save certificate to file."""
        try:
            # Save as JSON
            certificate_path = (
                Path(self.config.output_directory)
                / f"{certificate.certificate_id}.json"
            )

            certificate_dict = {
                "certificate_id": certificate.certificate_id,
                "model_id": certificate.model_id,
                "model_name": certificate.model_name,
                "organization": certificate.organization,
                "certification_level": certificate.certification_level.value,
                "overall_score": certificate.overall_score,
                "validation_results": [
                    {
                        "criterion": result.criterion.value,
                        "status": result.status.value,
                        "score": result.score,
                        "details": result.details,
                        "evidence": result.evidence,
                        "recommendations": result.recommendations,
                        "validated_at": result.validated_at.isoformat(),
                    }
                    for result in certificate.validation_results
                ],
                "issued_at": certificate.issued_at.isoformat(),
                "valid_until": certificate.valid_until.isoformat(),
                "issued_by": certificate.issued_by,
                "certificate_hash": certificate.certificate_hash,
                "verification_url": certificate.verification_url,
                "metadata": certificate.metadata,
            }

            with open(certificate_path, "w") as f:
                json.dump(certificate_dict, f, indent=2)

            logger.info(f"Certificate saved: {certificate_path}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to save certificate {certificate.certificate_id}: {e}"
            )
            return False

    def verify_certificate(
        self, certificate_id: str
    ) -> Optional[SustainableAICertificate]:
        """Verify a certificate by ID."""
        if certificate_id in self.certificates:
            certificate = self.certificates[certificate_id]

            # Check if certificate is still valid
            if datetime.now() > certificate.valid_until:
                logger.warning(f"Certificate {certificate_id} has expired")
                return None

            return certificate

        logger.warning(f"Certificate {certificate_id} not found")
        return None

    def get_certification_summary(self) -> Dict[str, Any]:
        """Get summary of all certifications."""
        total_certificates = len(self.certificates)
        valid_certificates = sum(
            1
            for cert in self.certificates.values()
            if datetime.now() <= cert.valid_until
        )

        # Count by certification level
        level_counts = {}
        for level in CertificationLevel:
            level_counts[level.value] = sum(
                1
                for cert in self.certificates.values()
                if cert.certification_level == level
            )

        # Calculate average scores
        if self.certificates:
            avg_overall_score = np.mean(
                [cert.overall_score for cert in self.certificates.values()]
            )
            avg_environmental_score = np.mean(
                [
                    cert.metadata.get("environmental_score", 0)
                    for cert in self.certificates.values()
                ]
            )
            avg_performance_score = np.mean(
                [
                    cert.metadata.get("performance_score", 0)
                    for cert in self.certificates.values()
                ]
            )
            avg_governance_score = np.mean(
                [
                    cert.metadata.get("governance_score", 0)
                    for cert in self.certificates.values()
                ]
            )
        else:
            avg_overall_score = avg_environmental_score = avg_performance_score = (
                avg_governance_score
            ) = 0.0

        return {
            "total_certificates": total_certificates,
            "valid_certificates": valid_certificates,
            "expired_certificates": total_certificates - valid_certificates,
            "certification_levels": level_counts,
            "average_scores": {
                "overall": avg_overall_score,
                "environmental": avg_environmental_score,
                "performance": avg_performance_score,
                "governance": avg_governance_score,
            },
            "certification_thresholds": {
                "bronze": self.config.bronze_threshold,
                "silver": self.config.silver_threshold,
                "gold": self.config.gold_threshold,
                "platinum": self.config.platinum_threshold,
                "carbon_neutral": self.config.carbon_neutral_threshold,
            },
        }


# Utility functions
def create_sustainable_ai_certification_framework(
    config_dict: Optional[Dict[str, Any]] = None,
) -> SustainableAICertificationFramework:
    """Create sustainable AI certification framework with configuration."""
    if config_dict:
        config = CertificationConfig(**config_dict)
    else:
        config = CertificationConfig()
    return SustainableAICertificationFramework(config)


def demo_sustainable_ai_certification() -> Dict[str, Any]:
    """Demonstrate sustainable AI certification framework."""
    logger.info("Starting sustainable AI certification demo")

    # Create certification framework
    cert_framework = create_sustainable_ai_certification_framework()

    # Sample model metrics for demonstration
    model_metrics = {
        # Environmental metrics
        "carbon_footprint_kg": 0.15,
        "carbon_offsets_kg": 0.18,
        "energy_consumption_kwh": 0.8,
        # Performance metrics
        "model_size_mb": 25.5,
        "accuracy": 0.92,
        "inference_time_ms": 12.5,
        "inference_count": 10000,
        "batch_size": 32,
        # Governance metrics
        "explainability_score": 0.85,
        "documentation_quality": 0.90,
        "model_interpretability": 0.88,
        "bias_score": 0.05,
        "fairness_metrics": {"demographic_parity": 0.95, "equalized_odds": 0.93},
    }

    # Validate model
    model_id = "demo_credit_risk_model"
    validation_results = cert_framework.validate_model(model_id, model_metrics)

    # Issue certificate
    certificate = cert_framework.issue_certificate(
        model_id=model_id,
        model_name="Advanced Credit Risk Assessment Model",
        organization="Sustainable AI Credit Corp",
        validation_results=validation_results,
    )

    # Get certification summary
    summary = cert_framework.get_certification_summary()

    return {
        "model_id": model_id,
        "validation_results": [
            {
                "criterion": result.criterion.value,
                "status": result.status.value,
                "score": result.score,
                "details": result.details,
                "recommendations_count": len(result.recommendations),
            }
            for result in validation_results
        ],
        "certificate": (
            {
                "certificate_id": certificate.certificate_id if certificate else None,
                "certification_level": (
                    certificate.certification_level.value if certificate else None
                ),
                "overall_score": certificate.overall_score if certificate else None,
                "issued_at": certificate.issued_at.isoformat() if certificate else None,
                "valid_until": (
                    certificate.valid_until.isoformat() if certificate else None
                ),
                "verification_url": (
                    certificate.verification_url if certificate else None
                ),
            }
            if certificate
            else None
        ),
        "certification_summary": summary,
        "demo_status": "completed",
    }
