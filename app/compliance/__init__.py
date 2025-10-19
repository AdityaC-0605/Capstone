"""
Compliance and Fairness Monitoring Module.

This module provides comprehensive compliance and fairness monitoring
capabilities for credit risk AI systems, including bias detection,
regulatory compliance validation, and fairness metrics calculation.
"""

from .bias_detector import (
    BiasDetectionResult,
    BiasDetector,
    BiasLevel,
    FairnessMetric,
    FairnessMetricsCalculator,
    FairnessThreshold,
    ProtectedAttribute,
    ProtectedAttributeAnalyzer,
    ProtectedGroupStats,
    analyze_dataset_bias,
    create_bias_detector,
)
from .regulatory_compliance import (
    AuditTrailEntry,
    AuditTrailManager,
    ComplianceFramework,
    ComplianceRule,
    ComplianceStatus,
    ComplianceViolation,
    DataProcessingRecord,
    ECOAComplianceChecker,
    FCRAComplianceChecker,
    GDPRComplianceChecker,
    RegulatoryComplianceValidator,
    ViolationSeverity,
    create_compliance_validator,
    validate_credit_decision_compliance,
)

__all__ = [
    # Bias Detection
    "BiasDetector",
    "FairnessMetricsCalculator",
    "ProtectedAttributeAnalyzer",
    "FairnessMetric",
    "ProtectedAttribute",
    "BiasLevel",
    "FairnessThreshold",
    "BiasDetectionResult",
    "ProtectedGroupStats",
    "create_bias_detector",
    "analyze_dataset_bias",
    # Regulatory Compliance
    "RegulatoryComplianceValidator",
    "FCRAComplianceChecker",
    "ECOAComplianceChecker",
    "GDPRComplianceChecker",
    "AuditTrailManager",
    "ComplianceFramework",
    "ComplianceStatus",
    "ViolationSeverity",
    "ComplianceRule",
    "ComplianceViolation",
    "AuditTrailEntry",
    "DataProcessingRecord",
    "create_compliance_validator",
    "validate_credit_decision_compliance",
]
