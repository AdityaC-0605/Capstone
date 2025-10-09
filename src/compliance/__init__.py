"""
Compliance and Fairness Monitoring Module.

This module provides comprehensive compliance and fairness monitoring
capabilities for credit risk AI systems, including bias detection,
regulatory compliance validation, and fairness metrics calculation.
"""

from .bias_detector import (
    BiasDetector,
    FairnessMetricsCalculator,
    ProtectedAttributeAnalyzer,
    FairnessMetric,
    ProtectedAttribute,
    BiasLevel,
    FairnessThreshold,
    BiasDetectionResult,
    ProtectedGroupStats,
    create_bias_detector,
    analyze_dataset_bias
)

from .regulatory_compliance import (
    RegulatoryComplianceValidator,
    FCRAComplianceChecker,
    ECOAComplianceChecker,
    GDPRComplianceChecker,
    AuditTrailManager,
    ComplianceFramework,
    ComplianceStatus,
    ViolationSeverity,
    ComplianceRule,
    ComplianceViolation,
    AuditTrailEntry,
    DataProcessingRecord,
    create_compliance_validator,
    validate_credit_decision_compliance
)

__all__ = [
    # Bias Detection
    'BiasDetector',
    'FairnessMetricsCalculator', 
    'ProtectedAttributeAnalyzer',
    'FairnessMetric',
    'ProtectedAttribute',
    'BiasLevel',
    'FairnessThreshold',
    'BiasDetectionResult',
    'ProtectedGroupStats',
    'create_bias_detector',
    'analyze_dataset_bias',
    
    # Regulatory Compliance
    'RegulatoryComplianceValidator',
    'FCRAComplianceChecker',
    'ECOAComplianceChecker',
    'GDPRComplianceChecker',
    'AuditTrailManager',
    'ComplianceFramework',
    'ComplianceStatus',
    'ViolationSeverity',
    'ComplianceRule',
    'ComplianceViolation',
    'AuditTrailEntry',
    'DataProcessingRecord',
    'create_compliance_validator',
    'validate_credit_decision_compliance'
]