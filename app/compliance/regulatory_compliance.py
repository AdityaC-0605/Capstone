"""
Regulatory Compliance Validation System for Credit Risk AI.

This module implements comprehensive regulatory compliance validation
including FCRA, ECOA, GDPR compliance checks, audit trail generation,
and compliance reporting for credit risk AI systems.
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
import re
from abc import ABC, abstractmethod

try:
    from ..core.logging import get_logger, get_audit_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class ComplianceFramework(Enum):
    """Regulatory compliance frameworks."""

    FCRA = "fcra"  # Fair Credit Reporting Act
    ECOA = "ecoa"  # Equal Credit Opportunity Act
    GDPR = "gdpr"  # General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    SOX = "sox"  # Sarbanes-Oxley Act
    BASEL = "basel"  # Basel III


class ComplianceStatus(Enum):
    """Compliance check status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    UNKNOWN = "unknown"


class ViolationSeverity(Enum):
    """Severity levels for compliance violations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceRule:
    """Compliance rule definition."""

    rule_id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirement: str
    severity: ViolationSeverity
    check_function: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ComplianceViolation:
    """Compliance violation record."""

    violation_id: str
    rule_id: str
    framework: ComplianceFramework
    severity: ViolationSeverity
    title: str
    description: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None


@dataclass
class AuditTrailEntry:
    """Audit trail entry for compliance tracking."""

    entry_id: str
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    compliance_relevant: bool = True


@dataclass
class DataProcessingRecord:
    """GDPR data processing record."""

    record_id: str
    data_subject_id: str
    processing_purpose: str
    legal_basis: str
    data_categories: List[str]
    retention_period: int  # days
    processing_start: datetime
    processing_end: Optional[datetime] = None
    consent_given: bool = False
    consent_timestamp: Optional[datetime] = None
    data_minimization_applied: bool = True
    anonymization_applied: bool = False


class FCRAComplianceChecker:
    """Fair Credit Reporting Act compliance checker."""

    def __init__(self):
        self.rules = self._initialize_fcra_rules()

    def _initialize_fcra_rules(self) -> List[ComplianceRule]:
        """Initialize FCRA compliance rules."""
        return [
            ComplianceRule(
                rule_id="FCRA_001",
                framework=ComplianceFramework.FCRA,
                title="Permissible Purpose Verification",
                description="Credit reports must only be used for permissible purposes",
                requirement="15 USC 1681b - Permissible purposes of consumer reports",
                severity=ViolationSeverity.CRITICAL,
                check_function="check_permissible_purpose",
            ),
            ComplianceRule(
                rule_id="FCRA_002",
                framework=ComplianceFramework.FCRA,
                title="Adverse Action Notice",
                description="Must provide adverse action notice when credit is denied",
                requirement="15 USC 1681m - Requirements on users of consumer reports",
                severity=ViolationSeverity.HIGH,
                check_function="check_adverse_action_notice",
            ),
            ComplianceRule(
                rule_id="FCRA_003",
                framework=ComplianceFramework.FCRA,
                title="Maximum Accuracy",
                description="Must follow reasonable procedures to ensure maximum accuracy",
                requirement="15 USC 1681e(b) - Assuring accuracy of information",
                severity=ViolationSeverity.HIGH,
                check_function="check_accuracy_procedures",
            ),
            ComplianceRule(
                rule_id="FCRA_004",
                framework=ComplianceFramework.FCRA,
                title="Dispute Resolution",
                description="Must have procedures for investigating disputes",
                requirement="15 USC 1681i - Procedure in case of disputed accuracy",
                severity=ViolationSeverity.MEDIUM,
                check_function="check_dispute_procedures",
            ),
        ]

    def check_permissible_purpose(
        self, context: Dict[str, Any]
    ) -> Tuple[ComplianceStatus, Dict[str, Any]]:
        """Check if credit report usage has permissible purpose."""

        purpose = context.get("purpose", "")
        user_consent = context.get("user_consent", False)
        business_need = context.get("business_need", False)

        permissible_purposes = [
            "credit_transaction",
            "employment",
            "insurance",
            "legitimate_business_need",
            "court_order",
            "consumer_consent",
        ]

        if purpose in permissible_purposes:
            if purpose == "consumer_consent" and not user_consent:
                return ComplianceStatus.NON_COMPLIANT, {
                    "reason": "Consumer consent claimed but not verified",
                    "required_action": "Obtain and verify consumer consent",
                }

            if purpose == "legitimate_business_need" and not business_need:
                return ComplianceStatus.NON_COMPLIANT, {
                    "reason": "Legitimate business need not established",
                    "required_action": "Document legitimate business need",
                }

            return ComplianceStatus.COMPLIANT, {
                "purpose": purpose,
                "verification_status": "verified",
            }

        return ComplianceStatus.NON_COMPLIANT, {
            "reason": f'Purpose "{purpose}" is not permissible under FCRA',
            "permissible_purposes": permissible_purposes,
        }

    def check_adverse_action_notice(
        self, context: Dict[str, Any]
    ) -> Tuple[ComplianceStatus, Dict[str, Any]]:
        """Check adverse action notice requirements."""

        decision = context.get("decision", "")
        notice_sent = context.get("adverse_action_notice_sent", False)
        notice_content = context.get("notice_content", {})

        if decision in ["denied", "approved_with_conditions", "rate_increase"]:
            if not notice_sent:
                return ComplianceStatus.NON_COMPLIANT, {
                    "reason": "Adverse action notice not sent",
                    "required_action": "Send adverse action notice within required timeframe",
                }

            required_elements = [
                "credit_score_used",
                "key_factors",
                "credit_reporting_agency",
                "consumer_rights_notice",
            ]

            missing_elements = [
                elem for elem in required_elements if elem not in notice_content
            ]

            if missing_elements:
                return ComplianceStatus.NON_COMPLIANT, {
                    "reason": "Adverse action notice missing required elements",
                    "missing_elements": missing_elements,
                }

            return ComplianceStatus.COMPLIANT, {
                "notice_status": "complete",
                "elements_included": list(notice_content.keys()),
            }

        return ComplianceStatus.COMPLIANT, {
            "reason": "No adverse action taken, notice not required"
        }

    def check_accuracy_procedures(
        self, context: Dict[str, Any]
    ) -> Tuple[ComplianceStatus, Dict[str, Any]]:
        """Check accuracy maintenance procedures."""

        data_validation = context.get("data_validation_performed", False)
        source_verification = context.get("source_verification", False)
        update_frequency = context.get("update_frequency_days", 0)
        error_correction_process = context.get("error_correction_process", False)

        issues = []

        if not data_validation:
            issues.append("No data validation procedures documented")

        if not source_verification:
            issues.append("Source verification not performed")

        if update_frequency > 30:
            issues.append(f"Update frequency too low: {update_frequency} days")

        if not error_correction_process:
            issues.append("Error correction process not established")

        if issues:
            return ComplianceStatus.NON_COMPLIANT, {
                "issues": issues,
                "required_actions": [
                    "Implement data validation procedures",
                    "Establish source verification process",
                    "Increase update frequency to at least monthly",
                    "Create error correction procedures",
                ],
            }

        return ComplianceStatus.COMPLIANT, {
            "validation_status": "adequate",
            "update_frequency": update_frequency,
        }

    def check_dispute_procedures(
        self, context: Dict[str, Any]
    ) -> Tuple[ComplianceStatus, Dict[str, Any]]:
        """Check dispute resolution procedures."""

        dispute_process = context.get("dispute_process_exists", False)
        investigation_timeframe = context.get("investigation_timeframe_days", 0)
        consumer_notification = context.get("consumer_notification_process", False)

        if not dispute_process:
            return ComplianceStatus.NON_COMPLIANT, {
                "reason": "No dispute resolution process established",
                "required_action": "Establish dispute resolution procedures",
            }

        if investigation_timeframe > 30:
            return ComplianceStatus.NON_COMPLIANT, {
                "reason": f"Investigation timeframe too long: {investigation_timeframe} days",
                "required_action": "Reduce investigation timeframe to 30 days or less",
            }

        if not consumer_notification:
            return ComplianceStatus.WARNING, {
                "reason": "Consumer notification process not clearly defined",
                "recommendation": "Establish clear consumer notification procedures",
            }

        return ComplianceStatus.COMPLIANT, {
            "process_status": "adequate",
            "investigation_timeframe": investigation_timeframe,
        }


class ECOAComplianceChecker:
    """Equal Credit Opportunity Act compliance checker."""

    def __init__(self):
        self.rules = self._initialize_ecoa_rules()
        self.prohibited_factors = [
            "race",
            "color",
            "religion",
            "national_origin",
            "sex",
            "marital_status",
            "age",
            "receipt_of_public_assistance",
        ]

    def _initialize_ecoa_rules(self) -> List[ComplianceRule]:
        """Initialize ECOA compliance rules."""
        return [
            ComplianceRule(
                rule_id="ECOA_001",
                framework=ComplianceFramework.ECOA,
                title="Prohibited Basis Discrimination",
                description="Must not discriminate based on prohibited characteristics",
                requirement="15 USC 1691(a) - Prohibited discrimination",
                severity=ViolationSeverity.CRITICAL,
                check_function="check_prohibited_discrimination",
            ),
            ComplianceRule(
                rule_id="ECOA_002",
                framework=ComplianceFramework.ECOA,
                title="Adverse Action Notice Requirements",
                description="Must provide specific reasons for adverse action",
                requirement="15 USC 1691(d) - Reason for adverse action",
                severity=ViolationSeverity.HIGH,
                check_function="check_adverse_action_reasons",
            ),
            ComplianceRule(
                rule_id="ECOA_003",
                framework=ComplianceFramework.ECOA,
                title="Data Collection Limitations",
                description="Limited collection of prohibited basis information",
                requirement="12 CFR 1002.5 - Rules concerning requests for information",
                severity=ViolationSeverity.MEDIUM,
                check_function="check_data_collection",
            ),
            ComplianceRule(
                rule_id="ECOA_004",
                framework=ComplianceFramework.ECOA,
                title="Record Retention",
                description="Must retain records for required periods",
                requirement="12 CFR 1002.12 - Record retention",
                severity=ViolationSeverity.MEDIUM,
                check_function="check_record_retention",
            ),
        ]

    def check_prohibited_discrimination(
        self, context: Dict[str, Any]
    ) -> Tuple[ComplianceStatus, Dict[str, Any]]:
        """Check for discrimination based on prohibited factors."""

        model_features = context.get("model_features", [])
        decision_factors = context.get("decision_factors", [])
        bias_test_results = context.get("bias_test_results", {})

        # Check for direct use of prohibited factors
        prohibited_used = [
            factor
            for factor in model_features
            if factor.lower() in self.prohibited_factors
        ]

        if prohibited_used:
            return ComplianceStatus.NON_COMPLIANT, {
                "reason": "Model uses prohibited factors directly",
                "prohibited_factors_used": prohibited_used,
                "required_action": "Remove prohibited factors from model",
            }

        # Check bias test results
        if bias_test_results:
            significant_bias = []
            for attribute, results in bias_test_results.items():
                if attribute.lower() in self.prohibited_factors:
                    if results.get("bias_detected", False):
                        significant_bias.append(
                            {
                                "attribute": attribute,
                                "bias_level": results.get("bias_level", "unknown"),
                                "metric_value": results.get("metric_value", 0),
                            }
                        )

            if significant_bias:
                return ComplianceStatus.NON_COMPLIANT, {
                    "reason": "Significant bias detected for prohibited characteristics",
                    "bias_details": significant_bias,
                    "required_action": "Implement bias mitigation techniques",
                }

        return ComplianceStatus.COMPLIANT, {
            "discrimination_check": "passed",
            "features_reviewed": len(model_features),
        }

    def check_adverse_action_reasons(
        self, context: Dict[str, Any]
    ) -> Tuple[ComplianceStatus, Dict[str, Any]]:
        """Check adverse action reason requirements."""

        decision = context.get("decision", "")
        reasons_provided = context.get("reasons_provided", [])
        reasons_specific = context.get("reasons_specific", False)

        if decision in ["denied", "approved_with_conditions"]:
            if not reasons_provided:
                return ComplianceStatus.NON_COMPLIANT, {
                    "reason": "No reasons provided for adverse action",
                    "required_action": "Provide specific reasons for adverse action",
                }

            if not reasons_specific:
                return ComplianceStatus.NON_COMPLIANT, {
                    "reason": "Reasons provided are too vague or generic",
                    "required_action": "Provide specific, meaningful reasons",
                }

            # Check for prohibited reasons
            prohibited_reason_indicators = [
                "race",
                "color",
                "religion",
                "national origin",
                "sex",
                "gender",
                "marital status",
                "age",
            ]

            problematic_reasons = []
            for reason in reasons_provided:
                reason_lower = reason.lower()
                for indicator in prohibited_reason_indicators:
                    if indicator in reason_lower:
                        problematic_reasons.append(reason)

            if problematic_reasons:
                return ComplianceStatus.NON_COMPLIANT, {
                    "reason": "Reasons may reference prohibited factors",
                    "problematic_reasons": problematic_reasons,
                    "required_action": "Revise reasons to avoid prohibited factor references",
                }

            return ComplianceStatus.COMPLIANT, {
                "reasons_status": "adequate",
                "reasons_count": len(reasons_provided),
            }

        return ComplianceStatus.COMPLIANT, {
            "reason": "No adverse action taken, reasons not required"
        }

    def check_data_collection(
        self, context: Dict[str, Any]
    ) -> Tuple[ComplianceStatus, Dict[str, Any]]:
        """Check data collection compliance."""

        collected_data = context.get("collected_data_fields", [])
        monitoring_purpose = context.get("monitoring_purpose", False)
        consumer_consent = context.get("consumer_consent_for_monitoring", False)

        prohibited_collected = []
        for field in collected_data:
            field_lower = field.lower()
            for prohibited in self.prohibited_factors:
                if prohibited in field_lower:
                    prohibited_collected.append(field)

        if prohibited_collected:
            if not monitoring_purpose:
                return ComplianceStatus.NON_COMPLIANT, {
                    "reason": "Prohibited data collected without monitoring purpose",
                    "prohibited_fields": prohibited_collected,
                    "required_action": "Remove prohibited data or establish monitoring purpose",
                }

            if monitoring_purpose and not consumer_consent:
                return ComplianceStatus.WARNING, {
                    "reason": "Monitoring data collected without clear consumer consent",
                    "recommendation": "Obtain explicit consumer consent for monitoring data",
                }

        return ComplianceStatus.COMPLIANT, {
            "data_collection_status": "compliant",
            "fields_reviewed": len(collected_data),
        }

    def check_record_retention(
        self, context: Dict[str, Any]
    ) -> Tuple[ComplianceStatus, Dict[str, Any]]:
        """Check record retention compliance."""

        application_records = context.get("application_records_retained", False)
        retention_period_months = context.get("retention_period_months", 0)
        adverse_action_records = context.get("adverse_action_records_retained", False)

        required_retention_months = 25  # ECOA requirement

        if not application_records:
            return ComplianceStatus.NON_COMPLIANT, {
                "reason": "Application records not being retained",
                "required_action": "Implement application record retention",
            }

        if retention_period_months < required_retention_months:
            return ComplianceStatus.NON_COMPLIANT, {
                "reason": f"Retention period too short: {retention_period_months} months",
                "required_period": required_retention_months,
                "required_action": f"Extend retention period to {required_retention_months} months",
            }

        if not adverse_action_records:
            return ComplianceStatus.WARNING, {
                "reason": "Adverse action records retention not clearly established",
                "recommendation": "Ensure adverse action records are retained per requirements",
            }

        return ComplianceStatus.COMPLIANT, {
            "retention_status": "adequate",
            "retention_period": retention_period_months,
        }


class GDPRComplianceChecker:
    """GDPR compliance checker."""

    def __init__(self):
        self.rules = self._initialize_gdpr_rules()
        self.legal_bases = [
            "consent",
            "contract",
            "legal_obligation",
            "vital_interests",
            "public_task",
            "legitimate_interests",
        ]

    def _initialize_gdpr_rules(self) -> List[ComplianceRule]:
        """Initialize GDPR compliance rules."""
        return [
            ComplianceRule(
                rule_id="GDPR_001",
                framework=ComplianceFramework.GDPR,
                title="Lawful Basis for Processing",
                description="Must have lawful basis for processing personal data",
                requirement="Article 6 - Lawfulness of processing",
                severity=ViolationSeverity.CRITICAL,
                check_function="check_lawful_basis",
            ),
            ComplianceRule(
                rule_id="GDPR_002",
                framework=ComplianceFramework.GDPR,
                title="Data Minimization",
                description="Process only data necessary for the purpose",
                requirement="Article 5(1)(c) - Data minimisation",
                severity=ViolationSeverity.HIGH,
                check_function="check_data_minimization",
            ),
            ComplianceRule(
                rule_id="GDPR_003",
                framework=ComplianceFramework.GDPR,
                title="Data Subject Rights",
                description="Must provide mechanisms for data subject rights",
                requirement="Chapter III - Rights of the data subject",
                severity=ViolationSeverity.HIGH,
                check_function="check_data_subject_rights",
            ),
            ComplianceRule(
                rule_id="GDPR_004",
                framework=ComplianceFramework.GDPR,
                title="Data Retention Limits",
                description="Must not retain data longer than necessary",
                requirement="Article 5(1)(e) - Storage limitation",
                severity=ViolationSeverity.MEDIUM,
                check_function="check_retention_limits",
            ),
            ComplianceRule(
                rule_id="GDPR_005",
                framework=ComplianceFramework.GDPR,
                title="Privacy by Design",
                description="Must implement privacy by design principles",
                requirement="Article 25 - Data protection by design and by default",
                severity=ViolationSeverity.MEDIUM,
                check_function="check_privacy_by_design",
            ),
        ]

    def check_lawful_basis(
        self, context: Dict[str, Any]
    ) -> Tuple[ComplianceStatus, Dict[str, Any]]:
        """Check lawful basis for processing."""

        legal_basis = context.get("legal_basis", "")
        consent_obtained = context.get("consent_obtained", False)
        consent_specific = context.get("consent_specific", False)
        consent_withdrawable = context.get("consent_withdrawable", False)

        if legal_basis not in self.legal_bases:
            return ComplianceStatus.NON_COMPLIANT, {
                "reason": f"Invalid or missing legal basis: {legal_basis}",
                "valid_bases": self.legal_bases,
                "required_action": "Establish valid legal basis for processing",
            }

        if legal_basis == "consent":
            issues = []
            if not consent_obtained:
                issues.append("Consent not obtained")
            if not consent_specific:
                issues.append("Consent not specific to purpose")
            if not consent_withdrawable:
                issues.append("Consent withdrawal mechanism not provided")

            if issues:
                return ComplianceStatus.NON_COMPLIANT, {
                    "reason": "Consent requirements not met",
                    "issues": issues,
                    "required_action": "Address consent requirement issues",
                }

        return ComplianceStatus.COMPLIANT, {
            "legal_basis": legal_basis,
            "basis_status": "valid",
        }

    def check_data_minimization(
        self, context: Dict[str, Any]
    ) -> Tuple[ComplianceStatus, Dict[str, Any]]:
        """Check data minimization compliance."""

        data_collected = context.get("data_fields_collected", [])
        processing_purpose = context.get("processing_purpose", "")
        necessity_assessment = context.get("necessity_assessment_performed", False)
        unnecessary_data = context.get("unnecessary_data_identified", [])

        if not necessity_assessment:
            return ComplianceStatus.NON_COMPLIANT, {
                "reason": "Data necessity assessment not performed",
                "required_action": "Perform data necessity assessment",
            }

        if unnecessary_data:
            return ComplianceStatus.NON_COMPLIANT, {
                "reason": "Unnecessary data being collected",
                "unnecessary_fields": unnecessary_data,
                "required_action": "Remove unnecessary data collection",
            }

        # Check for excessive data collection
        if len(data_collected) > 20:  # Arbitrary threshold for demonstration
            return ComplianceStatus.WARNING, {
                "reason": f"Large number of data fields collected: {len(data_collected)}",
                "recommendation": "Review data collection for minimization opportunities",
            }

        return ComplianceStatus.COMPLIANT, {
            "minimization_status": "adequate",
            "fields_collected": len(data_collected),
        }

    def check_data_subject_rights(
        self, context: Dict[str, Any]
    ) -> Tuple[ComplianceStatus, Dict[str, Any]]:
        """Check data subject rights implementation."""

        rights_mechanisms = context.get("rights_mechanisms", {})
        required_rights = [
            "access",
            "rectification",
            "erasure",
            "portability",
            "restriction",
            "objection",
            "automated_decision_making",
        ]

        missing_rights = []
        for right in required_rights:
            if not rights_mechanisms.get(right, False):
                missing_rights.append(right)

        if missing_rights:
            return ComplianceStatus.NON_COMPLIANT, {
                "reason": "Missing data subject rights mechanisms",
                "missing_rights": missing_rights,
                "required_action": "Implement missing rights mechanisms",
            }

        response_timeframe = context.get("response_timeframe_days", 0)
        if response_timeframe > 30:
            return ComplianceStatus.NON_COMPLIANT, {
                "reason": f"Response timeframe too long: {response_timeframe} days",
                "required_action": "Reduce response timeframe to 30 days or less",
            }

        return ComplianceStatus.COMPLIANT, {
            "rights_status": "implemented",
            "response_timeframe": response_timeframe,
        }

    def check_retention_limits(
        self, context: Dict[str, Any]
    ) -> Tuple[ComplianceStatus, Dict[str, Any]]:
        """Check data retention limits."""

        retention_policy = context.get("retention_policy_exists", False)
        retention_periods = context.get("retention_periods", {})
        automated_deletion = context.get("automated_deletion", False)

        if not retention_policy:
            return ComplianceStatus.NON_COMPLIANT, {
                "reason": "No data retention policy established",
                "required_action": "Establish data retention policy",
            }

        if not retention_periods:
            return ComplianceStatus.NON_COMPLIANT, {
                "reason": "Retention periods not defined",
                "required_action": "Define specific retention periods for data categories",
            }

        if not automated_deletion:
            return ComplianceStatus.WARNING, {
                "reason": "Automated deletion not implemented",
                "recommendation": "Implement automated data deletion processes",
            }

        return ComplianceStatus.COMPLIANT, {
            "retention_status": "adequate",
            "categories_defined": len(retention_periods),
        }

    def check_privacy_by_design(
        self, context: Dict[str, Any]
    ) -> Tuple[ComplianceStatus, Dict[str, Any]]:
        """Check privacy by design implementation."""

        privacy_impact_assessment = context.get("privacy_impact_assessment", False)
        default_privacy_settings = context.get("default_privacy_settings", False)
        data_encryption = context.get("data_encryption", False)
        access_controls = context.get("access_controls", False)

        issues = []
        if not privacy_impact_assessment:
            issues.append("Privacy impact assessment not performed")
        if not default_privacy_settings:
            issues.append("Privacy-friendly defaults not implemented")
        if not data_encryption:
            issues.append("Data encryption not implemented")
        if not access_controls:
            issues.append("Access controls not adequate")

        if len(issues) >= 3:
            return ComplianceStatus.NON_COMPLIANT, {
                "reason": "Privacy by design not adequately implemented",
                "issues": issues,
                "required_action": "Implement privacy by design measures",
            }
        elif issues:
            return ComplianceStatus.WARNING, {
                "reason": "Some privacy by design measures missing",
                "issues": issues,
                "recommendation": "Address remaining privacy by design issues",
            }

        return ComplianceStatus.COMPLIANT, {"privacy_by_design_status": "implemented"}


class AuditTrailManager:
    """Manages audit trails for compliance tracking."""

    def __init__(self, max_entries: int = 100000):
        self.max_entries = max_entries
        self.audit_trail: List[AuditTrailEntry] = []
        self.data_processing_records: Dict[str, DataProcessingRecord] = {}

    def log_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        details: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        compliance_relevant: bool = True,
    ) -> str:
        """Log an action to the audit trail."""

        entry_id = hashlib.md5(
            f"{user_id}_{action}_{resource}_{datetime.now().isoformat()}".encode()
        ).hexdigest()

        entry = AuditTrailEntry(
            entry_id=entry_id,
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            compliance_relevant=compliance_relevant,
        )

        self.audit_trail.append(entry)

        # Maintain size limit
        if len(self.audit_trail) > self.max_entries:
            self.audit_trail = self.audit_trail[-self.max_entries :]

        # Log to system audit logger
        audit_logger.log_model_operation(
            user_id=user_id,
            model_id="compliance_system",
            operation=action,
            success=True,
            details={
                "resource": resource,
                "entry_id": entry_id,
                "compliance_relevant": compliance_relevant,
                **details,
            },
        )

        return entry_id

    def create_data_processing_record(
        self,
        data_subject_id: str,
        processing_purpose: str,
        legal_basis: str,
        data_categories: List[str],
        retention_period: int,
        consent_given: bool = False,
    ) -> str:
        """Create a GDPR data processing record."""

        record_id = hashlib.md5(
            f"{data_subject_id}_{processing_purpose}_{datetime.now().isoformat()}".encode()
        ).hexdigest()

        record = DataProcessingRecord(
            record_id=record_id,
            data_subject_id=data_subject_id,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            data_categories=data_categories,
            retention_period=retention_period,
            processing_start=datetime.now(),
            consent_given=consent_given,
            consent_timestamp=datetime.now() if consent_given else None,
            data_minimization_applied=True,
            anonymization_applied=False,
        )

        self.data_processing_records[record_id] = record

        # Log the creation
        self.log_action(
            user_id="system",
            action="create_data_processing_record",
            resource=f"data_processing_record/{record_id}",
            details={
                "data_subject_id": data_subject_id,
                "processing_purpose": processing_purpose,
                "legal_basis": legal_basis,
                "data_categories": data_categories,
                "retention_period": retention_period,
            },
        )

        return record_id

    def get_audit_trail(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        compliance_relevant_only: bool = False,
    ) -> List[AuditTrailEntry]:
        """Retrieve audit trail entries with filtering."""

        filtered_entries = self.audit_trail

        if compliance_relevant_only:
            filtered_entries = [e for e in filtered_entries if e.compliance_relevant]

        if start_date:
            filtered_entries = [
                e for e in filtered_entries if e.timestamp >= start_date
            ]

        if end_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_date]

        if user_id:
            filtered_entries = [e for e in filtered_entries if e.user_id == user_id]

        if action:
            filtered_entries = [e for e in filtered_entries if e.action == action]

        return filtered_entries

    def get_data_processing_records(
        self, data_subject_id: Optional[str] = None
    ) -> List[DataProcessingRecord]:
        """Get data processing records."""

        records = list(self.data_processing_records.values())

        if data_subject_id:
            records = [r for r in records if r.data_subject_id == data_subject_id]

        return records

    def generate_audit_report(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report."""

        entries = self.get_audit_trail(
            start_date, end_date, compliance_relevant_only=True
        )

        # Aggregate statistics
        action_counts = {}
        user_activity = {}
        resource_access = {}

        for entry in entries:
            action_counts[entry.action] = action_counts.get(entry.action, 0) + 1
            user_activity[entry.user_id] = user_activity.get(entry.user_id, 0) + 1
            resource_access[entry.resource] = resource_access.get(entry.resource, 0) + 1

        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "summary": {
                "total_entries": len(entries),
                "unique_users": len(user_activity),
                "unique_actions": len(action_counts),
                "unique_resources": len(resource_access),
            },
            "action_breakdown": action_counts,
            "top_users": sorted(
                user_activity.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "top_resources": sorted(
                resource_access.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "data_processing_records": len(self.data_processing_records),
            "generated_at": datetime.now().isoformat(),
        }


class RegulatoryComplianceValidator:
    """Main regulatory compliance validation system."""

    def __init__(self):
        self.fcra_checker = FCRAComplianceChecker()
        self.ecoa_checker = ECOAComplianceChecker()
        self.gdpr_checker = GDPRComplianceChecker()
        self.audit_manager = AuditTrailManager()

        self.violations: List[ComplianceViolation] = []
        self.compliance_history: List[Dict[str, Any]] = []

        logger.info("Regulatory compliance validator initialized")

    def validate_compliance(
        self, framework: ComplianceFramework, context: Dict[str, Any]
    ) -> List[ComplianceViolation]:
        """Validate compliance for a specific framework."""

        violations = []

        if framework == ComplianceFramework.FCRA:
            checker = self.fcra_checker
        elif framework == ComplianceFramework.ECOA:
            checker = self.ecoa_checker
        elif framework == ComplianceFramework.GDPR:
            checker = self.gdpr_checker
        else:
            logger.warning(f"Unsupported compliance framework: {framework}")
            return violations

        for rule in checker.rules:
            if not rule.enabled:
                continue

            try:
                check_function = getattr(checker, rule.check_function)
                status, details = check_function(context)

                if status == ComplianceStatus.NON_COMPLIANT:
                    violation = ComplianceViolation(
                        violation_id=self._generate_violation_id(rule.rule_id),
                        rule_id=rule.rule_id,
                        framework=framework,
                        severity=rule.severity,
                        title=rule.title,
                        description=rule.description,
                        details=details,
                        timestamp=datetime.now(),
                    )
                    violations.append(violation)

                    # Log violation
                    logger.warning(
                        f"Compliance violation: {rule.rule_id} - {rule.title}"
                    )

                    # Audit log
                    self.audit_manager.log_action(
                        user_id="compliance_system",
                        action="compliance_violation_detected",
                        resource=f"compliance_rule/{rule.rule_id}",
                        details={
                            "framework": framework.value,
                            "rule_id": rule.rule_id,
                            "severity": rule.severity.value,
                            "violation_details": details,
                        },
                    )

                elif status == ComplianceStatus.WARNING:
                    # Log warning
                    logger.warning(f"Compliance warning: {rule.rule_id} - {details}")

                    self.audit_manager.log_action(
                        user_id="compliance_system",
                        action="compliance_warning",
                        resource=f"compliance_rule/{rule.rule_id}",
                        details={
                            "framework": framework.value,
                            "rule_id": rule.rule_id,
                            "warning_details": details,
                        },
                    )

            except Exception as e:
                logger.error(f"Error checking compliance rule {rule.rule_id}: {e}")

                # Create violation for check failure
                violation = ComplianceViolation(
                    violation_id=self._generate_violation_id(rule.rule_id),
                    rule_id=rule.rule_id,
                    framework=framework,
                    severity=ViolationSeverity.HIGH,
                    title=f"Compliance Check Failed: {rule.title}",
                    description=f"Failed to execute compliance check: {str(e)}",
                    details={"error": str(e), "check_function": rule.check_function},
                    timestamp=datetime.now(),
                )
                violations.append(violation)

        # Store violations
        self.violations.extend(violations)

        return violations

    def validate_all_frameworks(
        self, context: Dict[str, Any]
    ) -> Dict[ComplianceFramework, List[ComplianceViolation]]:
        """Validate compliance across all supported frameworks."""

        results = {}

        frameworks = [
            ComplianceFramework.FCRA,
            ComplianceFramework.ECOA,
            ComplianceFramework.GDPR,
        ]

        for framework in frameworks:
            try:
                violations = self.validate_compliance(framework, context)
                results[framework] = violations
            except Exception as e:
                logger.error(f"Error validating {framework.value} compliance: {e}")
                results[framework] = []

        # Store compliance check in history
        self.compliance_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "frameworks_checked": [f.value for f in frameworks],
                "total_violations": sum(len(v) for v in results.values()),
                "context_summary": {"keys": list(context.keys()), "size": len(context)},
            }
        )

        return results

    def _generate_violation_id(self, rule_id: str) -> str:
        """Generate unique violation ID."""
        return hashlib.md5(
            f"{rule_id}_{datetime.now().isoformat()}".encode()
        ).hexdigest()

    def resolve_violation(self, violation_id: str, resolution_notes: str) -> bool:
        """Mark a violation as resolved."""

        for violation in self.violations:
            if violation.violation_id == violation_id:
                violation.resolved = True
                violation.resolution_notes = resolution_notes
                violation.resolved_at = datetime.now()

                # Audit log
                self.audit_manager.log_action(
                    user_id="compliance_officer",
                    action="violation_resolved",
                    resource=f"violation/{violation_id}",
                    details={
                        "rule_id": violation.rule_id,
                        "framework": violation.framework.value,
                        "resolution_notes": resolution_notes,
                    },
                )

                logger.info(f"Compliance violation {violation_id} resolved")
                return True

        return False

    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status."""

        active_violations = [v for v in self.violations if not v.resolved]

        # Group by framework
        framework_status = {}
        for framework in ComplianceFramework:
            framework_violations = [
                v for v in active_violations if v.framework == framework
            ]

            if not framework_violations:
                status = "compliant"
            else:
                critical_violations = [
                    v
                    for v in framework_violations
                    if v.severity == ViolationSeverity.CRITICAL
                ]
                high_violations = [
                    v
                    for v in framework_violations
                    if v.severity == ViolationSeverity.HIGH
                ]

                if critical_violations:
                    status = "critical_violations"
                elif high_violations:
                    status = "high_violations"
                else:
                    status = "minor_violations"

            framework_status[framework.value] = {
                "status": status,
                "active_violations": len(framework_violations),
                "critical_violations": len(
                    [
                        v
                        for v in framework_violations
                        if v.severity == ViolationSeverity.CRITICAL
                    ]
                ),
                "high_violations": len(
                    [
                        v
                        for v in framework_violations
                        if v.severity == ViolationSeverity.HIGH
                    ]
                ),
            }

        return {
            "overall_status": (
                "compliant" if not active_violations else "violations_present"
            ),
            "total_active_violations": len(active_violations),
            "total_resolved_violations": len(
                [v for v in self.violations if v.resolved]
            ),
            "framework_status": framework_status,
            "last_check": (
                self.compliance_history[-1]["timestamp"]
                if self.compliance_history
                else None
            ),
        }

    def generate_compliance_report(
        self, include_resolved: bool = False
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""

        violations_to_include = self.violations
        if not include_resolved:
            violations_to_include = [v for v in self.violations if not v.resolved]

        # Group violations by framework and severity
        framework_breakdown = {}
        severity_breakdown = {}

        for violation in violations_to_include:
            framework = violation.framework.value
            severity = violation.severity.value

            if framework not in framework_breakdown:
                framework_breakdown[framework] = []
            framework_breakdown[framework].append(
                {
                    "violation_id": violation.violation_id,
                    "rule_id": violation.rule_id,
                    "title": violation.title,
                    "severity": severity,
                    "timestamp": violation.timestamp.isoformat(),
                    "resolved": violation.resolved,
                }
            )

            severity_breakdown[severity] = severity_breakdown.get(severity, 0) + 1

        # Recent compliance history
        recent_history = (
            self.compliance_history[-10:] if self.compliance_history else []
        )

        return {
            "report_generated": datetime.now().isoformat(),
            "summary": {
                "total_violations": len(violations_to_include),
                "active_violations": len(
                    [v for v in violations_to_include if not v.resolved]
                ),
                "resolved_violations": len(
                    [v for v in violations_to_include if v.resolved]
                ),
                "frameworks_covered": list(framework_breakdown.keys()),
                "severity_distribution": severity_breakdown,
            },
            "framework_breakdown": framework_breakdown,
            "recent_compliance_checks": recent_history,
            "recommendations": self._generate_compliance_recommendations(
                violations_to_include
            ),
        }

    def _generate_compliance_recommendations(
        self, violations: List[ComplianceViolation]
    ) -> List[str]:
        """Generate compliance recommendations based on violations."""

        recommendations = []

        # Count violations by framework and severity
        critical_violations = [
            v
            for v in violations
            if v.severity == ViolationSeverity.CRITICAL and not v.resolved
        ]
        high_violations = [
            v
            for v in violations
            if v.severity == ViolationSeverity.HIGH and not v.resolved
        ]

        if critical_violations:
            recommendations.append(
                f"URGENT: Address {len(critical_violations)} critical compliance violations immediately"
            )

        if high_violations:
            recommendations.append(
                f"HIGH PRIORITY: Resolve {len(high_violations)} high-severity compliance issues"
            )

        # Framework-specific recommendations
        fcra_violations = [
            v
            for v in violations
            if v.framework == ComplianceFramework.FCRA and not v.resolved
        ]
        if fcra_violations:
            recommendations.append(
                "Review FCRA compliance procedures, particularly around permissible purposes and adverse action notices"
            )

        ecoa_violations = [
            v
            for v in violations
            if v.framework == ComplianceFramework.ECOA and not v.resolved
        ]
        if ecoa_violations:
            recommendations.append(
                "Implement bias testing and mitigation to address ECOA discrimination concerns"
            )

        gdpr_violations = [
            v
            for v in violations
            if v.framework == ComplianceFramework.GDPR and not v.resolved
        ]
        if gdpr_violations:
            recommendations.append(
                "Strengthen data protection measures and data subject rights implementation"
            )

        if not violations:
            recommendations.append(
                "Maintain current compliance practices and continue regular monitoring"
            )

        return recommendations


# Utility functions


def create_compliance_validator() -> RegulatoryComplianceValidator:
    """Create regulatory compliance validator."""
    return RegulatoryComplianceValidator()


def validate_credit_decision_compliance(
    decision_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate compliance for a credit decision."""

    validator = create_compliance_validator()

    # Validate against relevant frameworks
    results = validator.validate_all_frameworks(decision_context)

    # Generate summary
    total_violations = sum(len(violations) for violations in results.values())

    return {
        "compliance_check_timestamp": datetime.now().isoformat(),
        "total_violations": total_violations,
        "framework_results": {
            framework.value: {
                "violations": len(violations),
                "violation_details": [
                    {
                        "rule_id": v.rule_id,
                        "title": v.title,
                        "severity": v.severity.value,
                        "details": v.details,
                    }
                    for v in violations
                ],
            }
            for framework, violations in results.items()
        },
        "overall_status": (
            "compliant" if total_violations == 0 else "violations_detected"
        ),
    }


if __name__ == "__main__":
    # Example usage
    validator = create_compliance_validator()

    # Example credit decision context
    context = {
        # FCRA context
        "purpose": "credit_transaction",
        "user_consent": True,
        "decision": "denied",
        "adverse_action_notice_sent": True,
        "notice_content": {
            "credit_score_used": True,
            "key_factors": True,
            "credit_reporting_agency": True,
            "consumer_rights_notice": True,
        },
        # ECOA context
        "model_features": ["income", "credit_score", "employment_length"],
        "bias_test_results": {
            "gender": {"bias_detected": False, "bias_level": "none"},
            "race": {
                "bias_detected": True,
                "bias_level": "moderate",
                "metric_value": 0.12,
            },
        },
        "reasons_provided": ["Low credit score", "High debt-to-income ratio"],
        "reasons_specific": True,
        # GDPR context
        "legal_basis": "legitimate_interests",
        "data_fields_collected": ["name", "address", "income", "credit_score"],
        "necessity_assessment_performed": True,
        "rights_mechanisms": {
            "access": True,
            "rectification": True,
            "erasure": True,
            "portability": True,
            "restriction": True,
            "objection": True,
            "automated_decision_making": True,
        },
        "response_timeframe_days": 25,
    }

    # Validate compliance
    results = validator.validate_all_frameworks(context)

    # Generate report
    report = validator.generate_compliance_report()

    print("Compliance Validation Results:")
    print(json.dumps(report, indent=2, default=str))
