"""
Compliance Documentation Generator for Credit Risk AI System.

This module generates comprehensive compliance documentation including
regulatory compliance reports, audit documentation, and compliance
templates for FCRA, ECOA, and GDPR requirements.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from ..core.logging import get_logger
    from .regulatory_compliance import (
        AuditTrailEntry,
        ComplianceFramework,
        ComplianceViolation,
        RegulatoryComplianceValidator,
    )
except ImportError:
    # Fallback for direct execution
    import sys

    sys.path.append(str(Path(__file__).parent.parent))

    from compliance.regulatory_compliance import (
        AuditTrailEntry,
        ComplianceFramework,
        ComplianceViolation,
        RegulatoryComplianceValidator,
    )
    from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ComplianceDocumentTemplate:
    """Template for compliance documents."""

    template_id: str
    framework: ComplianceFramework
    document_type: str
    title: str
    description: str
    template_content: str
    required_fields: List[str]
    version: str = "1.0"
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ComplianceDocumentationGenerator:
    """Generates compliance documentation and reports."""

    def __init__(self, output_dir: str = "compliance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load compliance document templates
        self.templates = self._initialize_templates()

        logger.info(
            f"Compliance documentation generator initialized with output dir: {output_dir}"
        )

    def _initialize_templates(self) -> Dict[str, ComplianceDocumentTemplate]:
        """Initialize compliance document templates."""

        templates = {}

        # FCRA Compliance Report Template
        fcra_template = ComplianceDocumentTemplate(
            template_id="fcra_compliance_report",
            framework=ComplianceFramework.FCRA,
            document_type="compliance_report",
            title="FCRA Compliance Assessment Report",
            description="Comprehensive FCRA compliance assessment and validation report",
            template_content="# FCRA Compliance Assessment Report\n\n**Overall Status:** Compliant\n**Generated:** {timestamp}",
            required_fields=["assessment_date", "violations_summary"],
        )
        templates[fcra_template.template_id] = fcra_template

        # ECOA Compliance Report Template
        ecoa_template = ComplianceDocumentTemplate(
            template_id="ecoa_compliance_report",
            framework=ComplianceFramework.ECOA,
            document_type="compliance_report",
            title="ECOA Compliance Assessment Report",
            description="Equal Credit Opportunity Act compliance assessment report",
            template_content="# ECOA Compliance Assessment Report\n\n**Overall Status:** Compliant\n**Generated:** {timestamp}",
            required_fields=["assessment_date", "bias_testing_results"],
        )
        templates[ecoa_template.template_id] = ecoa_template

        # GDPR Compliance Report Template
        gdpr_template = ComplianceDocumentTemplate(
            template_id="gdpr_compliance_report",
            framework=ComplianceFramework.GDPR,
            document_type="compliance_report",
            title="GDPR Data Protection Compliance Report",
            description="GDPR data protection and privacy compliance assessment report",
            template_content="# GDPR Data Protection Compliance Report\n\n**Overall Status:** Compliant\n**Generated:** {timestamp}",
            required_fields=["assessment_date", "lawful_basis_validation"],
        )
        templates[gdpr_template.template_id] = gdpr_template

        # Audit Trail Report Template
        audit_template = ComplianceDocumentTemplate(
            template_id="audit_trail_report",
            framework=ComplianceFramework.FCRA,
            document_type="audit_report",
            title="Compliance Audit Trail Report",
            description="Comprehensive audit trail and activity report for compliance monitoring",
            template_content="# Compliance Audit Trail Report\n\n**Period:** {period}\n**Generated:** {timestamp}",
            required_fields=["report_period", "total_activities"],
        )
        templates[audit_template.template_id] = audit_template

        return templates

    def generate_compliance_report(
        self,
        validator: RegulatoryComplianceValidator,
        framework: ComplianceFramework,
        context: Dict[str, Any],
    ) -> str:
        """Generate compliance report for specific framework."""

        # Get template for framework
        template_id = f"{framework.value}_compliance_report"
        if template_id not in self.templates:
            raise ValueError(
                f"No template found for framework: {framework.value}"
            )

        template = self.templates[template_id]

        # Validate compliance
        violations = validator.validate_compliance(framework, context)

        # Generate report content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_content = f"""# {template.title}

**Report Date:** {datetime.now().strftime("%Y-%m-%d")}
**Framework:** {framework.value.upper()}
**Generated By:** Credit Risk AI Compliance System

## Executive Summary

**Overall Compliance Status:** {"Non-Compliant" if violations else "Compliant"}
**Total Violations:** {len(violations)}
**Critical Issues:** {len([v for v in violations if v.severity.value == "critical"])}

## Violations Summary

"""

        if violations:
            report_content += "| Rule ID | Severity | Description |\n"
            report_content += "|---------|----------|-------------|\n"
            for violation in violations:
                report_content += f"| {violation.rule_id} | {violation.severity.value} | {violation.title} |\n"
        else:
            report_content += "No compliance violations detected.\n"

        report_content += f"""
## Recommendations

1. Implement regular compliance monitoring
2. Provide staff training on regulatory requirements
3. Establish clear compliance procedures
4. Conduct periodic compliance audits

---
*This report was generated automatically on {timestamp}*
"""

        # Save report
        filename = f"{framework.value}_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(
            f"Generated {framework.value} compliance report: {filepath}"
        )

        return str(filepath)

    def generate_audit_trail_report(
        self,
        validator: RegulatoryComplianceValidator,
        start_date: datetime,
        end_date: datetime,
    ) -> str:
        """Generate comprehensive audit trail report."""

        # Get audit trail data
        audit_report = validator.audit_manager.generate_audit_report(
            start_date, end_date
        )
        audit_entries = validator.audit_manager.get_audit_trail(
            start_date, end_date, compliance_relevant_only=True
        )

        # Get violations in period
        period_violations = [
            v
            for v in validator.violations
            if start_date <= v.timestamp <= end_date
        ]

        # Generate report content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_content = f"""# Compliance Audit Trail Report

**Report Period:** {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}
**Generated By:** Credit Risk AI Compliance System
**Report Generated:** {timestamp}

## Executive Summary

**Total Activities:** {audit_report["summary"]["total_entries"]}
**Unique Users:** {audit_report["summary"]["unique_users"]}
**Compliance Events:** {len([e for e in audit_entries if e.compliance_relevant])}
**Violations Detected:** {len(period_violations)}

## Activity Summary

### User Activity Breakdown
"""

        for user, count in audit_report["top_users"][:5]:
            report_content += f"- **{user}:** {count} activities\n"

        report_content += "\n### Action Type Breakdown\n"
        for action, count in list(audit_report["action_breakdown"].items())[
            :5
        ]:
            report_content += f"- **{action}:** {count} occurrences\n"

        report_content += "\n## Compliance Events\n\n"

        if period_violations:
            report_content += "### Violations Detected\n\n"
            report_content += "| Timestamp | Rule ID | Severity | Status |\n"
            report_content += "|-----------|---------|----------|--------|\n"
            for violation in period_violations[:10]:
                status = "resolved" if violation.resolved else "open"
                report_content += f"| {violation.timestamp.strftime('%Y-%m-%d %H:%M')} | {violation.rule_id} | {violation.severity.value} | {status} |\n"
        else:
            report_content += (
                "No compliance violations detected during this period.\n"
            )

        report_content += f"""
## Recommendations

1. Increase violation resolution rate to meet 90% target
2. Implement automated compliance monitoring for real-time detection
3. Enhance user training on compliance procedures
4. Review and update data retention policies

---
*This audit trail report is automatically generated and maintained for compliance purposes*
"""

        # Save report
        filename = (
            f"audit_trail_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Generated audit trail report: {filepath}")

        return str(filepath)

    def generate_comprehensive_compliance_documentation(
        self, validator: RegulatoryComplianceValidator, context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate comprehensive compliance documentation for all frameworks."""

        reports = {}

        # Generate reports for each framework
        frameworks = [
            ComplianceFramework.FCRA,
            ComplianceFramework.ECOA,
            ComplianceFramework.GDPR,
        ]

        for framework in frameworks:
            try:
                report_path = self.generate_compliance_report(
                    validator, framework, context
                )
                reports[framework.value] = report_path
                logger.info(f"Generated {framework.value} compliance report")
            except Exception as e:
                logger.error(
                    f"Failed to generate {framework.value} report: {e}"
                )

        # Generate audit trail report
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            audit_report_path = self.generate_audit_trail_report(
                validator, start_date, end_date
            )
            reports["audit_trail"] = audit_report_path
            logger.info("Generated audit trail report")
        except Exception as e:
            logger.error(f"Failed to generate audit trail report: {e}")

        # Generate summary index
        try:
            index_path = self._generate_documentation_index(reports)
            reports["index"] = index_path
            logger.info("Generated documentation index")
        except Exception as e:
            logger.error(f"Failed to generate documentation index: {e}")

        return reports

    def _generate_documentation_index(self, reports: Dict[str, str]) -> str:
        """Generate documentation index file."""

        index_content = f"""# Compliance Documentation Index

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Available Reports

"""

        for report_type, report_path in reports.items():
            if report_type != "index":
                filename = Path(report_path).name
                index_content += (
                    f"- [{report_type.upper()} Report](./{filename})\n"
                )

        index_content += f"""
## Report Descriptions

- **FCRA Report**: Fair Credit Reporting Act compliance assessment
- **ECOA Report**: Equal Credit Opportunity Act compliance assessment  
- **GDPR Report**: General Data Protection Regulation compliance assessment
- **Audit Trail Report**: Comprehensive audit trail and activity log

## Compliance Status Summary

All reports are generated automatically by the Credit Risk AI Compliance System.
For questions or concerns, contact the compliance team.

---
*This index was generated automatically on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

        index_path = self.output_dir / "README.md"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_content)

        return str(index_path)


# Utility functions


def create_compliance_documentation_generator(
    output_dir: str = "compliance_reports",
) -> ComplianceDocumentationGenerator:
    """Create compliance documentation generator."""
    return ComplianceDocumentationGenerator(output_dir)


def generate_all_compliance_reports(
    context: Dict[str, Any], output_dir: str = "compliance_reports"
) -> Dict[str, str]:
    """Generate all compliance reports for given context."""

    from .regulatory_compliance import create_compliance_validator

    # Create validator and documentation generator
    validator = create_compliance_validator()
    doc_generator = create_compliance_documentation_generator(output_dir)

    # Generate comprehensive documentation
    reports = doc_generator.generate_comprehensive_compliance_documentation(
        validator, context
    )

    return reports


if __name__ == "__main__":
    # Example usage
    from .regulatory_compliance import create_compliance_validator

    # Create test context
    test_context = {
        "purpose": "credit_transaction",
        "user_consent": True,
        "decision": "approved",
        "legal_basis": "legitimate_interests",
        "data_fields_collected": ["name", "address", "income", "credit_score"],
        "model_features": ["income", "credit_score", "employment_length"],
    }

    # Generate all compliance reports
    reports = generate_all_compliance_reports(test_context)

    print("Generated compliance documentation:")
    for report_type, path in reports.items():
        print(f"- {report_type}: {path}")
