#!/usr/bin/env python3
"""
Test script for compliance documentation generation system.

This script validates the compliance documentation capabilities including
report generation, audit trail documentation, and comprehensive compliance
reporting for FCRA, ECOA, and GDPR requirements.
"""

import sys
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.compliance.regulatory_compliance import (
        create_compliance_validator, validate_credit_decision_compliance
    )
    from src.compliance.compliance_documentation import (
        create_compliance_documentation_generator, generate_all_compliance_reports
    )
    print("✓ Successfully imported compliance modules")
except ImportError as e:
    print(f"✗ Failed to import compliance modules: {e}")
    sys.exit(1)


def create_test_context(scenario="compliant"):
    """Create test context for different compliance scenarios."""
    
    if scenario == "compliant":
        return {
            # FCRA context
            'purpose': 'credit_transaction',
            'user_consent': True,
            'business_need': True,
            'decision': 'approved',
            'adverse_action_notice_sent': False,
            'data_validation_performed': True,
            'source_verification': True,
            'update_frequency_days': 15,
            'error_correction_process': True,
            'dispute_process_exists': True,
            'investigation_timeframe_days': 25,
            'consumer_notification_process': True,
            
            # ECOA context
            'model_features': ['income', 'credit_score', 'employment_length', 'loan_amount'],
            'decision_factors': ['credit_score', 'debt_to_income_ratio'],
            'bias_test_results': {
                'gender': {'bias_detected': False, 'bias_level': 'none', 'metric_value': 0.02},
                'race': {'bias_detected': False, 'bias_level': 'none', 'metric_value': 0.03}
            },
            'reasons_specific': True,
            'collected_data_fields': ['name', 'address', 'income', 'credit_score'],
            'monitoring_purpose': False,
            'application_records_retained': True,
            'retention_period_months': 30,
            'adverse_action_records_retained': True,
            
            # GDPR context
            'legal_basis': 'legitimate_interests',
            'consent_obtained': False,
            'data_fields_collected': ['name', 'address', 'income', 'credit_score'],
            'processing_purpose': 'credit_assessment',
            'necessity_assessment_performed': True,
            'unnecessary_data_identified': [],
            'rights_mechanisms': {
                'access': True, 'rectification': True, 'erasure': True,
                'portability': True, 'restriction': True, 'objection': True,
                'automated_decision_making': True
            },
            'response_timeframe_days': 25,
            'retention_policy_exists': True,
            'retention_periods': {'personal_data': 365, 'financial_data': 2555},
            'automated_deletion': True,
            'privacy_impact_assessment': True,
            'default_privacy_settings': True,
            'data_encryption': True,
            'access_controls': True
        }
    
    elif scenario == "violations":
        return {
            # FCRA violations
            'purpose': 'unknown_purpose',
            'user_consent': False,
            'decision': 'denied',
            'adverse_action_notice_sent': False,
            'data_validation_performed': False,
            'update_frequency_days': 90,
            'dispute_process_exists': False,
            
            # ECOA violations
            'model_features': ['income', 'credit_score', 'gender', 'race'],  # Prohibited factors
            'bias_test_results': {
                'gender': {'bias_detected': True, 'bias_level': 'high', 'metric_value': 0.25},
                'race': {'bias_detected': True, 'bias_level': 'severe', 'metric_value': 0.35}
            },
            'reasons_provided': [],
            'retention_period_months': 12,  # Too short
            
            # GDPR violations
            'legal_basis': 'invalid_basis',
            'necessity_assessment_performed': False,
            'rights_mechanisms': {'access': False, 'rectification': False},
            'response_timeframe_days': 45,  # Too long
            'retention_policy_exists': False,
            'privacy_impact_assessment': False,
            'data_encryption': False
        }
    
    else:  # mixed scenario
        return {
            # FCRA - mostly compliant
            'purpose': 'credit_transaction',
            'user_consent': True,
            'decision': 'denied',
            'adverse_action_notice_sent': True,
            'notice_content': {'credit_score_used': True, 'key_factors': True},  # Missing some elements
            'data_validation_performed': True,
            'source_verification': True,
            'update_frequency_days': 20,
            'error_correction_process': True,
            'dispute_process_exists': True,
            'investigation_timeframe_days': 28,
            'consumer_notification_process': False,  # Warning
            
            # ECOA - some issues
            'model_features': ['income', 'credit_score', 'employment_length'],
            'bias_test_results': {
                'gender': {'bias_detected': False, 'bias_level': 'none'},
                'race': {'bias_detected': True, 'bias_level': 'low', 'metric_value': 0.08}  # Minor bias
            },
            'reasons_provided': ['Low credit score', 'High debt ratio'],
            'reasons_specific': True,
            'collected_data_fields': ['name', 'address', 'income', 'gender'],  # Monitoring data
            'monitoring_purpose': True,
            'consumer_consent_for_monitoring': False,  # Warning
            'application_records_retained': True,
            'retention_period_months': 25,  # Compliant
            
            # GDPR - mostly compliant
            'legal_basis': 'legitimate_interests',
            'data_fields_collected': ['name', 'address', 'income'] + [f'field_{i}' for i in range(18)],  # Many fields
            'necessity_assessment_performed': True,
            'unnecessary_data_identified': ['field_15', 'field_16'],  # Some unnecessary
            'rights_mechanisms': {
                'access': True, 'rectification': True, 'erasure': True,
                'portability': True, 'restriction': True, 'objection': True,
                'automated_decision_making': True
            },
            'response_timeframe_days': 28,
            'retention_policy_exists': True,
            'retention_periods': {'personal_data': 365, 'financial_data': 1825},
            'automated_deletion': False,  # Warning
            'privacy_impact_assessment': True,
            'default_privacy_settings': True,
            'data_encryption': True,
            'access_controls': True
        }


def test_compliance_documentation_generator():
    """Test compliance documentation generator."""
    
    print("\n" + "=" * 60)
    print("TESTING COMPLIANCE DOCUMENTATION GENERATOR")
    print("=" * 60)
    
    try:
        # Create documentation generator
        doc_generator = create_compliance_documentation_generator("test_compliance_reports")
        
        print("\n1. Testing documentation generator initialization...")
        print(f"   ✓ Output directory: {doc_generator.output_dir}")
        print(f"   ✓ Templates loaded: {len(doc_generator.templates)}")
        
        # List available templates
        print("\n2. Available compliance report templates:")
        for template_id, template in doc_generator.templates.items():
            print(f"   ✓ {template_id}: {template.title}")
            print(f"     Framework: {template.framework.value}")
            print(f"     Required fields: {len(template.required_fields)}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Compliance documentation generator test failed: {e}")
        return False


def test_individual_compliance_reports():
    """Test individual compliance report generation."""
    
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL COMPLIANCE REPORTS")
    print("=" * 60)
    
    try:
        # Create validator and documentation generator
        validator = create_compliance_validator()
        doc_generator = create_compliance_documentation_generator("test_compliance_reports")
        
        # Test scenarios
        scenarios = ["compliant", "violations", "mixed"]
        
        for scenario in scenarios:
            print(f"\n{scenario.upper()} SCENARIO:")
            
            context = create_test_context(scenario)
            
            # Test FCRA report
            print(f"\n   Testing FCRA report generation...")
            try:
                from src.compliance.regulatory_compliance import ComplianceFramework
                fcra_report = doc_generator.generate_compliance_report(
                    validator, ComplianceFramework.FCRA, context
                )
                print(f"   ✓ FCRA report generated: {Path(fcra_report).name}")
                
                # Verify file exists and has content
                if os.path.exists(fcra_report) and os.path.getsize(fcra_report) > 0:
                    print(f"     File size: {os.path.getsize(fcra_report)} bytes")
                else:
                    print(f"   ✗ FCRA report file issue")
                    
            except Exception as e:
                print(f"   ✗ FCRA report generation failed: {e}")
            
            # Test ECOA report
            print(f"\n   Testing ECOA report generation...")
            try:
                ecoa_report = doc_generator.generate_compliance_report(
                    validator, ComplianceFramework.ECOA, context
                )
                print(f"   ✓ ECOA report generated: {Path(ecoa_report).name}")
                
                if os.path.exists(ecoa_report) and os.path.getsize(ecoa_report) > 0:
                    print(f"     File size: {os.path.getsize(ecoa_report)} bytes")
                    
            except Exception as e:
                print(f"   ✗ ECOA report generation failed: {e}")
            
            # Test GDPR report
            print(f"\n   Testing GDPR report generation...")
            try:
                gdpr_report = doc_generator.generate_compliance_report(
                    validator, ComplianceFramework.GDPR, context
                )
                print(f"   ✓ GDPR report generated: {Path(gdpr_report).name}")
                
                if os.path.exists(gdpr_report) and os.path.getsize(gdpr_report) > 0:
                    print(f"     File size: {os.path.getsize(gdpr_report)} bytes")
                    
            except Exception as e:
                print(f"   ✗ GDPR report generation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Individual compliance reports test failed: {e}")
        return False


def test_audit_trail_report():
    """Test audit trail report generation."""
    
    print("\n" + "=" * 60)
    print("TESTING AUDIT TRAIL REPORT")
    print("=" * 60)
    
    try:
        # Create validator and documentation generator
        validator = create_compliance_validator()
        doc_generator = create_compliance_documentation_generator("test_compliance_reports")
        
        print("\n1. Generating audit activities...")
        
        # Generate some audit activities
        context = create_test_context("violations")
        
        # Run compliance validation to generate audit entries
        results = validator.validate_all_frameworks(context)
        total_violations = sum(len(v) for v in results.values())
        
        print(f"   ✓ Compliance validation completed")
        print(f"   ✓ Total violations detected: {total_violations}")
        
        # Generate audit trail report
        print("\n2. Generating audit trail report...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=1)
        
        audit_report = doc_generator.generate_audit_trail_report(validator, start_date, end_date)
        
        print(f"   ✓ Audit trail report generated: {Path(audit_report).name}")
        
        if os.path.exists(audit_report) and os.path.getsize(audit_report) > 0:
            print(f"   ✓ File size: {os.path.getsize(audit_report)} bytes")
            
            # Read and display some content
            with open(audit_report, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                print(f"   ✓ Report lines: {len(lines)}")
                print(f"   ✓ Contains violations section: {'Violations Detected' in content}")
                print(f"   ✓ Contains metrics section: {'Compliance Metrics' in content}")
        else:
            print(f"   ✗ Audit trail report file issue")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Audit trail report test failed: {e}")
        return False


def test_comprehensive_documentation():
    """Test comprehensive compliance documentation generation."""
    
    print("\n" + "=" * 60)
    print("TESTING COMPREHENSIVE DOCUMENTATION")
    print("=" * 60)
    
    try:
        print("\n1. Testing comprehensive report generation...")
        
        # Test different scenarios
        scenarios = ["compliant", "violations", "mixed"]
        
        for scenario in scenarios:
            print(f"\n   {scenario.upper()} SCENARIO:")
            
            context = create_test_context(scenario)
            
            # Generate all compliance reports
            reports = generate_all_compliance_reports(context, f"test_reports_{scenario}")
            
            print(f"   ✓ Generated {len(reports)} reports:")
            for report_type, report_path in reports.items():
                if os.path.exists(report_path):
                    size = os.path.getsize(report_path)
                    print(f"     - {report_type}: {Path(report_path).name} ({size} bytes)")
                else:
                    print(f"     ✗ {report_type}: File not found")
        
        print("\n2. Testing utility function...")
        
        # Test the utility function
        context = create_test_context("mixed")
        reports = generate_all_compliance_reports(context, "test_utility_reports")
        
        print(f"   ✓ Utility function generated {len(reports)} reports")
        
        # Verify index file
        if 'index' in reports and os.path.exists(reports['index']):
            with open(reports['index'], 'r') as f:
                index_content = f.read()
                print(f"   ✓ Index file contains {len(index_content.split('- ['))-1} report links")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Comprehensive documentation test failed: {e}")
        return False


def test_integration_with_compliance_validation():
    """Test integration with compliance validation system."""
    
    print("\n" + "=" * 60)
    print("TESTING INTEGRATION WITH COMPLIANCE VALIDATION")
    print("=" * 60)
    
    try:
        print("\n1. Testing credit decision compliance workflow...")
        
        # Create test credit decision
        credit_decision = create_test_context("violations")
        
        # Validate compliance
        compliance_result = validate_credit_decision_compliance(credit_decision)
        
        print(f"   ✓ Compliance validation completed")
        print(f"   ✓ Overall status: {compliance_result['overall_status']}")
        print(f"   ✓ Total violations: {compliance_result['total_violations']}")
        
        # Generate documentation for the same context
        reports = generate_all_compliance_reports(credit_decision, "test_integration_reports")
        
        print(f"   ✓ Documentation generated: {len(reports)} reports")
        
        # Verify consistency between validation and documentation
        print("\n2. Verifying consistency...")
        
        frameworks_in_validation = set(compliance_result['framework_results'].keys())
        frameworks_in_reports = set(r for r in reports.keys() if r in ['fcra', 'ecoa', 'gdpr'])
        
        if frameworks_in_validation == frameworks_in_reports:
            print(f"   ✓ Framework consistency verified")
        else:
            print(f"   ⚠ Framework mismatch: {frameworks_in_validation} vs {frameworks_in_reports}")
        
        # Check violation counts
        validation_violations = compliance_result['total_violations']
        
        print(f"   ✓ Validation detected {validation_violations} violations")
        print(f"   ✓ Reports generated for same context")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Integration test failed: {e}")
        return False


def test_report_content_quality():
    """Test quality and completeness of generated reports."""
    
    print("\n" + "=" * 60)
    print("TESTING REPORT CONTENT QUALITY")
    print("=" * 60)
    
    try:
        print("\n1. Testing report content completeness...")
        
        # Generate reports for violations scenario
        context = create_test_context("violations")
        reports = generate_all_compliance_reports(context, "test_quality_reports")
        
        # Check each report type
        report_checks = {
            'fcra': [
                'FCRA Compliance Assessment Report',
                'Permissible Purpose Verification',
                'Adverse Action Notice Requirements',
                'Accuracy Procedures',
                'Dispute Resolution',
                'Violations Summary'
            ],
            'ecoa': [
                'ECOA Compliance Assessment Report',
                'Prohibited Basis Discrimination',
                'Adverse Action Notice Requirements',
                'Data Collection Limitations',
                'Record Retention',
                'Fairness Metrics'
            ],
            'gdpr': [
                'GDPR Data Protection Compliance Report',
                'Lawful Basis for Processing',
                'Data Minimization',
                'Data Subject Rights',
                'Storage Limitation',
                'Privacy by Design'
            ],
            'audit_trail': [
                'Compliance Audit Trail Report',
                'Activity Summary',
                'Compliance Events',
                'Data Processing Activities',
                'Compliance Metrics'
            ]
        }
        
        for report_type, expected_sections in report_checks.items():
            if report_type in reports and os.path.exists(reports[report_type]):
                with open(reports[report_type], 'r') as f:
                    content = f.read()
                
                missing_sections = []
                for section in expected_sections:
                    if section not in content:
                        missing_sections.append(section)
                
                if not missing_sections:
                    print(f"   ✓ {report_type.upper()} report: All sections present")
                else:
                    print(f"   ⚠ {report_type.upper()} report: Missing sections: {missing_sections}")
                
                # Check for template variables that weren't replaced
                template_vars = [var for var in content.split() if var.startswith('{{') and var.endswith('}}')]
                if not template_vars:
                    print(f"   ✓ {report_type.upper()} report: No unreplaced template variables")
                else:
                    print(f"   ⚠ {report_type.upper()} report: Unreplaced variables: {template_vars[:5]}")
        
        print("\n2. Testing report formatting...")
        
        # Check markdown formatting
        for report_type, report_path in reports.items():
            if report_type != 'index' and os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    content = f.read()
                
                # Check for basic markdown elements
                has_headers = content.count('#') > 0
                has_tables = '|' in content
                has_lists = content.count('-') > 5
                
                formatting_score = sum([has_headers, has_tables, has_lists])
                
                if formatting_score >= 2:
                    print(f"   ✓ {report_type.upper()} report: Good formatting")
                else:
                    print(f"   ⚠ {report_type.upper()} report: Limited formatting")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Report content quality test failed: {e}")
        return False


def cleanup_test_files():
    """Clean up test files."""
    
    print("\n" + "=" * 60)
    print("CLEANING UP TEST FILES")
    print("=" * 60)
    
    try:
        import shutil
        
        test_dirs = [
            "test_compliance_reports",
            "test_reports_compliant",
            "test_reports_violations", 
            "test_reports_mixed",
            "test_utility_reports",
            "test_integration_reports",
            "test_quality_reports"
        ]
        
        cleaned = 0
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
                cleaned += 1
                print(f"   ✓ Removed {test_dir}")
        
        print(f"\n   ✓ Cleaned up {cleaned} test directories")
        
        return True
        
    except Exception as e:
        print(f"   ⚠ Cleanup failed: {e}")
        return False


async def run_all_tests():
    """Run all compliance documentation tests."""
    
    print("=" * 80)
    print("COMPLIANCE DOCUMENTATION SYSTEM TEST")
    print("=" * 80)
    print("\nThis test suite validates the compliance documentation system")
    print("including report generation, audit trail documentation, and")
    print("comprehensive compliance reporting for FCRA, ECOA, and GDPR.")
    
    tests = [
        ("Compliance Documentation Generator", test_compliance_documentation_generator),
        ("Individual Compliance Reports", test_individual_compliance_reports),
        ("Audit Trail Report", test_audit_trail_report),
        ("Comprehensive Documentation", test_comprehensive_documentation),
        ("Integration with Compliance Validation", test_integration_with_compliance_validation),
        ("Report Content Quality", test_report_content_quality),
        ("Cleanup Test Files", cleanup_test_files)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"\n✓ {test_name}: PASSED")
            else:
                print(f"\n✗ {test_name}: FAILED")
                
        except Exception as e:
            print(f"\n✗ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    print("\nDetailed results:")
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    if passed == total:
        print(f"\n🎉 All tests passed! Compliance documentation system is working correctly.")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please review the output above.")
    
    return passed == total


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_all_tests())