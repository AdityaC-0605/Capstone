#!/usr/bin/env python3
"""
Production validation script for the Sustainable Credit Risk AI system
Performs comprehensive validation of the production deployment
"""

import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import requests


class ProductionValidator:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.timeout = 30
        self.validation_results = []

    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """Log validation result"""
        result = {
            "test": test_name,
            "passed": passed,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        self.validation_results.append(result)

        status = "✅" if passed else "❌"
        print(f"{status} {test_name}: {message}")

    def validate_api_health(self) -> bool:
        """Validate API health and readiness"""
        try:
            # Health check
            health_response = self.session.get(f"{self.base_url}/health")
            if health_response.status_code != 200:
                self.log_result(
                    "API Health",
                    False,
                    f"Health endpoint returned {health_response.status_code}",
                )
                return False

            # Readiness check
            ready_response = self.session.get(f"{self.base_url}/ready")
            if ready_response.status_code != 200:
                self.log_result(
                    "API Readiness",
                    False,
                    f"Ready endpoint returned {ready_response.status_code}",
                )
                return False

            self.log_result("API Health", True, "Health and readiness checks passed")
            return True

        except Exception as e:
            self.log_result("API Health", False, f"Exception: {str(e)}")
            return False

    def validate_prediction_accuracy(self) -> bool:
        """Validate prediction endpoint with known test cases"""
        try:
            # Test case with expected low risk
            low_risk_case = {
                "application_id": "validation_low_risk",
                "applicant_id": "test_applicant_1",
                "loan_amount": 25000.0,
                "loan_term": 24,
                "annual_income": 100000.0,
                "employment_length": 10,
                "home_ownership": "OWN",
                "debt_to_income_ratio": 0.15,
                "credit_history_length": 15,
                "behavioral_features": {
                    "avg_monthly_spending": 2000.0,
                    "payment_frequency": 1.0,
                },
                "temporal_features": [],
                "relational_features": {},
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/predict",
                json=low_risk_case,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                self.log_result(
                    "Prediction Accuracy",
                    False,
                    f"Prediction endpoint returned {response.status_code}",
                )
                return False

            result = response.json()

            # Validate response structure
            required_fields = [
                "risk_score",
                "risk_category",
                "confidence",
                "explanation",
            ]
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                self.log_result(
                    "Prediction Accuracy", False, f"Missing fields: {missing_fields}"
                )
                return False

            # Validate risk score is reasonable for low-risk case
            risk_score = result["risk_score"]
            if not (0 <= risk_score <= 1):
                self.log_result(
                    "Prediction Accuracy",
                    False,
                    f"Risk score {risk_score} out of valid range [0,1]",
                )
                return False

            # For this low-risk case, we expect low risk score
            if risk_score > 0.5:
                self.log_result(
                    "Prediction Accuracy",
                    False,
                    f"Unexpected high risk score {risk_score} for low-risk case",
                )
                return False

            self.log_result(
                "Prediction Accuracy",
                True,
                f"Prediction validation passed (risk_score: {risk_score})",
            )
            return True

        except Exception as e:
            self.log_result("Prediction Accuracy", False, f"Exception: {str(e)}")
            return False

    def validate_batch_processing(self) -> bool:
        """Validate batch prediction endpoint"""
        try:
            batch_data = {
                "applications": [
                    {
                        "application_id": "batch_val_1",
                        "applicant_id": "test_batch_1",
                        "loan_amount": 30000.0,
                        "loan_term": 36,
                        "annual_income": 70000.0,
                        "employment_length": 5,
                        "home_ownership": "RENT",
                        "debt_to_income_ratio": 0.25,
                        "credit_history_length": 8,
                        "behavioral_features": {"avg_monthly_spending": 2800.0},
                        "temporal_features": [],
                        "relational_features": {},
                    },
                    {
                        "application_id": "batch_val_2",
                        "applicant_id": "test_batch_2",
                        "loan_amount": 45000.0,
                        "loan_term": 48,
                        "annual_income": 90000.0,
                        "employment_length": 8,
                        "home_ownership": "OWN",
                        "debt_to_income_ratio": 0.20,
                        "credit_history_length": 12,
                        "behavioral_features": {"avg_monthly_spending": 3200.0},
                        "temporal_features": [],
                        "relational_features": {},
                    },
                ]
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/batch",
                json=batch_data,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                self.log_result(
                    "Batch Processing",
                    False,
                    f"Batch endpoint returned {response.status_code}",
                )
                return False

            result = response.json()

            if "predictions" not in result:
                self.log_result(
                    "Batch Processing", False, "Missing 'predictions' in response"
                )
                return False

            predictions = result["predictions"]
            if len(predictions) != 2:
                self.log_result(
                    "Batch Processing",
                    False,
                    f"Expected 2 predictions, got {len(predictions)}",
                )
                return False

            # Validate each prediction
            for i, prediction in enumerate(predictions):
                required_fields = [
                    "application_id",
                    "risk_score",
                    "risk_category",
                    "confidence",
                ]
                missing_fields = [
                    field for field in required_fields if field not in prediction
                ]
                if missing_fields:
                    self.log_result(
                        "Batch Processing",
                        False,
                        f"Prediction {i} missing fields: {missing_fields}",
                    )
                    return False

            self.log_result(
                "Batch Processing", True, "Batch processing validation passed"
            )
            return True

        except Exception as e:
            self.log_result("Batch Processing", False, f"Exception: {str(e)}")
            return False

    def validate_response_times(self) -> bool:
        """Validate API response times meet SLA requirements"""
        try:
            # Test single prediction response time
            start_time = time.time()

            sample_data = {
                "application_id": "perf_test",
                "applicant_id": "perf_applicant",
                "loan_amount": 35000.0,
                "loan_term": 36,
                "annual_income": 65000.0,
                "employment_length": 4,
                "home_ownership": "RENT",
                "debt_to_income_ratio": 0.28,
                "credit_history_length": 6,
                "behavioral_features": {"avg_monthly_spending": 2600.0},
                "temporal_features": [],
                "relational_features": {},
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/predict",
                json=sample_data,
                headers={"Content-Type": "application/json"},
            )

            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            if response.status_code != 200:
                self.log_result(
                    "Response Times",
                    False,
                    f"Prediction failed with status {response.status_code}",
                )
                return False

            # SLA requirement: < 100ms per prediction
            if response_time_ms > 100:
                self.log_result(
                    "Response Times",
                    False,
                    f"Response time {response_time_ms:.2f}ms exceeds 100ms SLA",
                )
                return False

            self.log_result(
                "Response Times",
                True,
                f"Response time {response_time_ms:.2f}ms meets SLA",
            )
            return True

        except Exception as e:
            self.log_result("Response Times", False, f"Exception: {str(e)}")
            return False

    def validate_model_explainability(self) -> bool:
        """Validate that model explanations are provided"""
        try:
            sample_data = {
                "application_id": "explain_test",
                "applicant_id": "explain_applicant",
                "loan_amount": 40000.0,
                "loan_term": 36,
                "annual_income": 75000.0,
                "employment_length": 6,
                "home_ownership": "MORTGAGE",
                "debt_to_income_ratio": 0.30,
                "credit_history_length": 10,
                "behavioral_features": {
                    "avg_monthly_spending": 3000.0,
                    "payment_frequency": 0.95,
                },
                "temporal_features": [],
                "relational_features": {},
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/predict",
                json=sample_data,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                self.log_result(
                    "Model Explainability",
                    False,
                    f"Prediction failed with status {response.status_code}",
                )
                return False

            result = response.json()

            # Check for explanation fields
            if "explanation" not in result:
                self.log_result(
                    "Model Explainability", False, "Missing explanation in response"
                )
                return False

            explanation = result["explanation"]

            # Check for SHAP values or feature importance
            required_explanation_fields = ["feature_importance", "shap_values"]
            if not any(field in explanation for field in required_explanation_fields):
                self.log_result(
                    "Model Explainability",
                    False,
                    "Missing feature importance or SHAP values",
                )
                return False

            self.log_result(
                "Model Explainability", True, "Model explanations are provided"
            )
            return True

        except Exception as e:
            self.log_result("Model Explainability", False, f"Exception: {str(e)}")
            return False

    def validate_sustainability_metrics(self) -> bool:
        """Validate that sustainability metrics are being tracked"""
        try:
            # Check if metrics endpoint includes sustainability metrics
            response = self.session.get(f"{self.base_url}/metrics")

            if response.status_code != 200:
                self.log_result(
                    "Sustainability Metrics",
                    False,
                    f"Metrics endpoint returned {response.status_code}",
                )
                return False

            metrics_content = response.text

            # Look for sustainability-related metrics
            sustainability_metrics = [
                "energy_consumption",
                "carbon_emissions",
                "model_efficiency",
                "prediction_energy",
            ]

            found_metrics = [
                metric for metric in sustainability_metrics if metric in metrics_content
            ]

            if not found_metrics:
                self.log_result(
                    "Sustainability Metrics", False, "No sustainability metrics found"
                )
                return False

            self.log_result(
                "Sustainability Metrics",
                True,
                f"Found sustainability metrics: {found_metrics}",
            )
            return True

        except Exception as e:
            self.log_result("Sustainability Metrics", False, f"Exception: {str(e)}")
            return False

    def validate_security_headers(self) -> bool:
        """Validate security headers are present"""
        try:
            response = self.session.get(f"{self.base_url}/health")

            if response.status_code != 200:
                self.log_result(
                    "Security Headers",
                    False,
                    f"Health endpoint returned {response.status_code}",
                )
                return False

            headers = response.headers

            # Check for important security headers
            security_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": ["DENY", "SAMEORIGIN"],
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": None,  # Just check presence
            }

            missing_headers = []
            for header, expected_value in security_headers.items():
                if header not in headers:
                    missing_headers.append(header)
                elif expected_value and headers[header] not in (
                    expected_value
                    if isinstance(expected_value, list)
                    else [expected_value]
                ):
                    missing_headers.append(f"{header} (incorrect value)")

            if missing_headers:
                self.log_result(
                    "Security Headers",
                    False,
                    f"Missing/incorrect headers: {missing_headers}",
                )
                return False

            self.log_result(
                "Security Headers", True, "Security headers are properly configured"
            )
            return True

        except Exception as e:
            self.log_result("Security Headers", False, f"Exception: {str(e)}")
            return False

    def run_all_validations(self) -> bool:
        """Run all production validations"""
        print("🚀 Running production validation suite...")
        print("=" * 60)

        validations = [
            ("API Health & Readiness", self.validate_api_health),
            ("Prediction Accuracy", self.validate_prediction_accuracy),
            ("Batch Processing", self.validate_batch_processing),
            ("Response Times (SLA)", self.validate_response_times),
            ("Model Explainability", self.validate_model_explainability),
            ("Sustainability Metrics", self.validate_sustainability_metrics),
            ("Security Headers", self.validate_security_headers),
        ]

        passed_count = 0
        total_count = len(validations)

        for validation_name, validation_func in validations:
            print(f"\n🔍 Validating: {validation_name}")
            try:
                if validation_func():
                    passed_count += 1
            except Exception as e:
                self.log_result(validation_name, False, f"Unexpected error: {str(e)}")

        print("\n" + "=" * 60)
        print(f"📊 Validation Results: {passed_count}/{total_count} passed")

        # Generate summary report
        self.generate_validation_report()

        return passed_count == total_count

    def generate_validation_report(self):
        """Generate a detailed validation report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "total_tests": len(self.validation_results),
            "passed_tests": len([r for r in self.validation_results if r["passed"]]),
            "failed_tests": len(
                [r for r in self.validation_results if not r["passed"]]
            ),
            "results": self.validation_results,
        }

        # Save report to file
        report_filename = f"production_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Detailed report saved to: {report_filename}")

        # Print summary
        if report["failed_tests"] > 0:
            print("\n❌ Failed Validations:")
            for result in self.validation_results:
                if not result["passed"]:
                    print(f"   • {result['test']}: {result['message']}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python production_validation.py <base_url>")
        sys.exit(1)

    base_url = sys.argv[1]
    validator = ProductionValidator(base_url)

    if validator.run_all_validations():
        print("\n🎉 All production validations passed!")
        sys.exit(0)
    else:
        print("\n❌ Some production validations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
