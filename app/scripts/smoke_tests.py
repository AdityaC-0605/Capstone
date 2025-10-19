#!/usr/bin/env python3
"""
Smoke tests for the Sustainable Credit Risk AI API
These tests verify basic functionality after deployment
"""

import argparse
import json
import sys
import time
from typing import Any, Dict

import requests


class SmokeTestRunner:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout

    def test_health_endpoint(self) -> bool:
        """Test the health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ Health endpoint is accessible")
                return True
            else:
                print(f"❌ Health endpoint returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health endpoint failed: {e}")
            return False

    def test_ready_endpoint(self) -> bool:
        """Test the readiness endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/ready")
            if response.status_code == 200:
                print("✅ Ready endpoint is accessible")
                return True
            else:
                print(f"❌ Ready endpoint returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Ready endpoint failed: {e}")
            return False

    def test_docs_endpoint(self) -> bool:
        """Test the API documentation endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/docs")
            if response.status_code == 200:
                print("✅ API documentation is accessible")
                return True
            else:
                print(f"❌ API docs returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API docs failed: {e}")
            return False

    def test_openapi_spec(self) -> bool:
        """Test the OpenAPI specification endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/openapi.json")
            if response.status_code == 200:
                spec = response.json()
                if "openapi" in spec and "paths" in spec:
                    print("✅ OpenAPI specification is valid")
                    return True
                else:
                    print("❌ OpenAPI specification is invalid")
                    return False
            else:
                print(f"❌ OpenAPI spec returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ OpenAPI spec failed: {e}")
            return False

    def test_prediction_endpoint(self) -> bool:
        """Test the prediction endpoint with sample data"""
        try:
            # Sample credit application data
            sample_data = {
                "application_id": "test_001",
                "applicant_id": "applicant_001",
                "loan_amount": 50000.0,
                "loan_term": 36,
                "annual_income": 75000.0,
                "employment_length": 5,
                "home_ownership": "RENT",
                "debt_to_income_ratio": 0.25,
                "credit_history_length": 10,
                "behavioral_features": {
                    "avg_monthly_spending": 3000.0,
                    "payment_frequency": 0.95,
                },
                "temporal_features": [
                    {"month": 1, "spending": 2800.0, "payments": 1},
                    {"month": 2, "spending": 3200.0, "payments": 1},
                ],
                "relational_features": {"guarantors": [], "co_applicants": []},
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/predict",
                json=sample_data,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                required_fields = [
                    "risk_score",
                    "risk_category",
                    "confidence",
                    "explanation",
                ]
                if all(field in result for field in required_fields):
                    print("✅ Prediction endpoint works correctly")
                    return True
                else:
                    print(f"❌ Prediction response missing required fields: {result}")
                    return False
            else:
                print(f"❌ Prediction endpoint returned status {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Prediction endpoint failed: {e}")
            return False

    def test_batch_prediction_endpoint(self) -> bool:
        """Test the batch prediction endpoint"""
        try:
            # Sample batch data
            batch_data = {
                "applications": [
                    {
                        "application_id": "batch_001",
                        "applicant_id": "applicant_001",
                        "loan_amount": 30000.0,
                        "loan_term": 24,
                        "annual_income": 60000.0,
                        "employment_length": 3,
                        "home_ownership": "OWN",
                        "debt_to_income_ratio": 0.20,
                        "credit_history_length": 8,
                        "behavioral_features": {"avg_monthly_spending": 2500.0},
                        "temporal_features": [],
                        "relational_features": {},
                    },
                    {
                        "application_id": "batch_002",
                        "applicant_id": "applicant_002",
                        "loan_amount": 40000.0,
                        "loan_term": 36,
                        "annual_income": 80000.0,
                        "employment_length": 7,
                        "home_ownership": "MORTGAGE",
                        "debt_to_income_ratio": 0.30,
                        "credit_history_length": 12,
                        "behavioral_features": {"avg_monthly_spending": 3500.0},
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

            if response.status_code == 200:
                result = response.json()
                if "predictions" in result and len(result["predictions"]) == 2:
                    print("✅ Batch prediction endpoint works correctly")
                    return True
                else:
                    print(f"❌ Batch prediction response invalid: {result}")
                    return False
            else:
                print(
                    f"❌ Batch prediction endpoint returned status {response.status_code}"
                )
                return False
        except Exception as e:
            print(f"❌ Batch prediction endpoint failed: {e}")
            return False

    def test_metrics_endpoint(self) -> bool:
        """Test the metrics endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                # Check if it's Prometheus format
                content = response.text
                if "http_requests_total" in content or "prediction_latency" in content:
                    print("✅ Metrics endpoint is accessible")
                    return True
                else:
                    print("❌ Metrics endpoint doesn't contain expected metrics")
                    return False
            else:
                print(f"❌ Metrics endpoint returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Metrics endpoint failed: {e}")
            return False

    def test_response_time(self) -> bool:
        """Test API response time"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/health")
            end_time = time.time()

            response_time = (end_time - start_time) * 1000  # Convert to milliseconds

            if (
                response.status_code == 200 and response_time < 1000
            ):  # Less than 1 second
                print(f"✅ Response time is acceptable: {response_time:.2f}ms")
                return True
            else:
                print(f"❌ Response time too slow: {response_time:.2f}ms")
                return False
        except Exception as e:
            print(f"❌ Response time test failed: {e}")
            return False

    def run_all_tests(self) -> bool:
        """Run all smoke tests"""
        print(f"🚀 Running smoke tests against {self.base_url}")
        print("=" * 50)

        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Readiness Check", self.test_ready_endpoint),
            ("API Documentation", self.test_docs_endpoint),
            ("OpenAPI Specification", self.test_openapi_spec),
            ("Single Prediction", self.test_prediction_endpoint),
            ("Batch Prediction", self.test_batch_prediction_endpoint),
            ("Metrics Endpoint", self.test_metrics_endpoint),
            ("Response Time", self.test_response_time),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n🧪 Testing: {test_name}")
            try:
                if test_func():
                    passed += 1
                else:
                    print(f"   Test failed: {test_name}")
            except Exception as e:
                print(f"   Test error: {test_name} - {e}")

        print("\n" + "=" * 50)
        print(f"📊 Results: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All smoke tests passed!")
            return True
        else:
            print("❌ Some smoke tests failed!")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Run smoke tests for the Credit Risk AI API"
    )
    parser.add_argument("--url", required=True, help="Base URL of the API")
    parser.add_argument(
        "--timeout", type=int, default=30, help="Request timeout in seconds"
    )

    args = parser.parse_args()

    runner = SmokeTestRunner(args.url, args.timeout)

    if runner.run_all_tests():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
