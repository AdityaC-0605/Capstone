#!/usr/bin/env python3
"""
Performance validation script for production deployment
Validates that the system meets performance SLAs
"""

import argparse
import requests
import time
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any


class PerformanceValidator:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
    def single_request_latency(self) -> float:
        """Measure latency of a single prediction request"""
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
            "relational_features": {}
        }
        
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url}/api/v1/predict",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}")
        
        return (end_time - start_time) * 1000  # Convert to milliseconds
    
    def concurrent_requests_test(self, num_requests: int = 50, num_workers: int = 10) -> Dict[str, Any]:
        """Test concurrent request handling"""
        latencies = []
        errors = 0
        
        def make_request():
            try:
                return self.single_request_latency()
            except Exception:
                return None
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    latencies.append(result)
                else:
                    errors += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if not latencies:
            raise Exception("All requests failed")
        
        return {
            'total_requests': num_requests,
            'successful_requests': len(latencies),
            'failed_requests': errors,
            'total_time_seconds': total_time,
            'requests_per_second': len(latencies) / total_time,
            'avg_latency_ms': statistics.mean(latencies),
            'median_latency_ms': statistics.median(latencies),
            'p95_latency_ms': self.percentile(latencies, 95),
            'p99_latency_ms': self.percentile(latencies, 99),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'error_rate_percent': (errors / num_requests) * 100
        }
    
    def percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    def validate_sla_requirements(self, results: Dict[str, Any]) -> bool:
        """Validate that results meet SLA requirements"""
        sla_requirements = {
            'avg_latency_ms': 100,      # Average latency < 100ms
            'p95_latency_ms': 200,      # 95th percentile < 200ms
            'p99_latency_ms': 500,      # 99th percentile < 500ms
            'min_requests_per_second': 50,  # Minimum throughput
            'max_error_rate_percent': 1.0   # Error rate < 1%
        }
        
        violations = []
        
        for metric, threshold in sla_requirements.items():
            if metric.startswith('max_'):
                actual_metric = metric[4:]  # Remove 'max_' prefix
                if results[actual_metric] > threshold:
                    violations.append(f"{actual_metric}: {results[actual_metric]:.2f} > {threshold}")
            elif metric.startswith('min_'):
                actual_metric = metric[4:]  # Remove 'min_' prefix
                if results[actual_metric] < threshold:
                    violations.append(f"{actual_metric}: {results[actual_metric]:.2f} < {threshold}")
            else:
                if results[metric] > threshold:
                    violations.append(f"{metric}: {results[metric]:.2f} > {threshold}")
        
        if violations:
            print("❌ SLA Violations:")
            for violation in violations:
                print(f"   • {violation}")
            return False
        else:
            print("✅ All SLA requirements met")
            return True
    
    def print_performance_summary(self, results: Dict[str, Any]):
        """Print performance test summary"""
        print("\n📊 Performance Test Results:")
        print("=" * 50)
        print(f"Total Requests: {results['total_requests']}")
        print(f"Successful Requests: {results['successful_requests']}")
        print(f"Failed Requests: {results['failed_requests']}")
        print(f"Error Rate: {results['error_rate_percent']:.2f}%")
        print(f"Total Time: {results['total_time_seconds']:.2f}s")
        print(f"Requests/Second: {results['requests_per_second']:.2f}")
        print()
        print("Latency Statistics:")
        print(f"  Average: {results['avg_latency_ms']:.2f}ms")
        print(f"  Median: {results['median_latency_ms']:.2f}ms")
        print(f"  95th Percentile: {results['p95_latency_ms']:.2f}ms")
        print(f"  99th Percentile: {results['p99_latency_ms']:.2f}ms")
        print(f"  Min: {results['min_latency_ms']:.2f}ms")
        print(f"  Max: {results['max_latency_ms']:.2f}ms")
    
    def run_performance_validation(self, num_requests: int = 100, num_workers: int = 10) -> bool:
        """Run complete performance validation"""
        print(f"🚀 Running performance validation against {self.base_url}")
        print(f"Test parameters: {num_requests} requests, {num_workers} concurrent workers")
        print("=" * 60)
        
        try:
            # Test basic connectivity first
            print("🔍 Testing basic connectivity...")
            latency = self.single_request_latency()
            print(f"✅ Single request latency: {latency:.2f}ms")
            
            # Run concurrent load test
            print(f"\n🔄 Running concurrent load test...")
            results = self.concurrent_requests_test(num_requests, num_workers)
            
            # Print results
            self.print_performance_summary(results)
            
            # Validate SLA requirements
            print("\n🎯 Validating SLA requirements...")
            sla_passed = self.validate_sla_requirements(results)
            
            return sla_passed
            
        except Exception as e:
            print(f"❌ Performance validation failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Validate production performance")
    parser.add_argument("--url", required=True, help="Base URL of the API")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests to send")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    validator = PerformanceValidator(args.url, args.timeout)
    
    if validator.run_performance_validation(args.requests, args.workers):
        print("\n🎉 Performance validation passed!")
        sys.exit(0)
    else:
        print("\n❌ Performance validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()