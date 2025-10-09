#!/usr/bin/env python3
"""
Performance threshold checker for load test results
Validates that performance metrics meet acceptable thresholds
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Dict, Any, List


class PerformanceThresholdChecker:
    def __init__(self, thresholds: Dict[str, Any]):
        self.thresholds = thresholds
        self.violations = []
    
    def check_response_time_thresholds(self, stats: Dict[str, Any]) -> bool:
        """Check response time thresholds"""
        passed = True
        
        # Check average response time
        avg_response_time = stats.get('avg_response_time', 0)
        if avg_response_time > self.thresholds['avg_response_time_ms']:
            self.violations.append(
                f"Average response time {avg_response_time}ms exceeds threshold "
                f"{self.thresholds['avg_response_time_ms']}ms"
            )
            passed = False
        
        # Check 95th percentile response time
        p95_response_time = stats.get('95th_percentile_response_time', 0)
        if p95_response_time > self.thresholds['p95_response_time_ms']:
            self.violations.append(
                f"95th percentile response time {p95_response_time}ms exceeds threshold "
                f"{self.thresholds['p95_response_time_ms']}ms"
            )
            passed = False
        
        # Check 99th percentile response time
        p99_response_time = stats.get('99th_percentile_response_time', 0)
        if p99_response_time > self.thresholds['p99_response_time_ms']:
            self.violations.append(
                f"99th percentile response time {p99_response_time}ms exceeds threshold "
                f"{self.thresholds['p99_response_time_ms']}ms"
            )
            passed = False
        
        return passed
    
    def check_throughput_thresholds(self, stats: Dict[str, Any]) -> bool:
        """Check throughput thresholds"""
        passed = True
        
        # Check requests per second
        rps = stats.get('requests_per_second', 0)
        if rps < self.thresholds['min_requests_per_second']:
            self.violations.append(
                f"Requests per second {rps} below minimum threshold "
                f"{self.thresholds['min_requests_per_second']}"
            )
            passed = False
        
        return passed
    
    def check_error_rate_thresholds(self, stats: Dict[str, Any]) -> bool:
        """Check error rate thresholds"""
        passed = True
        
        # Check overall error rate
        error_rate = stats.get('error_rate_percent', 0)
        if error_rate > self.thresholds['max_error_rate_percent']:
            self.violations.append(
                f"Error rate {error_rate}% exceeds maximum threshold "
                f"{self.thresholds['max_error_rate_percent']}%"
            )
            passed = False
        
        return passed
    
    def check_resource_utilization_thresholds(self, stats: Dict[str, Any]) -> bool:
        """Check resource utilization thresholds"""
        passed = True
        
        # Check CPU utilization
        cpu_utilization = stats.get('cpu_utilization_percent', 0)
        if cpu_utilization > self.thresholds['max_cpu_utilization_percent']:
            self.violations.append(
                f"CPU utilization {cpu_utilization}% exceeds maximum threshold "
                f"{self.thresholds['max_cpu_utilization_percent']}%"
            )
            passed = False
        
        # Check memory utilization
        memory_utilization = stats.get('memory_utilization_percent', 0)
        if memory_utilization > self.thresholds['max_memory_utilization_percent']:
            self.violations.append(
                f"Memory utilization {memory_utilization}% exceeds maximum threshold "
                f"{self.thresholds['max_memory_utilization_percent']}%"
            )
            passed = False
        
        return passed
    
    def parse_locust_html_report(self, report_path: str) -> Dict[str, Any]:
        """Parse Locust HTML report to extract performance metrics"""
        try:
            with open(report_path, 'r') as f:
                content = f.read()
            
            stats = {}
            
            # Extract statistics using regex patterns
            # Average response time
            avg_match = re.search(r'Average.*?(\d+\.?\d*)\s*ms', content, re.IGNORECASE)
            if avg_match:
                stats['avg_response_time'] = float(avg_match.group(1))
            
            # 95th percentile
            p95_match = re.search(r'95%.*?(\d+\.?\d*)\s*ms', content, re.IGNORECASE)
            if p95_match:
                stats['95th_percentile_response_time'] = float(p95_match.group(1))
            
            # 99th percentile
            p99_match = re.search(r'99%.*?(\d+\.?\d*)\s*ms', content, re.IGNORECASE)
            if p99_match:
                stats['99th_percentile_response_time'] = float(p99_match.group(1))
            
            # Requests per second
            rps_match = re.search(r'(\d+\.?\d*)\s*RPS', content, re.IGNORECASE)
            if rps_match:
                stats['requests_per_second'] = float(rps_match.group(1))
            
            # Error rate
            error_match = re.search(r'(\d+\.?\d*)%.*?error', content, re.IGNORECASE)
            if error_match:
                stats['error_rate_percent'] = float(error_match.group(1))
            else:
                stats['error_rate_percent'] = 0.0
            
            return stats
            
        except Exception as e:
            print(f"Error parsing Locust report: {e}")
            return {}
    
    def parse_json_report(self, report_path: str) -> Dict[str, Any]:
        """Parse JSON performance report"""
        try:
            with open(report_path, 'r') as f:
                data = json.load(f)
            return data.get('stats', {})
        except Exception as e:
            print(f"Error parsing JSON report: {e}")
            return {}
    
    def check_all_thresholds(self, stats: Dict[str, Any]) -> bool:
        """Check all performance thresholds"""
        print("🔍 Checking performance thresholds...")
        print("=" * 50)
        
        checks = [
            ("Response Time", self.check_response_time_thresholds),
            ("Throughput", self.check_throughput_thresholds),
            ("Error Rate", self.check_error_rate_thresholds),
            ("Resource Utilization", self.check_resource_utilization_thresholds),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"\n📊 Checking: {check_name}")
            try:
                if check_func(stats):
                    print(f"   ✅ {check_name} thresholds passed")
                else:
                    print(f"   ❌ {check_name} thresholds failed")
                    all_passed = False
            except Exception as e:
                print(f"   ⚠️  {check_name} check error: {e}")
                all_passed = False
        
        return all_passed
    
    def print_violations(self):
        """Print all threshold violations"""
        if self.violations:
            print("\n❌ Threshold Violations:")
            print("=" * 50)
            for i, violation in enumerate(self.violations, 1):
                print(f"{i}. {violation}")
        else:
            print("\n✅ No threshold violations found!")
    
    def print_stats_summary(self, stats: Dict[str, Any]):
        """Print performance statistics summary"""
        print("\n📈 Performance Statistics Summary:")
        print("=" * 50)
        
        metrics = [
            ("Average Response Time", "avg_response_time", "ms"),
            ("95th Percentile Response Time", "95th_percentile_response_time", "ms"),
            ("99th Percentile Response Time", "99th_percentile_response_time", "ms"),
            ("Requests Per Second", "requests_per_second", "RPS"),
            ("Error Rate", "error_rate_percent", "%"),
            ("CPU Utilization", "cpu_utilization_percent", "%"),
            ("Memory Utilization", "memory_utilization_percent", "%"),
        ]
        
        for name, key, unit in metrics:
            value = stats.get(key, "N/A")
            if value != "N/A":
                print(f"{name}: {value} {unit}")
            else:
                print(f"{name}: {value}")


def get_default_thresholds() -> Dict[str, Any]:
    """Get default performance thresholds"""
    return {
        # Response time thresholds (milliseconds)
        'avg_response_time_ms': 100,
        'p95_response_time_ms': 200,
        'p99_response_time_ms': 500,
        
        # Throughput thresholds
        'min_requests_per_second': 100,
        
        # Error rate thresholds (percentage)
        'max_error_rate_percent': 1.0,
        
        # Resource utilization thresholds (percentage)
        'max_cpu_utilization_percent': 80,
        'max_memory_utilization_percent': 85,
    }


def main():
    parser = argparse.ArgumentParser(description="Check performance test results against thresholds")
    parser.add_argument("report_file", help="Path to the performance test report file")
    parser.add_argument("--thresholds", help="Path to custom thresholds JSON file")
    parser.add_argument("--format", choices=['html', 'json'], default='html', 
                       help="Report format (default: html)")
    
    args = parser.parse_args()
    
    # Load thresholds
    if args.thresholds:
        try:
            with open(args.thresholds, 'r') as f:
                thresholds = json.load(f)
        except Exception as e:
            print(f"Error loading thresholds file: {e}")
            sys.exit(1)
    else:
        thresholds = get_default_thresholds()
    
    # Initialize checker
    checker = PerformanceThresholdChecker(thresholds)
    
    # Parse report
    if args.format == 'html':
        stats = checker.parse_locust_html_report(args.report_file)
    else:
        stats = checker.parse_json_report(args.report_file)
    
    if not stats:
        print("❌ Failed to parse performance report")
        sys.exit(1)
    
    # Print statistics summary
    checker.print_stats_summary(stats)
    
    # Check thresholds
    if checker.check_all_thresholds(stats):
        print("\n🎉 All performance thresholds passed!")
        checker.print_violations()
        sys.exit(0)
    else:
        print("\n❌ Performance thresholds failed!")
        checker.print_violations()
        sys.exit(1)


if __name__ == "__main__":
    main()