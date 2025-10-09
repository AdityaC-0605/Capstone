#!/usr/bin/env python3
"""
Comprehensive test runner for all end-to-end tests in the Sustainable Credit Risk AI system.
Executes all integration, performance, security, and chaos engineering tests.
"""

import sys
import os
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_test_suite(test_file, test_name, verbose=True):
    """Run a specific test suite and return results."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {test_name}")
    print(f"File: {test_file}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run pytest on the specific test file
        cmd = [
            sys.executable, "-m", "pytest", 
            test_file, 
            "-v" if verbose else "-q",
            "--tb=short",
            "--no-header"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        test_result = {
            'test_name': test_name,
            'test_file': test_file,
            'success': success,
            'duration_seconds': duration,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if success:
            print(f"✓ {test_name} PASSED in {duration:.2f}s")
        else:
            print(f"✗ {test_name} FAILED in {duration:.2f}s")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"Errors: {result.stderr}")
        
        return test_result
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✗ {test_name} ERROR in {duration:.2f}s: {e}")
        
        return {
            'test_name': test_name,
            'test_file': test_file,
            'success': False,
            'duration_seconds': duration,
            'error': str(e)
        }


def main():
    """Run all end-to-end test suites."""
    print("🚀 Starting Comprehensive End-to-End Test Suite")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check if Bank_data.csv exists
    data_file = Path("Bank_data.csv")
    if not data_file.exists():
        print(f"⚠️  Warning: {data_file} not found. Some tests may fail.")
    else:
        print(f"✓ Test data file found: {data_file}")
    
    print("=" * 80)
    
    # Define test suites to run
    test_suites = [
        {
            'file': 'tests/test_end_to_end_workflow.py',
            'name': 'End-to-End Workflow Tests',
            'description': 'Complete data-to-prediction pipeline, explainability integration, sustainability monitoring'
        },
        {
            'file': 'tests/test_federated_learning_e2e.py',
            'name': 'Federated Learning End-to-End Tests',
            'description': 'Federated server-client communication, privacy preservation, model aggregation'
        },
        {
            'file': 'tests/test_sustainability_monitoring_e2e.py',
            'name': 'Sustainability Monitoring End-to-End Tests',
            'description': 'Energy tracking, carbon calculation, ESG reporting across full workflows'
        },
        {
            'file': 'tests/test_performance_benchmarking.py',
            'name': 'Performance Benchmarking Tests',
            'description': 'Load testing, model training performance, federated scalability, sustainability targets'
        },
        {
            'file': 'tests/test_security_privacy_validation.py',
            'name': 'Security and Privacy Validation Tests',
            'description': 'Data encryption, anonymization, API security, vulnerability assessment'
        },
        {
            'file': 'tests/test_stress_chaos_engineering.py',
            'name': 'Stress Testing and Chaos Engineering Tests',
            'description': 'Concurrent load testing, memory leak detection, resilience testing, resource constraints'
        }
    ]
    
    # Run all test suites
    all_results = []
    total_start_time = time.time()
    
    for i, suite in enumerate(test_suites, 1):
        print(f"\n[{i}/{len(test_suites)}] {suite['name']}")
        print(f"Description: {suite['description']}")
        
        # Check if test file exists
        test_file_path = Path(suite['file'])
        if not test_file_path.exists():
            print(f"⚠️  Test file not found: {suite['file']}")
            all_results.append({
                'test_name': suite['name'],
                'test_file': suite['file'],
                'success': False,
                'error': 'Test file not found'
            })
            continue
        
        # Run the test suite
        result = run_test_suite(suite['file'], suite['name'])
        all_results.append(result)
        
        # Short pause between test suites
        time.sleep(1)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("📊 TEST EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = [r for r in all_results if r.get('success', False)]
    failed_tests = [r for r in all_results if not r.get('success', False)]
    
    print(f"Total Test Suites: {len(all_results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {len(successful_tests)/len(all_results)*100:.1f}%")
    print(f"Total Duration: {total_duration:.2f} seconds")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for result in all_results:
        status = "✓ PASS" if result.get('success', False) else "✗ FAIL"
        duration = result.get('duration_seconds', 0)
        print(f"  {status} {result['test_name']} ({duration:.2f}s)")
        
        if not result.get('success', False) and 'error' in result:
            print(f"    Error: {result['error']}")
    
    # Failed test details
    if failed_tests:
        print(f"\n❌ FAILED TEST DETAILS:")
        for result in failed_tests:
            print(f"\nTest: {result['test_name']}")
            print(f"File: {result['test_file']}")
            
            if 'return_code' in result:
                print(f"Return Code: {result['return_code']}")
            
            if 'stderr' in result and result['stderr']:
                print(f"Error Output:")
                print(result['stderr'][:500] + "..." if len(result['stderr']) > 500 else result['stderr'])
    
    # Save detailed results to JSON
    results_file = Path("test_results") / f"e2e_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    summary_report = {
        'execution_timestamp': datetime.now().isoformat(),
        'total_duration_seconds': total_duration,
        'total_test_suites': len(all_results),
        'successful_test_suites': len(successful_tests),
        'failed_test_suites': len(failed_tests),
        'success_rate': len(successful_tests)/len(all_results) if all_results else 0,
        'test_results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Final status
    if len(successful_tests) == len(all_results):
        print(f"\n🎉 ALL TESTS PASSED! System is ready for production.")
        return 0
    elif len(successful_tests) >= len(all_results) * 0.8:
        print(f"\n⚠️  Most tests passed ({len(successful_tests)}/{len(all_results)}). Review failed tests.")
        return 1
    else:
        print(f"\n❌ Multiple test failures ({len(failed_tests)}/{len(all_results)}). System needs attention.")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)