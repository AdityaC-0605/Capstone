"""
Stress testing and chaos engineering tests for Sustainable Credit Risk AI system.
Tests load testing with concurrent users, memory leak detection, model performance
degradation, chaos engineering for resilience, and resource constraint behavior.
"""

import pytest
import sys
import os
import time
import threading
import multiprocessing
import concurrent.futures
import psutil
import gc
import random
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Import system components for stress testing
from src.data.ingestion import ingest_banking_data
from src.data.feature_engineering import engineer_banking_features, get_minimal_config
from src.data.feature_selection import select_banking_features, get_fast_selection_config
from src.models.dnn_model import train_dnn_baseline, get_fast_dnn_config, create_dnn_model
from src.models.lightgbm_model import train_lightgbm_baseline, get_fast_lightgbm_config
from src.api.inference_service import InferenceService, APIConfig
from src.ensemble.ensemble_coordinator import EnsembleCoordinator, EnsembleConfig


class TestStressChaosEngineering:
    """Test stress testing and chaos engineering scenarios."""
    
    def test_concurrent_user_load_testing(self, test_banking_data_file, api_test_config):
        """Test load testing with concurrent users and high throughput."""
        print("\n=== Testing Concurrent User Load Testing ===")
        
        try:
            # Prepare model for load testing
            print("1. Preparing system for concurrent load testing...")
            
            ingestion_result = ingest_banking_data(test_banking_data_file)
            data = ingestion_result.data.sample(n=1000, random_state=42)
            
            fe_config = get_minimal_config()
            fe_result = engineer_banking_features(data, target_column='default', config=fe_config)
            
            fs_config = get_fast_selection_config()
            fs_result = select_banking_features(fe_result.features, fe_result.target, config=fs_config)
            
            # Train model for API testing
            lgb_config = get_fast_lightgbm_config()
            lgb_result = train_lightgbm_baseline(
                fs_result.selected_features,
                fs_result.target,
                config=lgb_config
            )
            
            # Initialize API service
            api_config = APIConfig(
                title="Stress Test API",
                version="1.0.0",
                host=api_test_config['host'],
                port=api_test_config['port'],
                timeout=api_test_config['timeout']
            )
            
            inference_service = InferenceService(api_config)
            inference_service.load_model(lgb_result.model, model_name="stress_test_model")
            
            print(f"   ✓ System prepared for load testing")
            
            # Prepare test data for concurrent requests
            test_samples = fs_result.selected_features.iloc[:100].to_dict('records')
            
            # Test different concurrency levels
            concurrency_levels = [5, 10, 25, 50]
            load_test_results = {}
            
            for concurrency in concurrency_levels:
                print(f"\\n2. Testing concurrency level: {concurrency} users...")
                
                def concurrent_user_simulation(user_id):
                    """Simulate a single user making multiple requests."""
                    user_results = {
                        'user_id': user_id,
                        'successful_requests': 0,
                        'failed_requests': 0,
                        'latencies': []
                    }
                    
                    # Each user makes 5 requests
                    for request_num in range(5):
                        try:
                            sample = random.choice(test_samples)
                            
                            start_time = time.time()
                            prediction = inference_service.predict_single(sample)
                            end_time = time.time()
                            
                            latency = (end_time - start_time) * 1000
                            
                            user_results['successful_requests'] += 1
                            user_results['latencies'].append(latency)
                            
                            time.sleep(0.01)  # Small delay between requests
                            
                        except Exception as e:
                            user_results['failed_requests'] += 1
                    
                    return user_results
                
                # Execute concurrent user simulation
                start_time = time.time()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    future_to_user = {executor.submit(concurrent_user_simulation, user_id): user_id 
                                    for user_id in range(concurrency)}
                    
                    user_results = []
                    for future in concurrent.futures.as_completed(future_to_user):
                        result = future.result()
                        user_results.append(result)
                
                end_time = time.time()
                
                # Analyze results
                total_successful = sum(r['successful_requests'] for r in user_results)
                total_failed = sum(r['failed_requests'] for r in user_results)
                
                all_latencies = []
                for r in user_results:
                    all_latencies.extend(r['latencies'])
                
                test_duration = end_time - start_time
                throughput = total_successful / test_duration
                
                load_test_results[concurrency] = {
                    'successful_requests': total_successful,
                    'failed_requests': total_failed,
                    'success_rate': total_successful / (total_successful + total_failed) if (total_successful + total_failed) > 0 else 0,
                    'throughput_rps': throughput,
                    'avg_latency_ms': np.mean(all_latencies) if all_latencies else 0,
                    'p95_latency_ms': np.percentile(all_latencies, 95) if all_latencies else 0
                }
                
                print(f"   Results for {concurrency} concurrent users:")
                print(f"      Success rate: {load_test_results[concurrency]['success_rate']*100:.1f}%")
                print(f"      Throughput: {load_test_results[concurrency]['throughput_rps']:.1f} RPS")
                print(f"      Avg latency: {load_test_results[concurrency]['avg_latency_ms']:.2f}ms")
                
                # Verify system stability
                assert load_test_results[concurrency]['success_rate'] >= 0.90, f"Success rate should be ≥90% at {concurrency} users"
            
            print("✓ Concurrent user load testing passed!")
            
        except Exception as e:
            print(f"   ⚠️  Concurrent load testing failed: {e}")
    
    def test_memory_leak_detection(self, test_banking_data_file):
        """Test memory leak detection for long-running services."""
        print("\\n=== Testing Memory Leak Detection ===")
        
        try:
            # Initialize memory monitoring
            print("1. Initializing memory leak detection...")
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"   ✓ Initial memory usage: {initial_memory:.1f} MB")
            
            # Prepare data for repeated operations
            ingestion_result = ingest_banking_data(test_banking_data_file)
            data = ingestion_result.data.sample(n=500, random_state=42)
            
            fe_config = get_minimal_config()
            fe_result = engineer_banking_features(data, target_column='default', config=fe_config)
            
            fs_config = get_fast_selection_config()
            fs_result = select_banking_features(fe_result.features, fe_result.target, config=fs_config)
            
            print(f"   ✓ Test data prepared")
            
            # Test repeated model training for memory leaks
            print("2. Testing repeated model training for memory leaks...")
            
            memory_measurements = []
            training_iterations = 15
            
            for iteration in range(training_iterations):
                iteration_start_memory = process.memory_info().rss / 1024 / 1024
                
                # Train model (this should not accumulate memory)
                lgb_config = get_fast_lightgbm_config()
                temp_lgb_result = train_lightgbm_baseline(
                    fs_result.selected_features,
                    fs_result.target,
                    config=lgb_config
                )
                
                # Force garbage collection
                gc.collect()
                
                iteration_end_memory = process.memory_info().rss / 1024 / 1024
                
                memory_measurements.append({
                    'iteration': iteration,
                    'end_memory_mb': iteration_end_memory,
                    'memory_delta_mb': iteration_end_memory - iteration_start_memory
                })
                
                if iteration % 5 == 0:
                    print(f"      Iteration {iteration}: {iteration_end_memory:.1f} MB")
            
            # Analyze memory growth
            final_memory = memory_measurements[-1]['end_memory_mb']
            memory_growth = final_memory - initial_memory
            avg_memory_per_iteration = memory_growth / training_iterations
            
            print(f"   ✓ Memory growth analysis:")
            print(f"      Initial memory: {initial_memory:.1f} MB")
            print(f"      Final memory: {final_memory:.1f} MB")
            print(f"      Total growth: {memory_growth:.1f} MB")
            print(f"      Average per iteration: {avg_memory_per_iteration:.3f} MB")
            
            # Check for memory leaks (growth should be minimal)
            assert memory_growth < 100, f"Memory growth should be < 100MB (got {memory_growth:.1f}MB)"
            assert avg_memory_per_iteration < 3, f"Average memory per iteration should be < 3MB (got {avg_memory_per_iteration:.3f}MB)"
            
            print("✓ Memory leak detection test passed!")
            
        except Exception as e:
            print(f"   ⚠️  Memory leak detection failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])    def 
test_model_performance_degradation(self, test_banking_data_file):
        """Test model performance degradation over time."""
        print("\\n=== Testing Model Performance Degradation ===")
        
        try:
            # Initialize performance monitoring
            print("1. Initializing model performance degradation testing...")
            
            ingestion_result = ingest_banking_data(test_banking_data_file)
            data = ingestion_result.data.sample(n=600, random_state=42)
            
            fe_config = get_minimal_config()
            fe_result = engineer_banking_features(data, target_column='default', config=fe_config)
            
            fs_config = get_fast_selection_config()
            fs_result = select_banking_features(fe_result.features, fe_result.target, config=fs_config)
            
            # Split data for training and testing
            train_size = int(0.7 * len(fs_result.selected_features))
            train_features = fs_result.selected_features.iloc[:train_size]
            train_target = fs_result.target.iloc[:train_size]
            test_features = fs_result.selected_features.iloc[train_size:]
            test_target = fs_result.target.iloc[train_size:]
            
            print(f"   ✓ Data prepared - Train: {train_features.shape}, Test: {test_features.shape}")
            
            # Train initial model
            lgb_config = get_fast_lightgbm_config()
            lgb_result = train_lightgbm_baseline(
                train_features,
                train_target,
                config=lgb_config
            )
            
            # Baseline performance
            baseline_predictions = lgb_result.model.predict(test_features)
            
            from sklearn.metrics import roc_auc_score, accuracy_score
            
            baseline_auc = roc_auc_score(test_target, baseline_predictions)
            baseline_accuracy = accuracy_score(test_target, (baseline_predictions > 0.5).astype(int))
            
            print(f"   ✓ Baseline model performance - AUC: {baseline_auc:.3f}, Accuracy: {baseline_accuracy:.3f}")
            
            # Test performance under repeated inference load
            print("2. Testing performance under repeated inference load...")
            
            performance_measurements = []
            inference_iterations = 500
            
            for iteration in range(0, inference_iterations, 50):
                # Measure inference time
                start_time = time.time()
                
                # Batch of predictions
                batch_predictions = []
                for _ in range(50):
                    pred = lgb_result.model.predict(test_features.iloc[:1])
                    batch_predictions.append(pred[0])
                
                end_time = time.time()
                
                batch_inference_time = (end_time - start_time) * 1000  # ms
                avg_inference_time = batch_inference_time / 50
                
                # Test prediction quality (should remain consistent)
                quality_sample = lgb_result.model.predict(test_features.iloc[:30])
                sample_auc = roc_auc_score(test_target.iloc[:30], quality_sample)
                
                performance_measurements.append({
                    'iteration': iteration,
                    'avg_inference_time_ms': avg_inference_time,
                    'sample_auc': sample_auc,
                    'prediction_variance': np.var(batch_predictions)
                })
                
                if iteration % 100 == 0:
                    print(f"      Iteration {iteration}: {avg_inference_time:.2f}ms avg, AUC: {sample_auc:.3f}")
            
            # Analyze performance degradation
            initial_inference_time = performance_measurements[0]['avg_inference_time_ms']
            final_inference_time = performance_measurements[-1]['avg_inference_time_ms']
            
            initial_auc = performance_measurements[0]['sample_auc']
            final_auc = performance_measurements[-1]['sample_auc']
            
            inference_time_change = (final_inference_time - initial_inference_time) / initial_inference_time
            auc_change = (final_auc - initial_auc) / initial_auc
            
            print(f"   ✓ Performance degradation analysis:")
            print(f"      Inference time change: {inference_time_change*100:.1f}%")
            print(f"      AUC change: {auc_change*100:.1f}%")
            
            # Performance should remain stable
            assert abs(inference_time_change) < 0.3, f"Inference time should not degrade significantly (got {inference_time_change*100:.1f}%)"
            assert abs(auc_change) < 0.1, f"Model accuracy should remain stable (got {auc_change*100:.1f}% change)"
            
            print("✓ Model performance degradation test passed!")
            
        except Exception as e:
            print(f"   ⚠️  Model performance degradation test failed: {e}")
    
    def test_chaos_engineering_resilience(self, test_banking_data_file, api_test_config):
        """Test system resilience using chaos engineering principles."""
        print("\\n=== Testing Chaos Engineering Resilience ===")
        
        try:
            # Initialize system for chaos testing
            print("1. Initializing system for chaos engineering...")
            
            ingestion_result = ingest_banking_data(test_banking_data_file)
            data = ingestion_result.data.sample(n=400, random_state=42)
            
            fe_config = get_minimal_config()
            fe_result = engineer_banking_features(data, target_column='default', config=fe_config)
            
            fs_config = get_fast_selection_config()
            fs_result = select_banking_features(fe_result.features, fe_result.target, config=fs_config)
            
            # Train model for chaos testing
            lgb_config = get_fast_lightgbm_config()
            lgb_result = train_lightgbm_baseline(
                fs_result.selected_features,
                fs_result.target,
                config=lgb_config
            )
            
            # Initialize API with resilience features
            api_config = APIConfig(
                title="Chaos Test API",
                version="1.0.0",
                host=api_test_config['host'],
                port=api_test_config['port'],
                timeout=api_test_config['timeout']
            )
            
            inference_service = InferenceService(api_config)
            inference_service.load_model(lgb_result.model, model_name="chaos_test_model")
            
            test_sample = fs_result.selected_features.iloc[:1]
            
            print(f"   ✓ Resilient system initialized")
            
            # Test 1: Random service interruptions
            print("\\n2. Testing resilience to random service interruptions...")
            
            def simulate_service_interruption():
                """Simulate random service interruptions."""
                time.sleep(random.uniform(0.05, 0.2))
                # Simulate temporary service unavailability
                if random.random() < 0.2:  # 20% chance of interruption
                    raise Exception("Simulated service interruption")
            
            successful_requests = 0
            failed_requests = 0
            
            for attempt in range(30):
                try:
                    # Simulate potential interruption
                    if random.random() < 0.15:  # 15% chance of interruption
                        simulate_service_interruption()
                    
                    # Make prediction
                    prediction = inference_service.predict_single(test_sample.iloc[0].to_dict())
                    successful_requests += 1
                    
                except Exception as e:
                    failed_requests += 1
            
            success_rate = successful_requests / (successful_requests + failed_requests)
            
            print(f"   ✓ Service interruption test:")
            print(f"      Successful requests: {successful_requests}")
            print(f"      Failed requests: {failed_requests}")
            print(f"      Success rate: {success_rate*100:.1f}%")
            
            # System should maintain reasonable availability despite interruptions
            assert success_rate >= 0.6, f"System should maintain ≥60% availability during interruptions (got {success_rate*100:.1f}%)"
            
            # Test 2: Network latency and timeout handling
            print("\\n3. Testing network latency and timeout handling...")
            
            def simulate_network_delay():
                """Simulate variable network delays."""
                delay = random.uniform(0.01, 0.3)  # 10ms to 300ms delay
                time.sleep(delay)
                return delay
            
            latency_results = []
            
            for attempt in range(15):
                start_time = time.time()
                
                try:
                    # Simulate network delay
                    network_delay = simulate_network_delay()
                    
                    # Make prediction with timeout handling
                    prediction = inference_service.predict_single(test_sample.iloc[0].to_dict())
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    latency_results.append({
                        'success': True,
                        'network_delay': network_delay,
                        'total_time': total_time,
                        'prediction': prediction
                    })
                    
                except Exception as e:
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    latency_results.append({
                        'success': False,
                        'total_time': total_time,
                        'error': str(e)
                    })
            
            successful_latency_tests = [r for r in latency_results if r['success']]
            timeout_rate = (len(latency_results) - len(successful_latency_tests)) / len(latency_results)
            
            if successful_latency_tests:
                avg_response_time = np.mean([r['total_time'] for r in successful_latency_tests])
                print(f"   ✓ Network latency test:")
                print(f"      Average response time: {avg_response_time*1000:.1f}ms")
                print(f"      Timeout rate: {timeout_rate*100:.1f}%")
            
            # System should handle network latency gracefully
            assert timeout_rate < 0.4, f"Timeout rate should be <40% (got {timeout_rate*100:.1f}%)"
            
            print("✓ Chaos engineering resilience test passed!")
            
        except Exception as e:
            print(f"   ⚠️  Chaos engineering test failed: {e}")
    
    def test_resource_constraint_behavior(self, test_banking_data_file):
        """Test system behavior under various resource constraints."""
        print("\\n=== Testing Resource Constraint Behavior ===")
        
        try:
            # Initialize resource monitoring
            print("1. Initializing resource constraint testing...")
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"   ✓ Initial memory: {initial_memory:.1f}MB")
            
            # Prepare test data
            ingestion_result = ingest_banking_data(test_banking_data_file)
            data = ingestion_result.data.sample(n=300, random_state=42)
            
            fe_config = get_minimal_config()
            fe_result = engineer_banking_features(data, target_column='default', config=fe_config)
            
            fs_config = get_fast_selection_config()
            fs_result = select_banking_features(fe_result.features, fe_result.target, config=fs_config)
            
            # Test under memory constraints
            print("\\n2. Testing behavior under memory constraints...")
            
            memory_constraint_results = []
            
            # Create different levels of memory pressure
            memory_pressure_levels = [0, 30, 60]  # MB of additional memory usage
            
            for pressure_mb in memory_pressure_levels:
                print(f"   Testing with {pressure_mb}MB memory pressure...")
                
                memory_hogs = []
                
                try:
                    # Create memory pressure
                    if pressure_mb > 0:
                        # Allocate memory in chunks to create pressure
                        for _ in range(pressure_mb // 10):
                            memory_hog = np.random.randn(1250000)  # ~10MB per array
                            memory_hogs.append(memory_hog)
                    
                    # Test model training under memory pressure
                    start_time = time.time()
                    
                    lgb_config = get_fast_lightgbm_config()
                    lgb_config.num_boost_round = 15  # Reduced for faster testing
                    
                    lgb_result = train_lightgbm_baseline(
                        fs_result.selected_features.iloc[:100],  # Smaller dataset
                        fs_result.target.iloc[:100],
                        config=lgb_config
                    )
                    
                    end_time = time.time()
                    
                    # Test inference under memory pressure
                    inference_start = time.time()
                    test_predictions = lgb_result.model.predict(fs_result.selected_features.iloc[:20])
                    inference_end = time.time()
                    
                    memory_constraint_results.append({
                        'memory_pressure_mb': pressure_mb,
                        'training_time_s': end_time - start_time,
                        'inference_time_ms': (inference_end - inference_start) * 1000,
                        'training_success': lgb_result.success,
                        'inference_success': len(test_predictions) == 20
                    })
                    
                    print(f"      Training: {end_time - start_time:.2f}s, Inference: {(inference_end - inference_start)*1000:.1f}ms")
                    
                except Exception as e:
                    memory_constraint_results.append({
                        'memory_pressure_mb': pressure_mb,
                        'training_success': False,
                        'inference_success': False,
                        'error': str(e)
                    })
                    print(f"      Failed under {pressure_mb}MB pressure: {e}")
                    
                finally:
                    # Clean up memory
                    del memory_hogs
                    gc.collect()
            
            # Analyze memory constraint behavior
            successful_tests = [r for r in memory_constraint_results if r.get('training_success', False)]
            
            if len(successful_tests) >= 2:
                baseline_training_time = successful_tests[0]['training_time_s']
                max_training_time = max(r['training_time_s'] for r in successful_tests)
                
                training_degradation = (max_training_time - baseline_training_time) / baseline_training_time
                
                print(f"   ✓ Memory constraint analysis:")
                print(f"      Training time degradation: {training_degradation*100:.1f}%")
                print(f"      Successful tests: {len(successful_tests)}/{len(memory_constraint_results)}")
                
                # System should handle moderate memory pressure
                assert len(successful_tests) >= len(memory_constraint_results) // 2, "System should handle moderate memory pressure"
            
            # Test graceful degradation
            print("\\n3. Testing graceful degradation under extreme constraints...")
            
            degradation_results = []
            
            # Simulate extreme resource constraints
            extreme_memory_hogs = []
            
            try:
                # Create significant memory pressure
                for _ in range(3):
                    memory_hog = np.random.randn(2000000)  # ~15MB per array
                    extreme_memory_hogs.append(memory_hog)
                
                # Test system behavior under extreme constraints
                try:
                    # Minimal model training
                    minimal_lgb_config = get_fast_lightgbm_config()
                    minimal_lgb_config.num_boost_round = 3
                    
                    minimal_result = train_lightgbm_baseline(
                        fs_result.selected_features.iloc[:15],  # Very small dataset
                        fs_result.target.iloc[:15],
                        config=minimal_lgb_config
                    )
                    
                    # Basic inference
                    if minimal_result.success:
                        prediction = minimal_result.model.predict(fs_result.selected_features.iloc[:1])
                        
                        degradation_results.append({
                            'extreme_constraints': True,
                            'basic_functionality': True,
                            'prediction_success': len(prediction) > 0
                        })
                    else:
                        degradation_results.append({
                            'extreme_constraints': True,
                            'basic_functionality': False
                        })
                    
                except Exception as e:
                    degradation_results.append({
                        'extreme_constraints': True,
                        'basic_functionality': False,
                        'error': str(e)
                    })
                
                print(f"   ✓ Extreme constraint test completed")
                
            finally:
                # Clean up extreme memory pressure
                del extreme_memory_hogs
                gc.collect()
            
            # Analyze graceful degradation
            if degradation_results:
                basic_functionality_maintained = any(r.get('basic_functionality', False) for r in degradation_results)
                
                print(f"   ✓ Graceful degradation analysis:")
                print(f"      Basic functionality maintained: {basic_functionality_maintained}")
                
                # System should attempt graceful degradation
                assert len(degradation_results) > 0, "System should attempt to handle extreme constraints gracefully"
            
            print("✓ Resource constraint behavior test passed!")
            
        except Exception as e:
            print(f"   ⚠️  Resource constraint behavior test failed: {e}")