# End-to-End Test Suite for Sustainable Credit Risk AI

This directory contains comprehensive end-to-end tests for the Sustainable Credit Risk AI system, covering all major components and integration scenarios.

## Test Structure

### Core Test Files

1. **`test_end_to_end_workflow.py`** - Complete workflow testing
   - Data ingestion to prediction pipeline
   - Model performance requirements validation
   - Explainability integration
   - Sustainability monitoring integration

2. **`test_federated_learning_e2e.py`** - Federated learning end-to-end tests
   - Server-client setup and communication
   - Complete federated training cycles
   - Privacy preservation mechanisms
   - Convergence and model quality testing

3. **`test_sustainability_monitoring_e2e.py`** - Sustainability monitoring tests
   - Energy tracking across full workflows
   - Carbon footprint calculation
   - ESG metrics collection and reporting
   - Sustainability optimization recommendations

4. **`test_performance_benchmarking.py`** - Performance benchmarking tests
   - API load testing under realistic traffic
   - Model training performance and resource usage
   - Federated learning scalability
   - Sustainability targets achievement validation

5. **`test_security_privacy_validation.py`** - Security and privacy tests
   - Data encryption and decryption security
   - Data anonymization and privacy preservation
   - API security and authentication mechanisms
   - Vulnerability assessment and penetration testing

6. **`test_stress_chaos_engineering.py`** - Stress and chaos engineering tests
   - Concurrent user load testing
   - Memory leak detection for long-running services
   - Model performance degradation over time
   - System resilience under various failure scenarios
   - Resource constraint behavior testing

### Supporting Files

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`pytest.ini`** - Pytest configuration settings
- **`run_all_e2e_tests.py`** - Comprehensive test runner script
- **`README.md`** - This documentation file

## Running Tests

### Prerequisites

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. Ensure test data is available:
   - `Bank_data.csv` should be in the project root directory

### Running Individual Test Suites

```bash
# Run specific test suite
pytest tests/test_end_to_end_workflow.py -v

# Run with specific markers
pytest -m "e2e" -v
pytest -m "performance" -v
pytest -m "security" -v
```

### Running All Tests

```bash
# Run all end-to-end tests
python tests/run_all_e2e_tests.py

# Or using pytest directly
pytest tests/ -v
```

### Test Configuration

Tests can be configured using environment variables or by modifying the fixtures in `conftest.py`:

- **Data Configuration**: Modify `sample_banking_data` fixture for different test data sizes
- **API Configuration**: Modify `api_test_config` fixture for different API settings
- **Energy Tracking**: Modify `energy_tracking_config` fixture for different monitoring settings
- **Federated Learning**: Modify `federated_test_config` fixture for different FL scenarios

## Test Categories

### Integration Tests (Requirements 1.1-1.5)
- Complete data-to-prediction pipeline validation
- Model performance requirements verification
- Inference latency testing
- System integration validation

### Sustainability Tests (Requirements 2.1-2.5, 9.1-9.5)
- Energy consumption tracking and optimization
- Carbon footprint calculation and reporting
- ESG metrics collection and dashboard functionality
- Sustainability target achievement validation

### Explainability Tests (Requirements 3.1-3.5)
- SHAP value generation and validation
- LIME explanation testing
- Attention mechanism visualization
- Counterfactual explanation generation

### Federated Learning Tests (Requirements 4.1-4.5)
- Privacy-preserving collaborative training
- Differential privacy effectiveness
- Secure communication protocols
- Model aggregation and convergence

### Performance Tests (Requirements 5.1-5.5)
- Real-time inference API performance
- Batch prediction capabilities
- Load testing under realistic traffic
- Scalability and throughput validation

### Security Tests (Requirements 8.1-8.5)
- Data encryption and anonymization
- API security and authentication
- Privacy preservation validation
- Vulnerability assessment

### Deployment Tests (Requirements 10.1-10.5)
- System resilience and fault tolerance
- Resource constraint behavior
- Chaos engineering scenarios
- Production readiness validation

## Expected Test Results

### Performance Benchmarks
- **Inference Latency**: < 100ms per prediction (relaxed for testing environment)
- **Model Accuracy**: AUC-ROC ≥ 0.85 (may be lower with test data)
- **Throughput**: System should handle concurrent users gracefully
- **Memory Usage**: No significant memory leaks during extended operation

### Security Benchmarks
- **Encryption**: All sensitive data should be encrypted at rest and in transit
- **Anonymization**: PII should be properly anonymized with k-anonymity ≥ 5
- **API Security**: All security headers and authentication mechanisms should be in place
- **Vulnerability Assessment**: No critical security vulnerabilities should be found

### Sustainability Benchmarks
- **Energy Efficiency**: Optimized models should show measurable energy reduction
- **Carbon Footprint**: System should track and report carbon emissions
- **ESG Compliance**: ESG reports should be generated successfully
- **Optimization**: System should provide actionable sustainability recommendations

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install pytest psutil scikit-learn pandas numpy
   ```

2. **Missing Test Data**
   - Ensure `Bank_data.csv` is in the project root
   - Tests will use synthetic data if real data is unavailable

3. **Memory Issues**
   - Reduce test data sizes in `conftest.py`
   - Run tests individually instead of all at once

4. **Timeout Issues**
   - Increase timeout values in test configurations
   - Run tests on a machine with adequate resources

### Test Failures

- **Performance Test Failures**: May indicate system resource constraints or configuration issues
- **Security Test Failures**: May indicate missing security components or configuration
- **Integration Test Failures**: May indicate missing dependencies or incorrect system setup

### Debugging

Enable verbose output and detailed error reporting:
```bash
pytest tests/ -v --tb=long --capture=no
```

## Continuous Integration

These tests are designed to be run in CI/CD pipelines. The `run_all_e2e_tests.py` script provides:
- Structured test execution
- Detailed reporting
- JSON output for CI integration
- Appropriate exit codes for pipeline decisions

## Contributing

When adding new tests:
1. Follow the existing test structure and naming conventions
2. Add appropriate fixtures to `conftest.py` if needed
3. Update this README with new test descriptions
4. Ensure tests are deterministic and can run in isolation
5. Add appropriate pytest markers for test categorization