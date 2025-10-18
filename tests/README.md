# 🧪 Test Suite

This directory contains the comprehensive test suite for the Sustainable Credit Risk AI System.

## 📁 Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
├── integration/             # Integration tests for component interactions
├── conftest.py             # Test configuration and fixtures
├── pytest.ini             # Pytest configuration
└── README.md               # This file
```

## 🚀 Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only  
pytest tests/integration/

# Skip slow tests
pytest -m "not slow"
```

### Run with Coverage
```bash
pytest --cov=app --cov-report=html
```

## 🏷️ Test Markers

The test suite uses pytest markers to categorize tests:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.security`: Security-related tests
- `@pytest.mark.compliance`: Compliance tests

## 📝 Writing Tests

### Unit Tests
- Test individual functions and classes
- Use mocks for external dependencies
- Keep tests fast and isolated

### Integration Tests
- Test component interactions
- Use real data and services where appropriate
- Test end-to-end workflows

### Test Fixtures
- Use `conftest.py` for shared fixtures
- Create reusable test data
- Set up test environments

## 🔧 Test Configuration

The `pytest.ini` file contains:
- Test discovery patterns
- Markers and their descriptions
- Warning filters
- Output formatting options

## 📊 Test Coverage

Aim for high test coverage:
- Unit tests: >90% coverage
- Integration tests: Cover critical paths
- End-to-end tests: Cover main user workflows

## 🐛 Debugging Tests

```bash
# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Run specific test
pytest tests/unit/test_specific.py::test_function
```
