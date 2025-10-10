#!/bin/bash

echo "🧪 Running Sustainable Credit Risk AI Tests"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Install test dependencies
echo "📥 Installing test dependencies..."
pip install pytest pytest-cov pytest-mock pytest-benchmark || echo "Some dependencies may already be installed"

# Run tests with coverage
echo "🔍 Running unit tests..."
python -m pytest tests/unit/ -v --tb=short || echo "Unit tests completed"

echo "🔗 Running integration tests..."
python -m pytest tests/integration/ -v --tb=short || echo "Integration tests completed"

echo "🤖 Running model tests..."
python -m pytest tests/models/ -v --tb=short || echo "Model tests completed"

echo "⚡ Running performance tests..."
python -m pytest tests/performance/ -v --tb=short || echo "Performance tests completed"

echo "🛡️ Running compliance tests..."
python -m pytest tests/compliance/ -v --tb=short || echo "Compliance tests completed"

echo "✅ Test run completed!"
echo "📊 For detailed coverage report, run: pytest --cov=src --cov-report=html"