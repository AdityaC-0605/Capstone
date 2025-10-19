"""
Pytest configuration and shared fixtures for end-to-end testing.
"""

import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="credit_risk_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_banking_data():
    """Create sample banking data for testing."""
    np.random.seed(42)
    n_samples = 1000

    data = {
        "age": np.random.randint(18, 80, n_samples),
        "annual_income_inr": np.random.lognormal(12, 0.5, n_samples),
        "loan_amount_inr": np.random.lognormal(11, 0.7, n_samples),
        "credit_score": np.random.randint(300, 850, n_samples),
        "debt_to_income_ratio": np.random.uniform(0.1, 0.8, n_samples),
        "employment_length": np.random.randint(0, 30, n_samples),
        "home_ownership": np.random.choice(
            ["RENT", "OWN", "MORTGAGE"], n_samples
        ),
        "loan_purpose": np.random.choice(
            ["debt_consolidation", "home_improvement", "major_purchase"],
            n_samples,
        ),
        "default": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    }

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def test_banking_data_file(sample_banking_data, test_data_dir):
    """Save sample data to CSV file."""
    file_path = Path(test_data_dir) / "test_banking_data.csv"
    sample_banking_data.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def energy_tracking_config():
    """Configuration for energy tracking tests."""
    return {
        "track_gpu": True,
        "track_cpu": True,
        "measure_emissions": True,
        "country_iso_code": "USA",
        "region": "california",
    }


@pytest.fixture
def federated_test_config():
    """Configuration for federated learning tests."""
    return {
        "num_clients": 3,
        "local_epochs": 2,
        "communication_rounds": 3,
        "differential_privacy": True,
        "epsilon": 1.0,
    }


@pytest.fixture
def model_configs():
    """Standard model configurations for testing."""
    return {
        "dnn": {
            "hidden_sizes": [64, 32],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 5,
        },
        "lstm": {
            "hidden_size": 32,
            "num_layers": 2,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 5,
        },
        "lightgbm": {
            "num_boost_round": 50,
            "early_stopping_rounds": 10,
            "learning_rate": 0.1,
            "num_leaves": 31,
        },
    }


@pytest.fixture
def api_test_config():
    """Configuration for API testing."""
    return {"host": "127.0.0.1", "port": 8000, "timeout": 30, "max_retries": 3}
