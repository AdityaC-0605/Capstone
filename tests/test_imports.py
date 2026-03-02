"""
Test basic imports to ensure the application can start without errors.
"""

import pytest


def test_basic_imports():
    """Test that basic modules can be imported."""
    try:
        import sys
        from pathlib import Path

        # Add app to Python path
        sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

        # Test core imports
        from app.core.logging import get_logger

        # Test that we can create basic objects
        logger = get_logger(__name__)
        assert logger is not None

        print("✅ Basic imports successful")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


def test_sustainability_metrics_import():
    """Test that the shared sustainability metrics can be imported."""
    try:
        from app.sustainability.metrics import ks_statistic
        import numpy as np

        # Sanity check with trivially separable data
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        ks = ks_statistic(y_true, y_prob)
        assert 0.0 <= ks <= 1.0, f"KS out of range: {ks}"
        print("✅ Sustainability metrics imported and validated")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise


if __name__ == "__main__":
    test_basic_imports()
    test_sustainability_metrics_import()
    print("All import tests passed!")

