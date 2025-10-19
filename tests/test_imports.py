"""
Test basic imports to ensure the application can start without errors.
"""


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


def test_flower_import():
    """Test that flower can be imported with the new version."""
    try:
        import flower

        print(
            f"✅ Flower imported successfully, version: {flower.__version__}"
        )
        assert flower.__version__ is not None
    except ImportError as e:
        print(f"❌ Flower import error: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error with flower: {e}")
        raise


if __name__ == "__main__":
    test_basic_imports()
    test_flower_import()
    print("All import tests passed!")
