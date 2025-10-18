"""
Main entry point for the Sustainable Credit Risk AI System.
"""

import os
import sys
from pathlib import Path

# Add app to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.core.config import load_config
from app.core.logging import get_logger
from app.security.security_manager import get_security_manager


def main():
    """Main application entry point."""
    # Load configuration
    config = load_config()
    logger = get_logger(__name__)
    
    logger.info("Starting Sustainable Credit Risk AI System")
    logger.info(f"Environment: {config.environment.value}")
    
    # Initialize security manager
    security_manager = get_security_manager()
    
    # Generate initial security report
    security_report = security_manager.generate_security_report()
    logger.info(f"Security initialization complete: {security_report}")
    
    logger.info("System initialization complete")


if __name__ == "__main__":
    main()