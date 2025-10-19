"""
Configuration management system for different environments.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Environment(Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "credit_risk_ai"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"


@dataclass
class ModelConfig:
    """Model configuration."""

    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    model_save_path: str = "models/"
    checkpoint_interval: int = 10


@dataclass
class SecurityConfig:
    """Security configuration."""

    encryption_key_path: str = "keys/encryption.key"
    jwt_secret_key: str = ""
    jwt_expiration_hours: int = 24
    api_key_length: int = 32
    differential_privacy_epsilon: float = 1.0
    enable_audit_logging: bool = True


@dataclass
class FederatedConfig:
    """Federated learning configuration."""

    server_host: str = "localhost"
    server_port: int = 8080
    num_rounds: int = 10
    min_clients: int = 2
    client_fraction: float = 1.0
    secure_aggregation: bool = True


@dataclass
class SustainabilityConfig:
    """Sustainability monitoring configuration."""

    enable_energy_tracking: bool = True
    carbon_region: str = "US"
    energy_threshold_kwh: float = 10.0
    carbon_budget_kg: float = 100.0
    report_interval_hours: int = 24


@dataclass
class APIConfig:
    """API configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout_seconds: int = 30
    max_request_size_mb: int = 100
    rate_limit_per_minute: int = 1000


@dataclass
class Config:
    """Main configuration class."""

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: str = "INFO"

    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    sustainability: SustainabilityConfig = field(
        default_factory=SustainabilityConfig
    )
    api: APIConfig = field(default_factory=APIConfig)

    # Paths
    data_path: str = "data/"
    logs_path: str = "logs/"
    models_path: str = "models/"
    keys_path: str = "keys/"


class ConfigManager:
    """Configuration manager for loading and managing configurations."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/"
        self._config: Optional[Config] = None

    def load_config(self, environment: Optional[str] = None) -> Config:
        """Load configuration for the specified environment."""
        env = environment or os.getenv("ENVIRONMENT", "development")

        # Load base configuration
        base_config = self._load_base_config()

        # Load environment-specific overrides
        env_config = self._load_environment_config(env)

        # Merge configurations
        merged_config = self._merge_configs(base_config, env_config)

        # Apply environment variables
        final_config = self._apply_env_vars(merged_config)

        self._config = final_config
        return final_config

    def get_config(self) -> Config:
        """Get the current configuration."""
        if self._config is None:
            return self.load_config()
        return self._config

    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration."""
        config_file = Path(self.config_path) / "base.yaml"
        if config_file.exists():
            with open(config_file, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_environment_config(self, environment: str) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        config_file = Path(self.config_path) / f"{environment}.yaml"
        if config_file.exists():
            with open(config_file, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _merge_configs(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Config:
        """Merge base and override configurations."""
        merged = {**base, **override}

        # Convert to Config object
        config = Config()

        # Update fields if they exist in merged config
        for field_name, field_value in merged.items():
            if hasattr(config, field_name):
                if isinstance(field_value, dict):
                    # Handle nested configurations
                    nested_config = getattr(config, field_name)
                    for nested_key, nested_value in field_value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(config, field_name, field_value)

        return config

    def _apply_env_vars(self, config: Config) -> Config:
        """Apply environment variable overrides."""
        # Database overrides
        if os.getenv("DB_HOST"):
            config.database.host = os.getenv("DB_HOST")
        if os.getenv("DB_PORT"):
            config.database.port = int(os.getenv("DB_PORT"))
        if os.getenv("DB_PASSWORD"):
            config.database.password = os.getenv("DB_PASSWORD")

        # Security overrides
        if os.getenv("JWT_SECRET_KEY"):
            config.security.jwt_secret_key = os.getenv("JWT_SECRET_KEY")
        if os.getenv("ENCRYPTION_KEY_PATH"):
            config.security.encryption_key_path = os.getenv(
                "ENCRYPTION_KEY_PATH"
            )

        # API overrides
        if os.getenv("API_HOST"):
            config.api.host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            config.api.port = int(os.getenv("API_PORT"))

        return config

    def save_config(self, config: Config, filename: str) -> None:
        """Save configuration to file."""
        config_dict = self._config_to_dict(config)
        config_file = Path(self.config_path) / filename

        # Ensure directory exists
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert Config object to dictionary."""
        result = {}
        for field_name in config.__dataclass_fields__:
            field_value = getattr(config, field_name)
            if hasattr(field_value, "__dataclass_fields__"):
                # Handle nested dataclass
                result[field_name] = self._config_to_dict(field_value)
            elif isinstance(field_value, Enum):
                result[field_name] = field_value.value
            else:
                result[field_name] = field_value
        return result


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> Config:
    """Get the global configuration."""
    return config_manager.get_config()


def load_config(environment: Optional[str] = None) -> Config:
    """Load configuration for the specified environment."""
    return config_manager.load_config(environment)
