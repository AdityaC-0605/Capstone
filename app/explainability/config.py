"""Configuration for explainability service."""

from dataclasses import dataclass


@dataclass
class ExplainabilityConfig:
    """Centralized SHAP and explanation settings."""

    background_sample_size: int = 16
    max_features_to_display: int = 10
    model_type: str = "ensemble"
    shap_nsamples: int = 32
