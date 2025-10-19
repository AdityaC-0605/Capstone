"""
Core interfaces and abstract base classes for the sustainable credit risk AI system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


@dataclass
class PredictionResult:
    """Result of a credit risk prediction."""

    application_id: str
    risk_score: float
    risk_category: str  # 'low', 'medium', 'high'
    confidence: float
    model_contributions: Dict[str, float]
    feature_importance: Dict[str, float]
    processing_time_ms: float
    energy_consumed_mwh: float


@dataclass
class TrainingMetrics:
    """Metrics collected during model training."""

    experiment_id: str
    model_type: str
    epoch: int
    train_loss: float
    val_loss: float
    auc_roc: float
    f1_score: float
    precision: float
    recall: float
    energy_consumed_kwh: float
    carbon_emissions_kg: float
    training_time_seconds: float


class BaseModel(ABC, torch.nn.Module):
    """Abstract base class for all neural network models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass

    @abstractmethod
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        pass


class DataProcessor(ABC):
    """Abstract base class for data processing components."""

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process input data."""
        pass

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate data quality."""
        pass


class ExplainabilityProvider(ABC):
    """Abstract base class for explainability components."""

    @abstractmethod
    def explain_prediction(
        self, model: BaseModel, input_data: torch.Tensor
    ) -> Dict[str, Any]:
        """Generate explanation for a prediction."""
        pass

    @abstractmethod
    def get_global_importance(
        self, model: BaseModel, dataset: torch.Tensor
    ) -> Dict[str, float]:
        """Get global feature importance."""
        pass


class SustainabilityMonitor(ABC):
    """Abstract base class for sustainability monitoring."""

    @abstractmethod
    def start_tracking(self, experiment_id: str) -> None:
        """Start energy tracking for an experiment."""
        pass

    @abstractmethod
    def stop_tracking(self, experiment_id: str) -> Dict[str, float]:
        """Stop tracking and return metrics."""
        pass

    @abstractmethod
    def get_carbon_footprint(self, energy_kwh: float, region: str) -> float:
        """Calculate carbon footprint."""
        pass


class FederatedLearningNode(ABC):
    """Abstract base class for federated learning nodes."""

    @abstractmethod
    def train_local_model(self, epochs: int) -> Dict[str, torch.Tensor]:
        """Train local model and return updates."""
        pass

    @abstractmethod
    def update_global_model(
        self, global_weights: Dict[str, torch.Tensor]
    ) -> None:
        """Update local model with global weights."""
        pass


class SecurityProvider(ABC):
    """Abstract base class for security and privacy components."""

    @abstractmethod
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data."""
        pass

    @abstractmethod
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        pass

    @abstractmethod
    def anonymize_data(self, data: Any) -> Any:
        """Anonymize sensitive data."""
        pass
