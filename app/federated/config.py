"""Federated learning configuration."""

from dataclasses import dataclass


@dataclass
class FLConfig:
    """Centralized configuration for federated learning."""

    number_of_clients: int = 3
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    aggregation_rounds: int = 3
    validation_split: float = 0.2

    # Model defaults for the built-in simulation model.
    input_size: int = 20
    hidden_size: int = 32
    random_seed: int = 42

    # Training control.
    enable_early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-3

    # Artifact output.
    best_model_path: str = "model_registry/federated_best.pt"
