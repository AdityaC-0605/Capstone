"""Ensemble package for model coordination and combination."""

from .ensemble_coordinator import (
    BlendingEnsemble,
    EnsembleConfig,
    EnsembleModel,
    EnsembleResult,
    EnsembleTrainer,
    ModelInfo,
    StackingEnsemble,
    WeightedAverageEnsemble,
    create_ensemble_from_results,
    get_blending_ensemble_config,
    get_default_ensemble_config,
    get_stacking_ensemble_config,
    get_weighted_ensemble_config,
    train_ensemble_from_models,
)

__all__ = [
    "EnsembleModel",
    "EnsembleTrainer",
    "EnsembleConfig",
    "EnsembleResult",
    "ModelInfo",
    "WeightedAverageEnsemble",
    "StackingEnsemble",
    "BlendingEnsemble",
    "create_ensemble_from_results",
    "train_ensemble_from_models",
    "get_default_ensemble_config",
    "get_weighted_ensemble_config",
    "get_stacking_ensemble_config",
    "get_blending_ensemble_config",
]
