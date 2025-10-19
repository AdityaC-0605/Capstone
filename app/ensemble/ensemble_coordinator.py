"""
Ensemble model coordinator for combining multiple models with weighted averaging,
stacking, and blending methods. Includes model contribution tracking for explainability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import warnings
from abc import ABC, abstractmethod
import pickle

# ML imports
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    from ..models.dnn_model import DNNModel, DNNTrainer, DNNConfig
    from ..models.lstm_model import LSTMModel, LSTMTrainer, LSTMConfig
    from ..models.gnn_model import GNNModel, GNNTrainer, GNNConfig
    from ..models.tcn_model import TCNModel, TCNTrainer, TCNConfig
    from ..models.lightgbm_model import LightGBMModel, LightGBMTrainer, LightGBMConfig
    from ..core.interfaces import BaseModel, TrainingMetrics
    from ..core.logging import get_logger, get_audit_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from models.dnn_model import DNNModel, DNNTrainer, DNNConfig
    from models.lstm_model import LSTMModel, LSTMTrainer, LSTMConfig
    from models.gnn_model import GNNModel, GNNTrainer, GNNConfig
    from models.tcn_model import TCNModel, TCNTrainer, TCNConfig
    from models.lightgbm_model import LightGBMModel, LightGBMTrainer, LightGBMConfig
    from core.interfaces import BaseModel, TrainingMetrics
    from core.logging import get_logger, get_audit_logger

    # Create minimal implementations for testing
    class MockAuditLogger:
        def log_model_operation(self, **kwargs):
            pass

    def get_audit_logger():
        return MockAuditLogger()


logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class ModelInfo:
    """Information about a model in the ensemble."""

    model_id: str
    model: Any  # Can be BaseModel or sklearn model
    model_type: str
    weight: float = 1.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    prediction_time: float = 0.0
    memory_usage: float = 0.0
    energy_consumption: float = 0.0
    is_active: bool = True


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""

    # Ensemble method
    ensemble_method: str = (
        "weighted_average"  # 'weighted_average', 'stacking', 'blending'
    )

    # Weight optimization
    optimize_weights: bool = True
    weight_optimization_method: str = (
        "validation_performance"  # 'validation_performance', 'bayesian', 'grid_search'
    )

    # Stacking configuration
    meta_learner: str = (
        "logistic_regression"  # 'logistic_regression', 'random_forest', 'neural_network'
    )
    use_cross_validation_stacking: bool = True
    cv_folds: int = 5

    # Blending configuration
    holdout_ratio: float = 0.2  # For blending holdout set

    # Performance thresholds
    min_model_performance: float = 0.6  # Minimum AUC to include model
    diversity_threshold: float = 0.1  # Minimum diversity between models

    # Explainability
    track_contributions: bool = True
    contribution_method: str = (
        "shapley"  # 'shapley', 'weight_based', 'prediction_variance'
    )

    # Model management
    max_models: int = 10
    auto_remove_poor_models: bool = True
    performance_decay_factor: float = 0.95  # For time-weighted performance

    # Saving and loading
    save_ensemble: bool = True
    ensemble_path: str = "models/ensemble"


@dataclass
class EnsembleResult:
    """Result of ensemble training and evaluation."""

    success: bool
    ensemble: Optional["EnsembleModel"]
    config: EnsembleConfig
    individual_performances: Dict[str, Dict[str, float]]
    ensemble_performance: Dict[str, float]
    model_weights: Dict[str, float]
    model_contributions: Dict[str, float]
    diversity_metrics: Dict[str, float]
    training_time_seconds: float
    ensemble_path: Optional[str]
    message: str


class BaseEnsembleMethod(ABC):
    """Abstract base class for ensemble methods."""

    @abstractmethod
    def fit(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Fit the ensemble method."""
        pass

    @abstractmethod
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        pass

    @abstractmethod
    def predict_proba(self, predictions: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities."""
        pass


class WeightedAverageEnsemble(BaseEnsembleMethod):
    """Weighted average ensemble method."""

    def __init__(self, optimize_weights: bool = True):
        self.optimize_weights = optimize_weights
        self.weights = None

    def fit(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        Fit weighted average ensemble.

        Args:
            predictions: Shape (n_samples, n_models) - model predictions
            targets: Shape (n_samples,) - true targets
        """
        n_models = predictions.shape[1]

        if self.optimize_weights:
            # Optimize weights using validation performance
            self.weights = self._optimize_weights(predictions, targets)
        else:
            # Equal weights
            self.weights = np.ones(n_models) / n_models

    def _optimize_weights(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Optimize ensemble weights using grid search."""
        from scipy.optimize import minimize

        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize
            ensemble_pred = np.dot(predictions, weights)
            return -roc_auc_score(targets, ensemble_pred)  # Minimize negative AUC

        n_models = predictions.shape[1]
        initial_weights = np.ones(n_models) / n_models

        # Constraints: weights sum to 1 and are non-negative
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return result.x
        else:
            logger.warning("Weight optimization failed, using equal weights")
            return initial_weights

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Make binary predictions."""
        proba = self.predict_proba(predictions)
        return (proba[:, 1] > 0.5).astype(int)

    def predict_proba(self, predictions: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.weights is None:
            raise ValueError("Ensemble not fitted")

        # Weighted average of predictions
        ensemble_pred = np.dot(predictions, self.weights)

        # Convert to probabilities
        neg_probs = 1 - ensemble_pred
        pos_probs = ensemble_pred

        return np.column_stack([neg_probs, pos_probs])


class StackingEnsemble(BaseEnsembleMethod):
    """Stacking ensemble method with meta-learner."""

    def __init__(self, meta_learner: str = "logistic_regression", cv_folds: int = 5):
        self.meta_learner_name = meta_learner
        self.cv_folds = cv_folds
        self.meta_learner = None
        self._create_meta_learner()

    def _create_meta_learner(self):
        """Create meta-learner based on configuration."""
        if self.meta_learner_name == "logistic_regression":
            self.meta_learner = LogisticRegression(random_state=42)
        elif self.meta_learner_name == "random_forest":
            self.meta_learner = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
        else:
            # Default to logistic regression
            self.meta_learner = LogisticRegression(random_state=42)

    def fit(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        Fit stacking ensemble using cross-validation.

        Args:
            predictions: Shape (n_samples, n_models) - model predictions
            targets: Shape (n_samples,) - true targets
        """
        # Use cross-validation to create meta-features
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        meta_features = np.zeros_like(predictions)

        for train_idx, val_idx in cv.split(predictions, targets):
            # For stacking, we use the predictions as features directly
            meta_features[val_idx] = predictions[val_idx]

        # Train meta-learner on meta-features
        self.meta_learner.fit(meta_features, targets)

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Make binary predictions."""
        if self.meta_learner is None:
            raise ValueError("Ensemble not fitted")

        return self.meta_learner.predict(predictions)

    def predict_proba(self, predictions: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.meta_learner is None:
            raise ValueError("Ensemble not fitted")

        return self.meta_learner.predict_proba(predictions)


class BlendingEnsemble(BaseEnsembleMethod):
    """Blending ensemble method using holdout set."""

    def __init__(self, meta_learner: str = "logistic_regression"):
        self.meta_learner_name = meta_learner
        self.meta_learner = None
        self._create_meta_learner()

    def _create_meta_learner(self):
        """Create meta-learner based on configuration."""
        if self.meta_learner_name == "logistic_regression":
            self.meta_learner = LogisticRegression(random_state=42)
        elif self.meta_learner_name == "random_forest":
            self.meta_learner = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
        else:
            self.meta_learner = LogisticRegression(random_state=42)

    def fit(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        Fit blending ensemble using holdout predictions.

        Args:
            predictions: Shape (n_samples, n_models) - holdout predictions
            targets: Shape (n_samples,) - holdout targets
        """
        # Train meta-learner directly on holdout predictions
        self.meta_learner.fit(predictions, targets)

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Make binary predictions."""
        if self.meta_learner is None:
            raise ValueError("Ensemble not fitted")

        return self.meta_learner.predict(predictions)

    def predict_proba(self, predictions: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.meta_learner is None:
            raise ValueError("Ensemble not fitted")

        return self.meta_learner.predict_proba(predictions)


class EnsembleModel:
    """Main ensemble model coordinator."""

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.models: Dict[str, ModelInfo] = {}
        self.ensemble_method: Optional[BaseEnsembleMethod] = None
        self.is_fitted = False
        self.feature_names = None
        self.training_history = []

        # Initialize ensemble method
        self._initialize_ensemble_method()

    def _initialize_ensemble_method(self):
        """Initialize the ensemble method based on configuration."""
        if self.config.ensemble_method == "weighted_average":
            self.ensemble_method = WeightedAverageEnsemble(
                optimize_weights=self.config.optimize_weights
            )
        elif self.config.ensemble_method == "stacking":
            self.ensemble_method = StackingEnsemble(
                meta_learner=self.config.meta_learner, cv_folds=self.config.cv_folds
            )
        elif self.config.ensemble_method == "blending":
            self.ensemble_method = BlendingEnsemble(
                meta_learner=self.config.meta_learner
            )
        else:
            raise ValueError(f"Unknown ensemble method: {self.config.ensemble_method}")

    def add_model(
        self,
        model_id: str,
        model: Any,
        model_type: str,
        weight: float = 1.0,
        performance_metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Add a model to the ensemble.

        Args:
            model_id: Unique identifier for the model
            model: The trained model object
            model_type: Type of model (e.g., 'dnn', 'lstm', 'lightgbm')
            weight: Initial weight for the model
            performance_metrics: Performance metrics for the model

        Returns:
            bool: True if model was added successfully
        """
        if len(self.models) >= self.config.max_models:
            logger.warning(
                f"Maximum number of models ({self.config.max_models}) reached"
            )
            return False

        if model_id in self.models:
            logger.warning(f"Model {model_id} already exists, updating...")

        # Check minimum performance requirement
        performance_metrics = performance_metrics or {}
        auc_score = performance_metrics.get("roc_auc", 0.0)

        if auc_score < self.config.min_model_performance:
            logger.warning(
                f"Model {model_id} performance ({auc_score:.3f}) below threshold "
                f"({self.config.min_model_performance})"
            )
            if not self.config.auto_remove_poor_models:
                return False

        # Create model info
        model_info = ModelInfo(
            model_id=model_id,
            model=model,
            model_type=model_type,
            weight=weight,
            performance_metrics=performance_metrics,
            is_active=auc_score >= self.config.min_model_performance,
        )

        self.models[model_id] = model_info
        logger.info(f"Added model {model_id} ({model_type}) to ensemble")

        # Reset fitted status since ensemble changed
        self.is_fitted = False

        return True

    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the ensemble."""
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found in ensemble")
            return False

        del self.models[model_id]
        logger.info(f"Removed model {model_id} from ensemble")

        # Reset fitted status
        self.is_fitted = False

        return True

    def get_model_predictions(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Get predictions from all active models."""
        predictions = {}

        for model_id, model_info in self.models.items():
            if not model_info.is_active:
                continue

            try:
                model = model_info.model

                # Handle different model types
                if hasattr(model, "predict_proba"):
                    # Neural network models or sklearn models
                    if isinstance(model, (DNNModel, LSTMModel, GNNModel, TCNModel)):
                        # Convert to tensor if needed
                        if isinstance(X, pd.DataFrame):
                            X_tensor = torch.FloatTensor(X.values)
                        else:
                            X_tensor = torch.FloatTensor(X)

                        probs = model.predict_proba(X_tensor)
                        if isinstance(probs, torch.Tensor):
                            probs = probs.cpu().numpy()
                        predictions[model_id] = probs[
                            :, 1
                        ]  # Positive class probability
                    else:
                        # Sklearn-like models
                        probs = model.predict_proba(X)
                        predictions[model_id] = probs[:, 1]

                elif hasattr(model, "predict"):
                    # Models with only predict method
                    pred = model.predict(X)
                    if isinstance(pred, torch.Tensor):
                        pred = pred.cpu().numpy()
                    predictions[model_id] = pred

                else:
                    logger.warning(f"Model {model_id} has no predict method")
                    continue

            except Exception as e:
                logger.error(f"Error getting predictions from model {model_id}: {e}")
                continue

        return predictions

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Fit the ensemble on training data.

        Args:
            X: Training features
            y: Training targets
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")

        # Get predictions from all models
        model_predictions = self.get_model_predictions(X)

        if len(model_predictions) == 0:
            raise ValueError("No active models with valid predictions")

        # Convert to array format
        model_ids = list(model_predictions.keys())
        predictions_array = np.column_stack(
            [model_predictions[mid] for mid in model_ids]
        )

        # Convert targets to numpy array
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Fit ensemble method
        self.ensemble_method.fit(predictions_array, y_array)

        self.is_fitted = True
        self.feature_names = getattr(X, "columns", None)

        logger.info(
            f"Ensemble fitted with {len(model_predictions)} models using {self.config.ensemble_method}"
        )

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")

        # Get predictions from all models
        model_predictions = self.get_model_predictions(X)

        # Convert to array format (same order as training)
        model_ids = [mid for mid in self.models.keys() if mid in model_predictions]
        predictions_array = np.column_stack(
            [model_predictions[mid] for mid in model_ids]
        )

        # Get ensemble predictions
        return self.ensemble_method.predict(predictions_array)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get ensemble prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")

        # Get predictions from all models
        model_predictions = self.get_model_predictions(X)

        # Convert to array format
        model_ids = [mid for mid in self.models.keys() if mid in model_predictions]
        predictions_array = np.column_stack(
            [model_predictions[mid] for mid in model_ids]
        )

        # Get ensemble probabilities
        return self.ensemble_method.predict_proba(predictions_array)

    def get_model_contributions(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate model contributions to ensemble predictions.

        Args:
            X: Input features

        Returns:
            Dictionary mapping model_id to contribution score
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")

        contributions = {}

        if self.config.contribution_method == "weight_based":
            # Use ensemble weights as contributions
            if isinstance(self.ensemble_method, WeightedAverageEnsemble):
                model_ids = [
                    mid for mid in self.models.keys() if self.models[mid].is_active
                ]
                for i, model_id in enumerate(model_ids):
                    contributions[model_id] = float(self.ensemble_method.weights[i])
            else:
                # Equal contributions for non-weighted methods
                active_models = [
                    mid for mid in self.models.keys() if self.models[mid].is_active
                ]
                equal_contrib = 1.0 / len(active_models)
                for model_id in active_models:
                    contributions[model_id] = equal_contrib

        elif self.config.contribution_method == "prediction_variance":
            # Calculate contributions based on prediction variance
            model_predictions = self.get_model_predictions(X)
            ensemble_pred = self.predict_proba(X)[:, 1]

            total_variance = 0.0
            variances = {}

            for model_id, pred in model_predictions.items():
                variance = np.var(pred - ensemble_pred)
                variances[model_id] = variance
                total_variance += variance

            # Normalize variances to get contributions
            for model_id, variance in variances.items():
                contributions[model_id] = (
                    variance / total_variance if total_variance > 0 else 0.0
                )

        else:
            # Default: equal contributions
            active_models = [
                mid for mid in self.models.keys() if self.models[mid].is_active
            ]
            equal_contrib = 1.0 / len(active_models)
            for model_id in active_models:
                contributions[model_id] = equal_contrib

        return contributions

    def calculate_diversity_metrics(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate diversity metrics between models."""
        model_predictions = self.get_model_predictions(X)

        if len(model_predictions) < 2:
            return {"pairwise_correlation": 0.0, "disagreement_rate": 0.0}

        # Convert to array
        pred_array = np.column_stack(list(model_predictions.values()))

        # Calculate pairwise correlations
        correlations = []
        n_models = pred_array.shape[1]

        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr = np.corrcoef(pred_array[:, i], pred_array[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        avg_correlation = np.mean(correlations) if correlations else 0.0

        # Calculate disagreement rate (for binary predictions)
        binary_preds = (pred_array > 0.5).astype(int)
        disagreements = 0
        total_pairs = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreements += np.sum(binary_preds[:, i] != binary_preds[:, j])
                total_pairs += len(binary_preds)

        disagreement_rate = disagreements / total_pairs if total_pairs > 0 else 0.0

        return {
            "pairwise_correlation": avg_correlation,
            "disagreement_rate": disagreement_rate,
            "diversity_score": 1.0 - avg_correlation,  # Higher is more diverse
        }

    def save_ensemble(self, path: Optional[str] = None) -> str:
        """Save the ensemble model."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before saving")

        # Create save path
        save_path = path or self.config.ensemble_path
        ensemble_dir = Path(save_path)
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Save ensemble state
        ensemble_file = ensemble_dir / "ensemble.pkl"

        # Prepare data for saving
        save_data = {
            "config": self.config,
            "models": self.models,
            "ensemble_method": self.ensemble_method,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "training_history": self.training_history,
        }

        with open(ensemble_file, "wb") as f:
            pickle.dump(save_data, f)

        # Save metadata
        metadata = {
            "ensemble_method": self.config.ensemble_method,
            "num_models": len(self.models),
            "active_models": sum(1 for m in self.models.values() if m.is_active),
            "model_types": list(set(m.model_type for m in self.models.values())),
            "saved_at": datetime.now().isoformat(),
        }

        metadata_file = ensemble_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Ensemble saved to {ensemble_file}")
        return str(ensemble_file)

    def load_ensemble(self, path: str) -> "EnsembleModel":
        """Load a saved ensemble model."""
        ensemble_path = Path(path)

        if ensemble_path.is_file():
            ensemble_file = ensemble_path
        else:
            ensemble_file = ensemble_path / "ensemble.pkl"

        # Load ensemble
        with open(ensemble_file, "rb") as f:
            save_data = pickle.load(f)

        # Restore state
        self.config = save_data["config"]
        self.models = save_data["models"]
        self.ensemble_method = save_data["ensemble_method"]
        self.is_fitted = save_data["is_fitted"]
        self.feature_names = save_data.get("feature_names")
        self.training_history = save_data.get("training_history", [])

        logger.info(f"Ensemble loaded from {ensemble_file}")
        return self


class EnsembleTrainer:
    """Trainer for ensemble models."""

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()

    def train_and_evaluate(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> EnsembleResult:
        """
        Train and evaluate ensemble model.

        Args:
            models: Dictionary of {model_id: trained_model}
            X: Feature data
            y: Target data
            test_size: Test set size

        Returns:
            EnsembleResult with training results
        """
        start_time = datetime.now()

        try:
            logger.info("Starting ensemble training and evaluation")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # For stacking/blending, split training data further
            if self.config.ensemble_method in ["stacking", "blending"]:
                if self.config.ensemble_method == "blending":
                    # Use holdout set for blending
                    X_train, X_blend, y_train, y_blend = train_test_split(
                        X_train,
                        y_train,
                        test_size=self.config.holdout_ratio,
                        random_state=42,
                        stratify=y_train,
                    )
                else:
                    X_blend, y_blend = X_train, y_train
            else:
                X_blend, y_blend = X_train, y_train

            logger.info(
                f"Data split - Train: {len(X_train)}, Blend: {len(X_blend)}, Test: {len(X_test)}"
            )

            # Create ensemble
            ensemble = EnsembleModel(self.config)

            # Evaluate individual models and add to ensemble
            individual_performances = {}

            for model_id, model in models.items():
                try:
                    # Evaluate individual model
                    perf_metrics = self._evaluate_individual_model(
                        model, X_test, y_test, model_id
                    )
                    individual_performances[model_id] = perf_metrics

                    # Determine model type
                    model_type = self._get_model_type(model)

                    # Add to ensemble
                    success = ensemble.add_model(
                        model_id=model_id,
                        model=model,
                        model_type=model_type,
                        performance_metrics=perf_metrics,
                    )

                    if success:
                        logger.info(
                            f"Added {model_id} to ensemble (AUC: {perf_metrics.get('roc_auc', 0.0):.4f})"
                        )
                    else:
                        logger.warning(f"Failed to add {model_id} to ensemble")

                except Exception as e:
                    logger.error(f"Error evaluating model {model_id}: {e}")
                    continue

            if len(ensemble.models) == 0:
                raise ValueError("No models successfully added to ensemble")

            # Fit ensemble
            ensemble.fit(X_blend, y_blend)

            # Evaluate ensemble
            ensemble_performance = self._evaluate_ensemble(ensemble, X_test, y_test)

            # Get model weights and contributions
            model_weights = self._get_model_weights(ensemble)
            model_contributions = ensemble.get_model_contributions(X_test)

            # Calculate diversity metrics
            diversity_metrics = ensemble.calculate_diversity_metrics(X_test)

            # Save ensemble if requested
            ensemble_path = None
            if self.config.save_ensemble:
                ensemble_path = ensemble.save_ensemble()

            training_time = (datetime.now() - start_time).total_seconds()

            # Log completion
            audit_logger.log_model_operation(
                user_id="system",
                model_id="ensemble_coordinator",
                operation="ensemble_training_completed",
                success=True,
                details={
                    "training_time_seconds": training_time,
                    "ensemble_auc": ensemble_performance.get("roc_auc", 0.0),
                    "num_models": len(ensemble.models),
                    "ensemble_method": self.config.ensemble_method,
                    "diversity_score": diversity_metrics.get("diversity_score", 0.0),
                },
            )

            logger.info(f"Ensemble training completed in {training_time:.2f} seconds")
            logger.info(f"Ensemble AUC: {ensemble_performance.get('roc_auc', 0.0):.4f}")

            return EnsembleResult(
                success=True,
                ensemble=ensemble,
                config=self.config,
                individual_performances=individual_performances,
                ensemble_performance=ensemble_performance,
                model_weights=model_weights,
                model_contributions=model_contributions,
                diversity_metrics=diversity_metrics,
                training_time_seconds=training_time,
                ensemble_path=ensemble_path,
                message="Ensemble training completed successfully",
            )

        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Ensemble training failed: {str(e)}"
            logger.error(error_message)

            return EnsembleResult(
                success=False,
                ensemble=None,
                config=self.config,
                individual_performances={},
                ensemble_performance={},
                model_weights={},
                model_contributions={},
                diversity_metrics={},
                training_time_seconds=training_time,
                ensemble_path=None,
                message=error_message,
            )

    def _evaluate_individual_model(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_id: str
    ) -> Dict[str, float]:
        """Evaluate individual model performance."""
        try:
            # Get predictions
            if hasattr(model, "predict_proba"):
                if isinstance(model, (DNNModel, LSTMModel, GNNModel, TCNModel)):
                    # Neural network models
                    X_tensor = torch.FloatTensor(X_test.values)
                    probs = model.predict_proba(X_tensor)
                    if isinstance(probs, torch.Tensor):
                        probs = probs.cpu().numpy()
                    y_pred_proba = probs[:, 1]
                    y_pred = (y_pred_proba > 0.5).astype(int)
                else:
                    # Sklearn-like models
                    probs = model.predict_proba(X_test)
                    y_pred_proba = probs[:, 1]
                    y_pred = model.predict(X_test)
            else:
                # Models with only predict method
                y_pred = model.predict(X_test)
                if isinstance(y_pred, torch.Tensor):
                    y_pred = y_pred.cpu().numpy()
                y_pred_proba = y_pred  # Assume predictions are probabilities

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model {model_id}: {e}")
            return {
                "roc_auc": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

    def _evaluate_ensemble(
        self, ensemble: EnsembleModel, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        try:
            # Get ensemble predictions
            y_pred = ensemble.predict(X_test)
            y_pred_proba = ensemble.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating ensemble: {e}")
            return {
                "roc_auc": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

    def _get_model_type(self, model: Any) -> str:
        """Determine model type from model object."""
        if isinstance(model, DNNModel):
            return "dnn"
        elif isinstance(model, LSTMModel):
            return "lstm"
        elif isinstance(model, GNNModel):
            return "gnn"
        elif isinstance(model, TCNModel):
            return "tcn"
        elif isinstance(model, LightGBMModel):
            return "lightgbm"
        elif hasattr(model, "__class__"):
            return model.__class__.__name__.lower()
        else:
            return "unknown"

    def _get_model_weights(self, ensemble: EnsembleModel) -> Dict[str, float]:
        """Get model weights from ensemble."""
        weights = {}

        if isinstance(ensemble.ensemble_method, WeightedAverageEnsemble):
            active_models = [
                mid for mid in ensemble.models.keys() if ensemble.models[mid].is_active
            ]
            if (
                hasattr(ensemble.ensemble_method, "weights")
                and ensemble.ensemble_method.weights is not None
            ):
                for i, model_id in enumerate(active_models):
                    weights[model_id] = float(ensemble.ensemble_method.weights[i])
            else:
                # Equal weights
                equal_weight = 1.0 / len(active_models)
                for model_id in active_models:
                    weights[model_id] = equal_weight
        else:
            # For stacking/blending, weights are implicit in meta-learner
            active_models = [
                mid for mid in ensemble.models.keys() if ensemble.models[mid].is_active
            ]
            equal_weight = 1.0 / len(active_models)
            for model_id in active_models:
                weights[model_id] = equal_weight

        return weights


# Utility functions
def create_ensemble_from_results(
    results: Dict[str, Any], config: Optional[EnsembleConfig] = None
) -> EnsembleModel:
    """
    Create ensemble from training results.

    Args:
        results: Dictionary of {model_id: training_result}
        config: Ensemble configuration

    Returns:
        EnsembleModel instance
    """
    ensemble = EnsembleModel(config)

    for model_id, result in results.items():
        if hasattr(result, "success") and result.success and hasattr(result, "model"):
            model_type = getattr(result, "model_type", "unknown")
            performance_metrics = getattr(result, "test_metrics", {})

            ensemble.add_model(
                model_id=model_id,
                model=result.model,
                model_type=model_type,
                performance_metrics=performance_metrics,
            )

    return ensemble


def train_ensemble_from_models(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[EnsembleConfig] = None,
) -> EnsembleResult:
    """
    Convenience function to train ensemble from models.

    Args:
        models: Dictionary of {model_id: trained_model}
        X: Feature data
        y: Target data
        config: Ensemble configuration

    Returns:
        EnsembleResult
    """
    trainer = EnsembleTrainer(config)
    return trainer.train_and_evaluate(models, X, y)


def get_default_ensemble_config() -> EnsembleConfig:
    """Get default ensemble configuration."""
    return EnsembleConfig()


def get_weighted_ensemble_config() -> EnsembleConfig:
    """Get weighted average ensemble configuration."""
    return EnsembleConfig(
        ensemble_method="weighted_average",
        optimize_weights=True,
        weight_optimization_method="validation_performance",
    )


def get_stacking_ensemble_config() -> EnsembleConfig:
    """Get stacking ensemble configuration."""
    return EnsembleConfig(
        ensemble_method="stacking",
        meta_learner="logistic_regression",
        use_cross_validation_stacking=True,
        cv_folds=5,
    )


def get_blending_ensemble_config() -> EnsembleConfig:
    """Get blending ensemble configuration."""
    return EnsembleConfig(
        ensemble_method="blending", meta_learner="random_forest", holdout_ratio=0.2
    )
