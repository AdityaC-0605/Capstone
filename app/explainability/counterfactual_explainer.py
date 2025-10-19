"""
Counterfactual Explanation System for Credit Risk Models.

This module implements comprehensive counterfactual explanation generation
for neural network models, providing "what-if" scenario analysis and decision
boundary exploration to help users understand what changes would flip predictions.
"""

import copy
import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Optimization imports
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

try:
    from ..core.interfaces import BaseModel
    from ..core.logging import get_audit_logger, get_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_audit_logger, get_logger

    # Create minimal implementations for testing
    class MockAuditLogger:
        def log_model_operation(self, **kwargs):
            pass

    def get_audit_logger():
        return MockAuditLogger()


logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual explanation generation."""

    # Generation parameters
    max_iterations: int = 1000
    learning_rate: float = 0.01
    target_class: Optional[int] = (
        None  # None for flip, specific class otherwise
    )

    # Optimization constraints
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    immutable_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)

    # Distance and similarity
    distance_metric: str = "euclidean"  # "euclidean", "manhattan", "cosine"
    lambda_distance: float = 0.1  # Weight for distance penalty
    lambda_sparsity: float = 0.01  # Weight for sparsity penalty

    # Diversity and ranking
    num_counterfactuals: int = 5
    diversity_weight: float = 0.1
    validity_threshold: float = 0.5

    # Method selection
    method: str = "gradient"  # "gradient", "genetic", "dice", "wachter"

    # Categorical handling
    categorical_penalty: float = 1.0

    # Convergence criteria
    tolerance: float = 1e-6
    patience: int = 50

    # Output settings
    explanation_path: str = "explanations/counterfactual"
    save_explanations: bool = True


@dataclass
class CounterfactualExplanation:
    """Container for counterfactual explanation results."""

    # Basic information
    instance_id: str
    original_instance: np.ndarray
    original_prediction: float
    original_class: int

    # Counterfactuals
    counterfactuals: List[np.ndarray]
    counterfactual_predictions: List[float]
    counterfactual_classes: List[int]

    # Analysis metrics
    distances: List[float]
    sparsity_scores: List[float]
    validity_scores: List[float]
    feasibility_scores: List[float]

    # Feature changes
    feature_changes: List[
        Dict[str, Tuple[float, float]]
    ]  # feature -> (original, new)
    changed_features: List[List[str]]

    # Metadata
    generation_method: str
    generation_time: float
    convergence_info: Dict[str, Any]

    # Ranking and selection
    ranked_indices: List[int]
    best_counterfactual_idx: int

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to dictionary."""
        return {
            "instance_id": self.instance_id,
            "original_instance": self.original_instance.tolist(),
            "original_prediction": float(self.original_prediction),
            "original_class": int(self.original_class),
            "counterfactuals": [cf.tolist() for cf in self.counterfactuals],
            "counterfactual_predictions": [
                float(p) for p in self.counterfactual_predictions
            ],
            "counterfactual_classes": [
                int(c) for c in self.counterfactual_classes
            ],
            "distances": [float(d) for d in self.distances],
            "sparsity_scores": [float(s) for s in self.sparsity_scores],
            "validity_scores": [float(v) for v in self.validity_scores],
            "feasibility_scores": [float(f) for f in self.feasibility_scores],
            "feature_changes": self.feature_changes,
            "changed_features": self.changed_features,
            "generation_method": self.generation_method,
            "generation_time": self.generation_time,
            "convergence_info": self.convergence_info,
            "ranked_indices": self.ranked_indices,
            "best_counterfactual_idx": self.best_counterfactual_idx,
            "timestamp": self.timestamp.isoformat(),
        }


class CounterfactualGenerator(ABC):
    """Abstract base class for counterfactual generation methods."""

    def __init__(self, model: nn.Module, config: CounterfactualConfig):
        self.model = model
        self.config = config
        self.model.eval()

        # Device handling
        self.device = next(model.parameters()).device

    @abstractmethod
    def generate(
        self, instance: np.ndarray, feature_names: List[str]
    ) -> List[np.ndarray]:
        """Generate counterfactual explanations for a given instance."""
        pass

    def _get_prediction(self, x: torch.Tensor) -> Tuple[float, int]:
        """Get model prediction and class."""
        with torch.no_grad():
            x = x.to(self.device)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)

            output = self.model(x)
            if len(output.shape) > 1:
                output = output.squeeze()

            prob = torch.sigmoid(output).item()
            pred_class = int(prob > 0.5)

            return prob, pred_class

    def _calculate_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate distance between two instances."""
        if self.config.distance_metric == "euclidean":
            return np.linalg.norm(x1 - x2)
        elif self.config.distance_metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        elif self.config.distance_metric == "cosine":
            return 1 - np.dot(x1, x2) / (
                np.linalg.norm(x1) * np.linalg.norm(x2)
            )
        else:
            return np.linalg.norm(x1 - x2)

    def _calculate_sparsity(
        self,
        original: np.ndarray,
        counterfactual: np.ndarray,
        threshold: float = 1e-6,
    ) -> float:
        """Calculate sparsity score (fraction of changed features)."""
        changes = np.abs(original - counterfactual) > threshold
        return np.sum(changes) / len(original)

    def _apply_constraints(
        self, x: np.ndarray, feature_names: List[str]
    ) -> np.ndarray:
        """Apply feature constraints to ensure feasibility."""
        x_constrained = x.copy()

        # Apply feature ranges
        if self.config.feature_ranges:
            for i, feature_name in enumerate(feature_names):
                if feature_name in self.config.feature_ranges:
                    min_val, max_val = self.config.feature_ranges[feature_name]
                    x_constrained[i] = np.clip(
                        x_constrained[i], min_val, max_val
                    )

        # Handle categorical features (round to nearest integer)
        for feature_name in self.config.categorical_features:
            if feature_name in feature_names:
                idx = feature_names.index(feature_name)
                x_constrained[idx] = np.round(x_constrained[idx])

        return x_constrained


class GradientBasedGenerator(CounterfactualGenerator):
    """Gradient-based counterfactual generation using optimization."""

    def generate(
        self, instance: np.ndarray, feature_names: List[str]
    ) -> List[np.ndarray]:
        """Generate counterfactuals using gradient-based optimization."""

        original_tensor = torch.FloatTensor(instance).to(self.device)
        original_pred, original_class = self._get_prediction(original_tensor)

        # Determine target class
        target_class = (
            1 - original_class
            if self.config.target_class is None
            else self.config.target_class
        )

        counterfactuals = []

        for attempt in range(
            self.config.num_counterfactuals * 2
        ):  # Generate more, select best
            # Initialize with small random perturbation
            x = original_tensor.clone().detach().requires_grad_(True)
            x.data += torch.randn_like(x) * 0.01

            optimizer = optim.Adam([x], lr=self.config.learning_rate)

            best_loss = float("inf")
            patience_counter = 0

            for iteration in range(self.config.max_iterations):
                optimizer.zero_grad()

                # Model prediction
                output = self.model(x.unsqueeze(0)).squeeze()
                pred_prob = torch.sigmoid(output)

                # Classification loss (encourage target class)
                if target_class == 1:
                    class_loss = -torch.log(pred_prob + 1e-8)
                else:
                    class_loss = -torch.log(1 - pred_prob + 1e-8)

                # Distance penalty
                distance_loss = torch.norm(x - original_tensor, p=2)

                # Sparsity penalty (L1 norm)
                sparsity_loss = torch.norm(x - original_tensor, p=1)

                # Total loss
                total_loss = (
                    class_loss
                    + self.config.lambda_distance * distance_loss
                    + self.config.lambda_sparsity * sparsity_loss
                )

                total_loss.backward()
                optimizer.step()

                # Apply constraints
                with torch.no_grad():
                    x_np = x.cpu().numpy()
                    x_constrained = self._apply_constraints(
                        x_np, feature_names
                    )
                    x.data = torch.FloatTensor(x_constrained).to(self.device)

                # Check convergence
                if total_loss.item() < best_loss - self.config.tolerance:
                    best_loss = total_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    break

            # Check if we achieved target class
            final_pred, final_class = self._get_prediction(x)
            if final_class == target_class:
                counterfactuals.append(x.detach().cpu().numpy())

            if len(counterfactuals) >= self.config.num_counterfactuals:
                break

        return counterfactuals


class GeneticAlgorithmGenerator(CounterfactualGenerator):
    """Genetic algorithm-based counterfactual generation."""

    def generate(
        self, instance: np.ndarray, feature_names: List[str]
    ) -> List[np.ndarray]:
        """Generate counterfactuals using genetic algorithm."""

        original_pred, original_class = self._get_prediction(
            torch.FloatTensor(instance)
        )
        target_class = (
            1 - original_class
            if self.config.target_class is None
            else self.config.target_class
        )

        def objective_function(x):
            """Objective function for genetic algorithm."""
            x_tensor = torch.FloatTensor(x).to(self.device)
            pred_prob, pred_class = self._get_prediction(x_tensor)

            # Classification objective
            if target_class == 1:
                class_obj = -pred_prob  # Maximize probability for class 1
            else:
                class_obj = pred_prob  # Minimize probability for class 1

            # Distance penalty
            distance_penalty = (
                self.config.lambda_distance
                * self._calculate_distance(instance, x)
            )

            # Sparsity penalty
            sparsity_penalty = (
                self.config.lambda_sparsity
                * self._calculate_sparsity(instance, x)
            )

            return class_obj + distance_penalty + sparsity_penalty

        # Define bounds
        bounds = []
        for i, feature_name in enumerate(feature_names):
            if (
                self.config.feature_ranges
                and feature_name in self.config.feature_ranges
            ):
                bounds.append(self.config.feature_ranges[feature_name])
            else:
                # Default bounds: ±3 standard deviations from original value
                std_dev = np.std(instance) if np.std(instance) > 0 else 1.0
                bounds.append(
                    (instance[i] - 3 * std_dev, instance[i] + 3 * std_dev)
                )

        counterfactuals = []

        for attempt in range(self.config.num_counterfactuals):
            try:
                result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=self.config.max_iterations // 10,
                    popsize=15,
                    seed=attempt,
                )

                if result.success:
                    candidate = self._apply_constraints(
                        result.x, feature_names
                    )
                    pred_prob, pred_class = self._get_prediction(
                        torch.FloatTensor(candidate)
                    )

                    if pred_class == target_class:
                        counterfactuals.append(candidate)

            except Exception as e:
                logger.warning(
                    f"Genetic algorithm attempt {attempt} failed: {e}"
                )
                continue

        return counterfactuals


class DiceGenerator(CounterfactualGenerator):
    """DiCE (Diverse Counterfactual Explanations) implementation."""

    def generate(
        self, instance: np.ndarray, feature_names: List[str]
    ) -> List[np.ndarray]:
        """Generate diverse counterfactuals using DiCE approach."""

        original_pred, original_class = self._get_prediction(
            torch.FloatTensor(instance)
        )
        target_class = (
            1 - original_class
            if self.config.target_class is None
            else self.config.target_class
        )

        counterfactuals = []

        # Generate multiple counterfactuals with diversity constraint
        for i in range(self.config.num_counterfactuals):
            x = (
                torch.FloatTensor(instance)
                .to(self.device)
                .clone()
                .detach()
                .requires_grad_(True)
            )

            # Add random initialization for diversity
            x.data += torch.randn_like(x) * 0.05 * (i + 1)

            optimizer = optim.Adam([x], lr=self.config.learning_rate)

            for iteration in range(self.config.max_iterations):
                optimizer.zero_grad()

                # Model prediction
                output = self.model(x.unsqueeze(0)).squeeze()
                pred_prob = torch.sigmoid(output)

                # Classification loss
                if target_class == 1:
                    class_loss = -torch.log(pred_prob + 1e-8)
                else:
                    class_loss = -torch.log(1 - pred_prob + 1e-8)

                # Distance penalty
                distance_loss = torch.norm(
                    x - torch.FloatTensor(instance).to(self.device), p=2
                )

                # Diversity penalty (encourage difference from existing counterfactuals)
                diversity_loss = 0
                if counterfactuals:
                    for cf in counterfactuals:
                        cf_tensor = torch.FloatTensor(cf).to(self.device)
                        diversity_loss -= torch.norm(x - cf_tensor, p=2)
                    diversity_loss /= len(counterfactuals)

                # Total loss
                total_loss = (
                    class_loss
                    + self.config.lambda_distance * distance_loss
                    + self.config.diversity_weight * diversity_loss
                )

                total_loss.backward()
                optimizer.step()

                # Apply constraints
                with torch.no_grad():
                    x_np = x.cpu().numpy()
                    x_constrained = self._apply_constraints(
                        x_np, feature_names
                    )
                    x.data = torch.FloatTensor(x_constrained).to(self.device)

            # Check if valid counterfactual
            final_pred, final_class = self._get_prediction(x)
            if final_class == target_class:
                counterfactuals.append(x.detach().cpu().numpy())

        return counterfactuals


class WachterGenerator(CounterfactualGenerator):
    """Wachter et al. counterfactual generation method."""

    def generate(
        self, instance: np.ndarray, feature_names: List[str]
    ) -> List[np.ndarray]:
        """Generate counterfactuals using Wachter et al. method."""

        original_pred, original_class = self._get_prediction(
            torch.FloatTensor(instance)
        )
        target_class = (
            1 - original_class
            if self.config.target_class is None
            else self.config.target_class
        )

        counterfactuals = []

        for attempt in range(self.config.num_counterfactuals * 2):
            # Initialize with original instance
            x = (
                torch.FloatTensor(instance)
                .to(self.device)
                .clone()
                .detach()
                .requires_grad_(True)
            )

            # Add small random perturbation
            x.data += torch.randn_like(x) * 0.01

            optimizer = optim.LBFGS([x], lr=0.1, max_iter=20)

            def closure():
                optimizer.zero_grad()

                # Model prediction
                output = self.model(x.unsqueeze(0)).squeeze()
                pred_prob = torch.sigmoid(output)

                # Loss function from Wachter et al.
                if target_class == 1:
                    prediction_loss = (
                        torch.max(
                            torch.tensor(0.0).to(self.device), 0.5 - pred_prob
                        )
                        ** 2
                    )
                else:
                    prediction_loss = (
                        torch.max(
                            torch.tensor(0.0).to(self.device), pred_prob - 0.5
                        )
                        ** 2
                    )

                # Distance penalty (L1 norm as in original paper)
                distance_loss = torch.norm(
                    x - torch.FloatTensor(instance).to(self.device), p=1
                )

                total_loss = (
                    prediction_loss
                    + self.config.lambda_distance * distance_loss
                )
                total_loss.backward()

                return total_loss

            # Optimize
            for _ in range(self.config.max_iterations // 20):
                optimizer.step(closure)

                # Apply constraints
                with torch.no_grad():
                    x_np = x.cpu().numpy()
                    x_constrained = self._apply_constraints(
                        x_np, feature_names
                    )
                    x.data = torch.FloatTensor(x_constrained).to(self.device)

            # Check validity
            final_pred, final_class = self._get_prediction(x)
            if final_class == target_class:
                counterfactuals.append(x.detach().cpu().numpy())

            if len(counterfactuals) >= self.config.num_counterfactuals:
                break

        return counterfactuals


class CounterfactualExplainer:
    """Main class for counterfactual explanation generation and analysis."""

    def __init__(
        self, model: nn.Module, config: Optional[CounterfactualConfig] = None
    ):
        self.model = model
        self.config = config or CounterfactualConfig()

        # Initialize generator based on method
        self.generator = self._create_generator()

        logger.info(
            f"Counterfactual explainer initialized with method: {self.config.method}"
        )

    def _create_generator(self) -> CounterfactualGenerator:
        """Create appropriate counterfactual generator."""
        if self.config.method == "gradient":
            return GradientBasedGenerator(self.model, self.config)
        elif self.config.method == "genetic":
            return GeneticAlgorithmGenerator(self.model, self.config)
        elif self.config.method == "dice":
            return DiceGenerator(self.model, self.config)
        elif self.config.method == "wachter":
            return WachterGenerator(self.model, self.config)
        else:
            raise ValueError(
                f"Unknown counterfactual method: {self.config.method}"
            )

    def explain(
        self,
        instance: np.ndarray,
        feature_names: List[str],
        instance_id: str = None,
    ) -> CounterfactualExplanation:
        """Generate counterfactual explanation for a single instance."""

        start_time = datetime.now()

        # Get original prediction
        original_tensor = torch.FloatTensor(instance)
        original_pred, original_class = self.generator._get_prediction(
            original_tensor
        )

        # Generate counterfactuals
        counterfactuals = self.generator.generate(instance, feature_names)

        if not counterfactuals:
            logger.warning(
                f"No valid counterfactuals found for instance {instance_id}"
            )
            # Return empty explanation
            return self._create_empty_explanation(
                instance, original_pred, original_class, instance_id
            )

        # Analyze counterfactuals
        analysis_results = self._analyze_counterfactuals(
            instance,
            counterfactuals,
            feature_names,
            original_pred,
            original_class,
        )

        # Rank counterfactuals
        ranked_indices = self._rank_counterfactuals(
            instance, counterfactuals, analysis_results
        )

        generation_time = (datetime.now() - start_time).total_seconds()

        explanation = CounterfactualExplanation(
            instance_id=instance_id
            or f"cf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            original_instance=instance,
            original_prediction=original_pred,
            original_class=original_class,
            counterfactuals=counterfactuals,
            counterfactual_predictions=analysis_results["predictions"],
            counterfactual_classes=analysis_results["classes"],
            distances=analysis_results["distances"],
            sparsity_scores=analysis_results["sparsity_scores"],
            validity_scores=analysis_results["validity_scores"],
            feasibility_scores=analysis_results["feasibility_scores"],
            feature_changes=analysis_results["feature_changes"],
            changed_features=analysis_results["changed_features"],
            generation_method=self.config.method,
            generation_time=generation_time,
            convergence_info={"num_generated": len(counterfactuals)},
            ranked_indices=ranked_indices,
            best_counterfactual_idx=ranked_indices[0] if ranked_indices else 0,
        )

        # Log explanation
        audit_logger.log_model_operation(
            user_id="system",
            model_id="counterfactual_explainer",
            operation="counterfactual_explanation",
            success=True,
            details={
                "instance_id": explanation.instance_id,
                "method": self.config.method,
                "num_counterfactuals": len(counterfactuals),
                "generation_time": generation_time,
            },
        )

        # Save explanation if requested
        if self.config.save_explanations:
            self._save_explanation(explanation)

        return explanation

    def _analyze_counterfactuals(
        self,
        original: np.ndarray,
        counterfactuals: List[np.ndarray],
        feature_names: List[str],
        original_pred: float,
        original_class: int,
    ) -> Dict[str, List]:
        """Analyze counterfactuals and compute metrics."""

        results = {
            "predictions": [],
            "classes": [],
            "distances": [],
            "sparsity_scores": [],
            "validity_scores": [],
            "feasibility_scores": [],
            "feature_changes": [],
            "changed_features": [],
        }

        for cf in counterfactuals:
            # Get prediction
            cf_tensor = torch.FloatTensor(cf)
            cf_pred, cf_class = self.generator._get_prediction(cf_tensor)

            results["predictions"].append(cf_pred)
            results["classes"].append(cf_class)

            # Calculate distance
            distance = self.generator._calculate_distance(original, cf)
            results["distances"].append(distance)

            # Calculate sparsity
            sparsity = self.generator._calculate_sparsity(original, cf)
            results["sparsity_scores"].append(sparsity)

            # Validity score (how confident is the prediction flip)
            if original_class != cf_class:
                validity = abs(
                    cf_pred - 0.5
                )  # Distance from decision boundary
            else:
                validity = 0.0
            results["validity_scores"].append(validity)

            # Feasibility score (based on feature constraints)
            feasibility = self._calculate_feasibility(cf, feature_names)
            results["feasibility_scores"].append(feasibility)

            # Feature changes
            changes = {}
            changed_features = []
            for i, feature_name in enumerate(feature_names):
                if abs(original[i] - cf[i]) > 1e-6:
                    changes[feature_name] = (float(original[i]), float(cf[i]))
                    changed_features.append(feature_name)

            results["feature_changes"].append(changes)
            results["changed_features"].append(changed_features)

        return results

    def _calculate_feasibility(
        self, counterfactual: np.ndarray, feature_names: List[str]
    ) -> float:
        """Calculate feasibility score based on feature constraints."""
        feasibility_score = 1.0

        # Check feature ranges
        if self.config.feature_ranges:
            for i, feature_name in enumerate(feature_names):
                if feature_name in self.config.feature_ranges:
                    min_val, max_val = self.config.feature_ranges[feature_name]
                    if (
                        counterfactual[i] < min_val
                        or counterfactual[i] > max_val
                    ):
                        feasibility_score *= 0.5  # Penalty for out-of-range

        # Check immutable features
        # Note: This would require original instance comparison,
        # which should be done in the generator

        return feasibility_score

    def _rank_counterfactuals(
        self,
        original: np.ndarray,
        counterfactuals: List[np.ndarray],
        analysis_results: Dict[str, List],
    ) -> List[int]:
        """Rank counterfactuals by quality score."""

        scores = []
        for i in range(len(counterfactuals)):
            # Composite score combining multiple factors
            validity = analysis_results["validity_scores"][i]
            distance = analysis_results["distances"][i]
            sparsity = analysis_results["sparsity_scores"][i]
            feasibility = analysis_results["feasibility_scores"][i]

            # Normalize distance (lower is better)
            max_distance = (
                max(analysis_results["distances"])
                if analysis_results["distances"]
                else 1.0
            )
            normalized_distance = (
                1.0 - (distance / max_distance) if max_distance > 0 else 1.0
            )

            # Normalize sparsity (lower is better - fewer changes)
            normalized_sparsity = 1.0 - sparsity

            # Composite score (higher is better)
            score = (
                0.4 * validity
                + 0.3 * normalized_distance
                + 0.2 * normalized_sparsity
                + 0.1 * feasibility
            )

            scores.append(score)

        # Return indices sorted by score (descending)
        return sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

    def _create_empty_explanation(
        self,
        instance: np.ndarray,
        original_pred: float,
        original_class: int,
        instance_id: str,
    ) -> CounterfactualExplanation:
        """Create empty explanation when no counterfactuals found."""

        return CounterfactualExplanation(
            instance_id=instance_id or "empty_explanation",
            original_instance=instance,
            original_prediction=original_pred,
            original_class=original_class,
            counterfactuals=[],
            counterfactual_predictions=[],
            counterfactual_classes=[],
            distances=[],
            sparsity_scores=[],
            validity_scores=[],
            feasibility_scores=[],
            feature_changes=[],
            changed_features=[],
            generation_method=self.config.method,
            generation_time=0.0,
            convergence_info={
                "num_generated": 0,
                "error": "No valid counterfactuals found",
            },
            ranked_indices=[],
            best_counterfactual_idx=-1,
        )

    def _save_explanation(self, explanation: CounterfactualExplanation) -> str:
        """Save explanation to file."""
        try:
            # Create directory if it doesn't exist
            save_dir = Path(self.config.explanation_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save explanation
            save_path = (
                save_dir / f"{explanation.instance_id}_counterfactual.json"
            )

            with open(save_path, "w") as f:
                json.dump(explanation.to_dict(), f, indent=2)

            logger.debug(f"Counterfactual explanation saved to {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"Failed to save counterfactual explanation: {e}")
            return None

    def batch_explain(
        self,
        instances: np.ndarray,
        feature_names: List[str],
        instance_ids: Optional[List[str]] = None,
    ) -> List[CounterfactualExplanation]:
        """Generate counterfactual explanations for multiple instances."""

        explanations = []

        for i, instance in enumerate(instances):
            instance_id = instance_ids[i] if instance_ids else f"batch_cf_{i}"

            try:
                explanation = self.explain(
                    instance, feature_names, instance_id
                )
                explanations.append(explanation)
            except Exception as e:
                logger.error(
                    f"Failed to generate counterfactual for instance {instance_id}: {e}"
                )
                # Add empty explanation
                empty_explanation = self._create_empty_explanation(
                    instance, 0.0, 0, instance_id
                )
                explanations.append(empty_explanation)

        logger.info(
            f"Generated counterfactual explanations for {len(explanations)} instances"
        )
        return explanations

    def analyze_decision_boundary(
        self,
        instance: np.ndarray,
        feature_names: List[str],
        feature_idx: int,
        num_points: int = 50,
    ) -> Dict[str, Any]:
        """Analyze decision boundary by varying a specific feature."""

        # Get original prediction
        original_pred, original_class = self.generator._get_prediction(
            torch.FloatTensor(instance)
        )

        # Determine feature range
        feature_name = feature_names[feature_idx]
        if (
            self.config.feature_ranges
            and feature_name in self.config.feature_ranges
        ):
            min_val, max_val = self.config.feature_ranges[feature_name]
        else:
            # Use ±3 standard deviations from original value
            original_val = instance[feature_idx]
            std_dev = abs(original_val * 0.3) if original_val != 0 else 1.0
            min_val = original_val - 3 * std_dev
            max_val = original_val + 3 * std_dev

        # Generate points along feature dimension
        feature_values = np.linspace(min_val, max_val, num_points)
        predictions = []
        classes = []

        for val in feature_values:
            modified_instance = instance.copy()
            modified_instance[feature_idx] = val

            pred, pred_class = self.generator._get_prediction(
                torch.FloatTensor(modified_instance)
            )
            predictions.append(pred)
            classes.append(pred_class)

        # Find decision boundary crossings
        boundary_crossings = []
        for i in range(1, len(classes)):
            if classes[i] != classes[i - 1]:
                boundary_crossings.append(
                    {
                        "feature_value": feature_values[i],
                        "prediction": predictions[i],
                        "from_class": classes[i - 1],
                        "to_class": classes[i],
                    }
                )

        return {
            "feature_name": feature_name,
            "feature_values": feature_values.tolist(),
            "predictions": predictions,
            "classes": classes,
            "boundary_crossings": boundary_crossings,
            "original_value": float(instance[feature_idx]),
            "original_prediction": original_pred,
            "original_class": original_class,
        }


# Utility functions for easy integration


def generate_counterfactuals(
    model: nn.Module,
    instance: np.ndarray,
    feature_names: List[str],
    config: Optional[CounterfactualConfig] = None,
    instance_id: str = None,
) -> CounterfactualExplanation:
    """
    Generate counterfactual explanations for a single instance.

    Args:
        model: Neural network model
        instance: Input instance to explain
        feature_names: List of feature names
        config: Counterfactual configuration
        instance_id: Unique identifier for the instance

    Returns:
        CounterfactualExplanation object
    """
    explainer = CounterfactualExplainer(model, config)
    return explainer.explain(instance, feature_names, instance_id)


def analyze_what_if_scenarios(
    model: nn.Module,
    instance: np.ndarray,
    feature_names: List[str],
    scenarios: Dict[str, float],
    config: Optional[CounterfactualConfig] = None,
) -> Dict[str, Any]:
    """
    Analyze "what-if" scenarios by modifying specific features.

    Args:
        model: Neural network model
        instance: Original instance
        feature_names: List of feature names
        scenarios: Dictionary of feature_name -> new_value
        config: Counterfactual configuration

    Returns:
        Dictionary containing scenario analysis results
    """
    config = config or CounterfactualConfig()
    explainer = CounterfactualExplainer(model, config)

    # Get original prediction
    original_pred, original_class = explainer.generator._get_prediction(
        torch.FloatTensor(instance)
    )

    results = {
        "original_prediction": original_pred,
        "original_class": original_class,
        "scenarios": {},
    }

    for scenario_name, modifications in scenarios.items():
        modified_instance = instance.copy()

        # Apply modifications
        if isinstance(modifications, dict):
            for feature_name, new_value in modifications.items():
                if feature_name in feature_names:
                    idx = feature_names.index(feature_name)
                    modified_instance[idx] = new_value
        else:
            # Single feature modification
            if scenario_name in feature_names:
                idx = feature_names.index(scenario_name)
                modified_instance[idx] = modifications

        # Get prediction for modified instance
        new_pred, new_class = explainer.generator._get_prediction(
            torch.FloatTensor(modified_instance)
        )

        # Calculate changes
        feature_changes = {}
        for i, feature_name in enumerate(feature_names):
            if abs(instance[i] - modified_instance[i]) > 1e-6:
                feature_changes[feature_name] = {
                    "original": float(instance[i]),
                    "modified": float(modified_instance[i]),
                    "change": float(modified_instance[i] - instance[i]),
                }

        results["scenarios"][scenario_name] = {
            "prediction": new_pred,
            "class": new_class,
            "prediction_change": new_pred - original_pred,
            "class_flipped": new_class != original_class,
            "feature_changes": feature_changes,
            "distance": explainer.generator._calculate_distance(
                instance, modified_instance
            ),
        }

    return results


def explore_decision_boundary(
    model: nn.Module,
    instance: np.ndarray,
    feature_names: List[str],
    feature_name: str,
    config: Optional[CounterfactualConfig] = None,
) -> Dict[str, Any]:
    """
    Explore decision boundary by varying a specific feature.

    Args:
        model: Neural network model
        instance: Input instance
        feature_names: List of feature names
        feature_name: Name of feature to vary
        config: Counterfactual configuration

    Returns:
        Dictionary containing boundary analysis results
    """
    config = config or CounterfactualConfig()
    explainer = CounterfactualExplainer(model, config)

    if feature_name not in feature_names:
        raise ValueError(
            f"Feature '{feature_name}' not found in feature_names"
        )

    feature_idx = feature_names.index(feature_name)
    return explainer.analyze_decision_boundary(
        instance, feature_names, feature_idx
    )
