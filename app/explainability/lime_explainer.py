"""
LIME (Local Interpretable Model-agnostic Explanations) Integration for Model Explanations.

This module implements comprehensive LIME explanation generation for all model types,
local linear approximation generation, explanation simplification for customer-facing reports,
and LIME visualization and reporting for credit risk models.
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

# LIME imports
try:
    import lime
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")

# Plotting imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")

try:
    from ..core.interfaces import BaseModel
    from ..core.logging import get_audit_logger, get_logger
    from ..models.dnn_model import DNNModel
    from ..models.gnn_model import GNNModel
    from ..models.lstm_model import LSTMModel
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
class LIMEConfig:
    """Configuration for LIME explanations."""

    # Explainer settings
    mode: str = "tabular"  # "tabular", "text", "image"
    num_features: int = 10  # Number of features to include in explanation
    num_samples: int = 5000  # Number of samples for local approximation

    # Discretization settings
    discretize_continuous: bool = True
    discretizer: str = "quartile"  # "quartile", "decile", "entropy"

    # Sampling settings
    sample_around_instance: bool = True
    random_state: int = 42

    # Visualization settings
    save_plots: bool = True
    plot_format: str = "png"  # "png", "svg", "html"
    explanation_path: str = "explanations/lime"

    # Customer-facing settings
    simplify_explanations: bool = True
    customer_friendly_names: Optional[Dict[str, str]] = None
    explanation_template: str = "default"

    # Feature settings
    feature_names: Optional[List[str]] = None
    categorical_features: Optional[List[int]] = None
    categorical_names: Optional[Dict[int, List[str]]] = None


@dataclass
class LIMEExplanation:
    """Container for LIME explanation results."""

    # Basic information
    instance_id: str
    model_prediction: float
    local_prediction: float

    # LIME explanation data
    feature_importance: Dict[str, float]
    local_explanation: List[Tuple[str, float]]
    intercept: float
    score: float  # R² score of local approximation

    # Customer-facing explanation
    simplified_explanation: str
    top_positive_factors: List[
        Tuple[str, float, str]
    ]  # (feature, importance, description)
    top_negative_factors: List[Tuple[str, float, str]]

    # Metadata
    explanation_time: float
    model_type: str
    num_features_used: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to dictionary."""
        return {
            "instance_id": self.instance_id,
            "model_prediction": float(self.model_prediction),
            "local_prediction": float(self.local_prediction),
            "feature_importance": self.feature_importance,
            "local_explanation": self.local_explanation,
            "intercept": float(self.intercept),
            "score": float(self.score),
            "simplified_explanation": self.simplified_explanation,
            "top_positive_factors": self.top_positive_factors,
            "top_negative_factors": self.top_negative_factors,
            "explanation_time": self.explanation_time,
            "model_type": self.model_type,
            "num_features_used": self.num_features_used,
            "timestamp": self.timestamp.isoformat(),
        }


class ModelWrapper:
    """Wrapper to make different model types compatible with LIME."""

    def __init__(self, model: nn.Module, model_type: str = "pytorch"):
        self.model = model
        self.model_type = model_type
        self.model.eval()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for LIME explainer."""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        with torch.no_grad():
            if self.model_type == "lstm":
                # For LSTM, we might need to reshape input
                if len(X.shape) == 2:
                    X = X.unsqueeze(1)  # Add sequence dimension

            outputs = self.model(X)

            # Handle different output formats
            if hasattr(outputs, "squeeze"):
                outputs = outputs.squeeze()

            # Convert to probabilities for binary classification
            if (
                len(outputs.shape) == 0
                or len(outputs.shape) == 1
                or (len(outputs.shape) > 0 and outputs.shape[-1] == 1)
            ):
                probs = torch.sigmoid(outputs)
                # Return both classes for binary classification
                if len(probs.shape) == 0:
                    probs = probs.unsqueeze(0)
                return np.column_stack(
                    [1 - probs.cpu().numpy(), probs.cpu().numpy()]
                )
            else:
                probs = torch.softmax(outputs, dim=-1)
                return probs.cpu().numpy()


class LIMEExplainer:
    """Main LIME explainer class for credit risk models."""

    def __init__(self, model: nn.Module, config: Optional[LIMEConfig] = None):
        if not LIME_AVAILABLE:
            raise ImportError(
                "LIME is required but not installed. Install with: pip install lime"
            )

        self.model = model
        self.config = config or LIMEConfig()
        self.model_wrapper = ModelWrapper(model)
        self.explainer = None
        self.training_data = None

        logger.info(f"LIME explainer initialized for {self.config.mode} mode")

    def set_training_data(self, X_train: Union[pd.DataFrame, np.ndarray]):
        """Set training data for LIME explainer."""
        if isinstance(X_train, pd.DataFrame):
            self.training_data = X_train.values
            if self.config.feature_names is None:
                self.config.feature_names = X_train.columns.tolist()
        else:
            self.training_data = X_train

        # Initialize LIME explainer
        self._create_explainer()

        logger.info(f"Training data set: {self.training_data.shape}")

    def _create_explainer(self):
        """Create the LIME explainer."""
        if self.training_data is None:
            raise ValueError(
                "Training data must be set before creating explainer"
            )

        try:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                self.training_data,
                feature_names=self.config.feature_names,
                categorical_features=self.config.categorical_features,
                categorical_names=self.config.categorical_names,
                mode="classification",
                discretize_continuous=self.config.discretize_continuous,
                discretizer=self.config.discretizer,
                sample_around_instance=self.config.sample_around_instance,
                random_state=self.config.random_state,
            )

            logger.info("LIME tabular explainer created successfully")

        except Exception as e:
            logger.error(f"Failed to create LIME explainer: {e}")
            raise

    def explain_instance(
        self, X: Union[pd.DataFrame, np.ndarray], instance_id: str = None
    ) -> LIMEExplanation:
        """
        Explain a single instance using LIME.

        Args:
            X: Input instance to explain
            instance_id: Optional identifier for the instance

        Returns:
            LIMEExplanation object
        """
        if self.explainer is None:
            raise ValueError(
                "Explainer not initialized. Set training data first."
            )

        start_time = datetime.now()

        # Prepare input
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            if len(X_array.shape) == 1:
                X_array = X_array.reshape(1, -1)
        else:
            X_array = X
            if len(X_array.shape) == 1:
                X_array = X_array.reshape(1, -1)

        instance = X_array[0]  # LIME expects 1D array for single instance

        # Get model prediction
        model_prediction = self.model_wrapper.predict_proba(X_array)[
            0, 1
        ]  # Probability of positive class

        # Generate LIME explanation
        try:
            explanation = self.explainer.explain_instance(
                instance,
                self.model_wrapper.predict_proba,
                num_features=self.config.num_features,
                num_samples=self.config.num_samples,
            )

            # Extract explanation data
            local_explanation = explanation.as_list()
            feature_importance = dict(local_explanation)

            # Get local model prediction and score
            local_prediction = (
                explanation.local_pred[1]
                if hasattr(explanation, "local_pred")
                else model_prediction
            )
            intercept = (
                explanation.intercept[1]
                if hasattr(explanation, "intercept")
                else 0.0
            )
            score = explanation.score if hasattr(explanation, "score") else 0.0

        except Exception as e:
            logger.error(f"Failed to generate LIME explanation: {e}")
            # Create dummy explanation
            feature_names = self.config.feature_names or [
                f"feature_{i}" for i in range(len(instance))
            ]
            local_explanation = [
                (name, 0.0)
                for name in feature_names[: self.config.num_features]
            ]
            feature_importance = dict(local_explanation)
            local_prediction = model_prediction
            intercept = 0.0
            score = 0.0

        # Create customer-facing explanation
        simplified_explanation, top_positive, top_negative = (
            self._create_customer_explanation(
                local_explanation, model_prediction
            )
        )

        explanation_time = (datetime.now() - start_time).total_seconds()

        lime_explanation = LIMEExplanation(
            instance_id=instance_id
            or f"lime_instance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_prediction=float(model_prediction),
            local_prediction=float(local_prediction),
            feature_importance=feature_importance,
            local_explanation=local_explanation,
            intercept=float(intercept),
            score=float(score),
            simplified_explanation=simplified_explanation,
            top_positive_factors=top_positive,
            top_negative_factors=top_negative,
            explanation_time=explanation_time,
            model_type=type(self.model).__name__,
            num_features_used=len(local_explanation),
        )

        # Log explanation
        audit_logger.log_model_operation(
            user_id="system",
            model_id="lime_explainer",
            operation="instance_explanation",
            success=True,
            details={
                "instance_id": lime_explanation.instance_id,
                "explanation_time": explanation_time,
                "num_features": len(local_explanation),
                "local_score": score,
            },
        )

        return lime_explanation

    def _create_customer_explanation(
        self, local_explanation: List[Tuple[str, float]], prediction: float
    ) -> Tuple[
        str, List[Tuple[str, float, str]], List[Tuple[str, float, str]]
    ]:
        """Create simplified, customer-facing explanation."""

        # Sort by absolute importance
        sorted_explanation = sorted(
            local_explanation, key=lambda x: abs(x[1]), reverse=True
        )

        # Separate positive and negative factors
        positive_factors = [
            (name, value, self._get_factor_description(name, value, True))
            for name, value in sorted_explanation
            if value > 0
        ][:3]
        negative_factors = [
            (name, value, self._get_factor_description(name, value, False))
            for name, value in sorted_explanation
            if value < 0
        ][:3]

        # Create simplified explanation text
        risk_level = (
            "high"
            if prediction > 0.7
            else "medium" if prediction > 0.3 else "low"
        )

        explanation_parts = [
            f"This credit application has been assessed as {risk_level} risk (score: {prediction:.1%})."
        ]

        if positive_factors:
            explanation_parts.append("Factors that increase the risk:")
            for name, value, desc in positive_factors:
                explanation_parts.append(f"• {desc}")

        if negative_factors:
            explanation_parts.append("Factors that decrease the risk:")
            for name, value, desc in negative_factors:
                explanation_parts.append(f"• {desc}")

        simplified_explanation = "\n".join(explanation_parts)

        return simplified_explanation, positive_factors, negative_factors

    def _get_factor_description(
        self, feature_name: str, importance: float, increases_risk: bool
    ) -> str:
        """Generate customer-friendly description for a factor."""

        # Use custom names if provided
        if (
            self.config.customer_friendly_names
            and feature_name in self.config.customer_friendly_names
        ):
            friendly_name = self.config.customer_friendly_names[feature_name]
        else:
            friendly_name = self._make_friendly_name(feature_name)

        # Create description based on importance and direction
        impact = (
            "significantly"
            if abs(importance) > 0.1
            else "moderately" if abs(importance) > 0.05 else "slightly"
        )
        direction = "increases" if increases_risk else "decreases"

        return f"{friendly_name} {impact} {direction} the risk"

    def _make_friendly_name(self, feature_name: str) -> str:
        """Convert technical feature names to customer-friendly names."""

        # Common mappings for credit risk features
        name_mappings = {
            "income": "Your income level",
            "debt_to_income": "Your debt-to-income ratio",
            "credit_score": "Your credit score",
            "employment_length": "Your employment history",
            "loan_amount": "The loan amount requested",
            "home_ownership": "Your home ownership status",
            "annual_inc": "Your annual income",
            "delinq_2yrs": "Recent payment delinquencies",
            "open_acc": "Number of open credit accounts",
            "pub_rec": "Public records on your credit report",
            "revol_bal": "Your revolving credit balance",
            "revol_util": "Your credit utilization rate",
            "total_acc": "Total number of credit accounts",
        }

        # Try exact match first
        if feature_name.lower() in name_mappings:
            return name_mappings[feature_name.lower()]

        # Try partial matches
        for key, value in name_mappings.items():
            if key in feature_name.lower():
                return value

        # Default: clean up the feature name
        cleaned = feature_name.replace("_", " ").replace("-", " ").title()
        return f"Your {cleaned.lower()}"

    def explain_batch(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        instance_ids: Optional[List[str]] = None,
    ) -> List[LIMEExplanation]:
        """
        Explain multiple instances using LIME.

        Args:
            X: Input instances to explain
            instance_ids: Optional identifiers for instances

        Returns:
            List of LIMEExplanation objects
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        if instance_ids is None:
            instance_ids = [f"lime_instance_{i}" for i in range(len(X_array))]

        explanations = []

        for i, (x, instance_id) in enumerate(zip(X_array, instance_ids)):
            try:
                explanation = self.explain_instance(
                    x.reshape(1, -1), instance_id
                )
                explanations.append(explanation)

                if (i + 1) % 10 == 0:
                    logger.debug(
                        f"Processed {i + 1}/{len(X_array)} LIME explanations"
                    )

            except Exception as e:
                logger.error(f"Failed to explain instance {instance_id}: {e}")
                continue

        logger.info(
            f"Generated LIME explanations for {len(explanations)} instances"
        )
        return explanations

    def create_visualization(
        self, explanation: LIMEExplanation, plot_type: str = "bar"
    ) -> Optional[str]:
        """
        Create LIME visualization for an explanation.

        Args:
            explanation: LIMEExplanation object
            plot_type: Type of plot to create ("bar", "horizontal_bar", "waterfall")

        Returns:
            Path to saved plot or None if plotting failed
        """
        try:
            plt.style.use("default")

            if plot_type == "bar":
                return self._create_bar_plot(explanation)
            elif plot_type == "horizontal_bar":
                return self._create_horizontal_bar_plot(explanation)
            elif plot_type == "waterfall":
                return self._create_waterfall_plot(explanation)
            else:
                return self._create_summary_plot(explanation)

        except Exception as e:
            logger.error(f"Failed to create {plot_type} plot: {e}")
            return None

    def _create_bar_plot(self, explanation: LIMEExplanation) -> str:
        """Create vertical bar plot for LIME explanation."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Get feature names and importance values
            features = [item[0] for item in explanation.local_explanation]
            importances = [item[1] for item in explanation.local_explanation]

            # Create colors based on positive/negative impact
            colors = ["green" if imp > 0 else "red" for imp in importances]

            # Create bar plot
            bars = ax.bar(
                range(len(features)), importances, color=colors, alpha=0.7
            )

            # Customize plot
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha="right")
            ax.set_ylabel("LIME Importance")
            ax.set_title(
                f"LIME Feature Importance - {explanation.instance_id}"
            )
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, imp in zip(bars, importances):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (0.01 if height > 0 else -0.01),
                    f"{imp:.3f}",
                    ha="center",
                    va="bottom" if height > 0 else "top",
                )

            # Add model prediction info
            ax.text(
                0.02,
                0.98,
                f"Model Prediction: {explanation.model_prediction:.3f}\n"
                f"Local R²: {explanation.score:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            plt.tight_layout()

            # Save plot
            if self.config.save_plots:
                plot_path = self._save_plot(
                    fig, f"lime_bar_{explanation.instance_id}"
                )
                plt.close(fig)
                return plot_path

            plt.show()
            return None

        except Exception as e:
            logger.error(f"Failed to create bar plot: {e}")
            return None

    def _create_horizontal_bar_plot(self, explanation: LIMEExplanation) -> str:
        """Create horizontal bar plot for LIME explanation."""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Get feature names and importance values
            features = [item[0] for item in explanation.local_explanation]
            importances = [item[1] for item in explanation.local_explanation]

            # Sort by absolute importance for better visualization
            sorted_items = sorted(
                zip(features, importances),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            features = [item[0] for item in sorted_items]
            importances = [item[1] for item in sorted_items]

            # Create colors
            colors = ["green" if imp > 0 else "red" for imp in importances]

            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importances, color=colors, alpha=0.7)

            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel("LIME Importance")
            ax.set_title(
                f"LIME Feature Importance - {explanation.instance_id}"
            )
            ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
            ax.grid(True, alpha=0.3)

            # Add value labels
            for i, (bar, imp) in enumerate(zip(bars, importances)):
                width = bar.get_width()
                ax.text(
                    width + (0.01 if width > 0 else -0.01),
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{imp:.3f}",
                    ha="left" if width > 0 else "right",
                    va="center",
                )

            plt.tight_layout()

            # Save plot
            if self.config.save_plots:
                plot_path = self._save_plot(
                    fig, f"lime_hbar_{explanation.instance_id}"
                )
                plt.close(fig)
                return plot_path

            plt.show()
            return None

        except Exception as e:
            logger.error(f"Failed to create horizontal bar plot: {e}")
            return None

    def _create_waterfall_plot(self, explanation: LIMEExplanation) -> str:
        """Create waterfall plot for LIME explanation."""
        try:
            fig, ax = plt.subplots(figsize=(14, 8))

            # Get sorted features by importance
            sorted_items = sorted(
                explanation.local_explanation,
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            features = [item[0] for item in sorted_items]
            importances = [item[1] for item in sorted_items]

            # Calculate cumulative values for waterfall
            cumulative = explanation.intercept
            positions = [cumulative]

            for imp in importances:
                cumulative += imp
                positions.append(cumulative)

            # Create waterfall plot
            for i, (feature, imp) in enumerate(zip(features, importances)):
                color = "green" if imp > 0 else "red"
                bottom = positions[i] if imp > 0 else positions[i] + imp

                ax.bar(
                    i,
                    abs(imp),
                    bottom=bottom,
                    color=color,
                    alpha=0.7,
                    width=0.6,
                )

                # Add value labels
                label_pos = positions[i] + imp / 2
                ax.text(
                    i,
                    label_pos,
                    f"{imp:.3f}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white",
                )

            # Add baseline and final prediction
            ax.axhline(
                y=explanation.intercept,
                color="blue",
                linestyle="--",
                label=f"Baseline: {explanation.intercept:.3f}",
            )
            ax.axhline(
                y=explanation.local_prediction,
                color="orange",
                linestyle="-",
                label=f"Local Prediction: {explanation.local_prediction:.3f}",
            )

            # Customize plot
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha="right")
            ax.set_ylabel("LIME Contribution")
            ax.set_title(f"LIME Waterfall Plot - {explanation.instance_id}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            if self.config.save_plots:
                plot_path = self._save_plot(
                    fig, f"lime_waterfall_{explanation.instance_id}"
                )
                plt.close(fig)
                return plot_path

            plt.show()
            return None

        except Exception as e:
            logger.error(f"Failed to create waterfall plot: {e}")
            return None

    def _create_summary_plot(self, explanation: LIMEExplanation) -> str:
        """Create summary plot with multiple views."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                2, 2, figsize=(16, 12)
            )

            # Get data
            features = [item[0] for item in explanation.local_explanation]
            importances = [item[1] for item in explanation.local_explanation]
            colors = ["green" if imp > 0 else "red" for imp in importances]

            # Plot 1: Horizontal bar chart
            y_pos = np.arange(len(features))
            ax1.barh(y_pos, importances, color=colors, alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(features)
            ax1.set_xlabel("LIME Importance")
            ax1.set_title("Feature Importance")
            ax1.axvline(x=0, color="black", linestyle="-", alpha=0.3)
            ax1.grid(True, alpha=0.3)

            # Plot 2: Absolute importance
            abs_importances = [abs(imp) for imp in importances]
            ax2.bar(
                range(len(features)),
                abs_importances,
                color="skyblue",
                alpha=0.7,
            )
            ax2.set_xticks(range(len(features)))
            ax2.set_xticklabels(features, rotation=45, ha="right")
            ax2.set_ylabel("|LIME Importance|")
            ax2.set_title("Absolute Feature Importance")
            ax2.grid(True, alpha=0.3)

            # Plot 3: Prediction breakdown
            labels = ["Negative Impact", "Positive Impact"]
            negative_sum = sum(imp for imp in importances if imp < 0)
            positive_sum = sum(imp for imp in importances if imp > 0)
            values = [abs(negative_sum), positive_sum]
            colors_pie = ["red", "green"]

            ax3.pie(
                values,
                labels=labels,
                colors=colors_pie,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax3.set_title("Impact Distribution")

            # Plot 4: Model vs Local prediction comparison
            categories = ["Model\nPrediction", "Local\nPrediction"]
            predictions = [
                explanation.model_prediction,
                explanation.local_prediction,
            ]
            bars = ax4.bar(
                categories, predictions, color=["blue", "orange"], alpha=0.7
            )
            ax4.set_ylabel("Prediction Value")
            ax4.set_title("Model vs Local Prediction")
            ax4.set_ylim(0, 1)

            # Add value labels
            for bar, pred in zip(bars, predictions):
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{pred:.3f}",
                    ha="center",
                    va="bottom",
                )

            # Add R² score
            ax4.text(
                0.5,
                0.9,
                f"Local R²: {explanation.score:.3f}",
                transform=ax4.transAxes,
                ha="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            plt.suptitle(
                f"LIME Explanation Summary - {explanation.instance_id}"
            )
            plt.tight_layout()

            # Save plot
            if self.config.save_plots:
                plot_path = self._save_plot(
                    fig, f"lime_summary_{explanation.instance_id}"
                )
                plt.close(fig)
                return plot_path

            plt.show()
            return None

        except Exception as e:
            logger.error(f"Failed to create summary plot: {e}")
            return None

    def _save_plot(self, fig, filename: str) -> str:
        """Save plot to file."""
        try:
            # Create directory if it doesn't exist
            plot_dir = Path(self.config.explanation_path) / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Save plot
            plot_path = plot_dir / f"{filename}.{self.config.plot_format}"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            logger.debug(f"Plot saved to {plot_path}")
            return str(plot_path)

        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
            return None

    def save_explanations(
        self, explanations: List[LIMEExplanation], filename: str = None
    ) -> str:
        """Save explanations to file."""
        try:
            # Create directory if it doesn't exist
            save_dir = Path(self.config.explanation_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"lime_explanations_{timestamp}.json"

            save_path = save_dir / filename

            # Convert explanations to dictionaries
            explanations_data = [exp.to_dict() for exp in explanations]

            # Save to JSON
            with open(save_path, "w") as f:
                json.dump(explanations_data, f, indent=2)

            logger.info(
                f"Saved {len(explanations)} explanations to {save_path}"
            )
            return str(save_path)

        except Exception as e:
            logger.error(f"Failed to save explanations: {e}")
            return None

    def load_explanations(self, filepath: str) -> List[LIMEExplanation]:
        """Load explanations from file."""
        try:
            with open(filepath, "r") as f:
                explanations_data = json.load(f)

            explanations = []
            for data in explanations_data:
                # Convert timestamp back to datetime
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])

                # Create LIMEExplanation object
                explanation = LIMEExplanation(**data)
                explanations.append(explanation)

            logger.info(
                f"Loaded {len(explanations)} explanations from {filepath}"
            )
            return explanations

        except Exception as e:
            logger.error(f"Failed to load explanations: {e}")
            return []


# Utility functions for easy integration


def create_lime_explainer(
    model: nn.Module,
    training_data: Union[pd.DataFrame, np.ndarray],
    config: Optional[LIMEConfig] = None,
) -> LIMEExplainer:
    """
    Create and initialize a LIME explainer.

    Args:
        model: PyTorch model to explain
        training_data: Training data for LIME explainer
        config: Optional LIME configuration

    Returns:
        Initialized LIMEExplainer
    """
    explainer = LIMEExplainer(model, config)
    explainer.set_training_data(training_data)
    return explainer


def explain_credit_decision(
    model: nn.Module,
    instance: Union[pd.DataFrame, np.ndarray],
    training_data: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str] = None,
) -> LIMEExplanation:
    """
    Generate LIME explanation for a single credit decision.

    Args:
        model: PyTorch model
        instance: Single instance to explain
        training_data: Training data for LIME
        feature_names: Optional feature names

    Returns:
        LIMEExplanation object
    """
    config = LIMEConfig(feature_names=feature_names, save_plots=False)
    explainer = create_lime_explainer(model, training_data, config)
    return explainer.explain_instance(instance, "credit_decision")


def batch_explain_decisions(
    model: nn.Module,
    instances: Union[pd.DataFrame, np.ndarray],
    training_data: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str] = None,
) -> List[LIMEExplanation]:
    """
    Generate LIME explanations for multiple credit decisions.

    Args:
        model: PyTorch model
        instances: Multiple instances to explain
        training_data: Training data for LIME
        feature_names: Optional feature names

    Returns:
        List of LIMEExplanation objects
    """
    config = LIMEConfig(feature_names=feature_names, save_plots=False)
    explainer = create_lime_explainer(model, training_data, config)
    return explainer.explain_batch(instances)
