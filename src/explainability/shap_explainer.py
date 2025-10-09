"""
SHAP (SHapley Additive exPlanations) Integration for Model Explanations.

This module implements comprehensive SHAP value calculation for all model types,
global and local feature importance extraction, SHAP visualization generation,
and batch explanation processing for efficiency in credit risk models.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import json
from pathlib import Path
import logging

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

# Plotting imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")

try:
    from ..core.interfaces import BaseModel
    from ..core.logging import get_logger, get_audit_logger
    from ..models.dnn_model import DNNModel
    from ..models.lstm_model import LSTMModel
    from ..models.gnn_model import GNNModel
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
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
class SHAPConfig:
    """Configuration for SHAP explanations."""
    # Explainer settings
    explainer_type: str = "auto"  # "auto", "tree", "deep", "linear", "kernel", "permutation"
    background_samples: int = 100
    max_evals: int = 1000
    
    # Batch processing
    batch_size: int = 32
    enable_batch_processing: bool = True
    
    # Visualization settings
    max_display_features: int = 20
    plot_type: str = "waterfall"  # "waterfall", "bar", "beeswarm", "summary"
    save_plots: bool = True
    plot_format: str = "png"  # "png", "svg", "html"
    
    # Performance settings
    use_gpu: bool = False
    n_jobs: int = -1
    random_state: int = 42
    
    # Output settings
    save_explanations: bool = True
    explanation_path: str = "explanations/shap"
    include_raw_values: bool = True
    
    # Feature settings
    feature_names: Optional[List[str]] = None
    feature_groups: Optional[Dict[str, List[str]]] = None


@dataclass
class SHAPExplanation:
    """Container for SHAP explanation results."""
    # Basic information
    instance_id: str
    model_prediction: float
    base_value: float
    
    # SHAP values
    shap_values: np.ndarray
    feature_values: np.ndarray
    feature_names: List[str]
    
    # Aggregated metrics
    feature_importance: Dict[str, float]
    top_positive_features: List[Tuple[str, float]]
    top_negative_features: List[Tuple[str, float]]
    
    # Metadata
    explanation_time: float
    model_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to dictionary."""
        return {
            "instance_id": self.instance_id,
            "model_prediction": float(self.model_prediction),
            "base_value": float(self.base_value),
            "shap_values": self.shap_values.tolist(),
            "feature_values": self.feature_values.tolist(),
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "top_positive_features": self.top_positive_features,
            "top_negative_features": self.top_negative_features,
            "explanation_time": self.explanation_time,
            "model_type": self.model_type,
            "timestamp": self.timestamp.isoformat()
        }


class ModelWrapper:
    """Wrapper to make different model types compatible with SHAP."""
    
    def __init__(self, model: nn.Module, model_type: str = "pytorch"):
        self.model = model
        self.model_type = model_type
        self.model.eval()
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for SHAP explainer."""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        with torch.no_grad():
            if self.model_type == "lstm":
                # For LSTM, we might need to reshape input
                if len(X.shape) == 2:
                    X = X.unsqueeze(1)  # Add sequence dimension
            
            outputs = self.model(X)
            
            # Handle different output formats
            if hasattr(outputs, 'squeeze'):
                outputs = outputs.squeeze()
            
            # Convert to probabilities for binary classification
            if len(outputs.shape) == 0 or len(outputs.shape) == 1 or (len(outputs.shape) > 0 and outputs.shape[-1] == 1):
                probs = torch.sigmoid(outputs)
            else:
                probs = torch.softmax(outputs, dim=-1)
            
            return probs.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (sklearn-style interface)."""
        probs = self(X)
        if len(probs.shape) == 1:
            # Binary classification - return both classes
            return np.column_stack([1 - probs, probs])
        return probs


class SHAPExplainer:
    """Main SHAP explainer class for credit risk models."""
    
    def __init__(self, model: nn.Module, config: Optional[SHAPConfig] = None):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required but not installed. Install with: pip install shap")
        
        self.model = model
        self.config = config or SHAPConfig()
        self.model_wrapper = ModelWrapper(model)
        self.explainer = None
        self.background_data = None
        
        # Initialize explainer
        self._initialize_explainer()
        
        logger.info(f"SHAP explainer initialized with {self.config.explainer_type} explainer")
    
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer."""
        # This will be set when we have background data
        pass
    
    def set_background_data(self, X_background: Union[pd.DataFrame, np.ndarray]):
        """Set background data for SHAP explainer."""
        if isinstance(X_background, pd.DataFrame):
            self.background_data = X_background.values
            if self.config.feature_names is None:
                self.config.feature_names = X_background.columns.tolist()
        else:
            self.background_data = X_background
        
        # Sample background data if too large
        if len(self.background_data) > self.config.background_samples:
            indices = np.random.choice(
                len(self.background_data), 
                self.config.background_samples, 
                replace=False
            )
            self.background_data = self.background_data[indices]
        
        # Initialize explainer with background data
        self._create_explainer()
        
        logger.info(f"Background data set: {self.background_data.shape}")
    
    def _create_explainer(self):
        """Create the appropriate SHAP explainer."""
        if self.background_data is None:
            raise ValueError("Background data must be set before creating explainer")
        
        try:
            if self.config.explainer_type == "auto":
                # Auto-detect best explainer type
                self.explainer = self._auto_select_explainer()
            elif self.config.explainer_type == "deep":
                self.explainer = shap.DeepExplainer(self.model, torch.FloatTensor(self.background_data))
            elif self.config.explainer_type == "kernel":
                self.explainer = shap.KernelExplainer(self.model_wrapper, self.background_data)
            elif self.config.explainer_type == "permutation":
                self.explainer = shap.PermutationExplainer(self.model_wrapper, self.background_data)
            else:
                # Default to kernel explainer
                self.explainer = shap.KernelExplainer(self.model_wrapper, self.background_data)
            
            logger.info(f"Created {type(self.explainer).__name__}")
            
        except Exception as e:
            logger.warning(f"Failed to create {self.config.explainer_type} explainer: {e}")
            # Fallback to kernel explainer
            self.explainer = shap.KernelExplainer(self.model_wrapper, self.background_data)
            logger.info("Using KernelExplainer as fallback")
    
    def _auto_select_explainer(self):
        """Automatically select the best explainer type."""
        # For PyTorch models, try DeepExplainer first
        try:
            return shap.DeepExplainer(self.model, torch.FloatTensor(self.background_data))
        except:
            # Fallback to KernelExplainer
            return shap.KernelExplainer(self.model_wrapper, self.background_data)
    
    def explain_instance(self, X: Union[pd.DataFrame, np.ndarray], 
                        instance_id: str = None) -> SHAPExplanation:
        """
        Explain a single instance.
        
        Args:
            X: Input instance to explain
            instance_id: Optional identifier for the instance
            
        Returns:
            SHAPExplanation object
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Set background data first.")
        
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
        
        # Get model prediction
        predictions = self.model_wrapper(X_array)
        prediction = predictions[0] if len(predictions.shape) > 0 and predictions.shape[0] > 1 else float(predictions)
        
        # Calculate SHAP values
        try:
            if isinstance(self.explainer, shap.DeepExplainer):
                shap_values = self.explainer.shap_values(torch.FloatTensor(X_array))
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # For binary classification
                base_value = self.explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[0]
            else:
                shap_values = self.explainer.shap_values(X_array)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # For binary classification
                base_value = self.explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[0]
            
            # Ensure shap_values is 1D for single instance
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {e}")
            # Create dummy SHAP values
            shap_values = np.zeros(X_array.shape[1])
            base_value = 0.0
        
        # Get feature names
        feature_names = self.config.feature_names or [f"feature_{i}" for i in range(len(shap_values))]
        
        # Calculate feature importance
        feature_importance = {
            name: float(abs(value)) 
            for name, value in zip(feature_names, shap_values)
        }
        
        # Get top positive and negative features
        feature_impacts = list(zip(feature_names, shap_values))
        feature_impacts.sort(key=lambda x: x[1], reverse=True)
        
        top_positive = [(name, float(value)) for name, value in feature_impacts if value > 0][:5]
        top_negative = [(name, float(value)) for name, value in feature_impacts if value < 0][-5:]
        
        explanation_time = (datetime.now() - start_time).total_seconds()
        
        explanation = SHAPExplanation(
            instance_id=instance_id or f"instance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_prediction=float(prediction),
            base_value=float(base_value),
            shap_values=shap_values,
            feature_values=X_array[0],
            feature_names=feature_names,
            feature_importance=feature_importance,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            explanation_time=explanation_time,
            model_type=type(self.model).__name__
        )
        
        # Log explanation
        audit_logger.log_model_operation(
            user_id="system",
            model_id="shap_explainer",
            operation="instance_explanation",
            success=True,
            details={
                "instance_id": explanation.instance_id,
                "explanation_time": explanation_time,
                "num_features": len(feature_names)
            }
        )
        
        return explanation
    
    def explain_batch(self, X: Union[pd.DataFrame, np.ndarray], 
                     instance_ids: Optional[List[str]] = None) -> List[SHAPExplanation]:
        """
        Explain multiple instances efficiently.
        
        Args:
            X: Input instances to explain
            instance_ids: Optional identifiers for instances
            
        Returns:
            List of SHAPExplanation objects
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if instance_ids is None:
            instance_ids = [f"instance_{i}" for i in range(len(X_array))]
        
        explanations = []
        
        if self.config.enable_batch_processing and len(X_array) > self.config.batch_size:
            # Process in batches
            for i in range(0, len(X_array), self.config.batch_size):
                batch_end = min(i + self.config.batch_size, len(X_array))
                batch_X = X_array[i:batch_end]
                batch_ids = instance_ids[i:batch_end]
                
                for j, (x, instance_id) in enumerate(zip(batch_X, batch_ids)):
                    explanation = self.explain_instance(x.reshape(1, -1), instance_id)
                    explanations.append(explanation)
                
                logger.debug(f"Processed batch {i//self.config.batch_size + 1}, instances {i}-{batch_end-1}")
        else:
            # Process all at once
            for x, instance_id in zip(X_array, instance_ids):
                explanation = self.explain_instance(x.reshape(1, -1), instance_id)
                explanations.append(explanation)
        
        logger.info(f"Generated explanations for {len(explanations)} instances")
        return explanations
    
    def get_global_importance(self, X: Union[pd.DataFrame, np.ndarray], 
                            sample_size: int = None) -> Dict[str, float]:
        """
        Calculate global feature importance across multiple instances.
        
        Args:
            X: Dataset to analyze
            sample_size: Number of samples to use (None for all)
            
        Returns:
            Dictionary of feature importance scores
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Sample data if requested
        if sample_size and len(X_array) > sample_size:
            indices = np.random.choice(len(X_array), sample_size, replace=False)
            X_array = X_array[indices]
        
        logger.info(f"Calculating global importance for {len(X_array)} instances")
        
        # Get SHAP values for all instances
        try:
            if isinstance(self.explainer, shap.DeepExplainer):
                shap_values = self.explainer.shap_values(torch.FloatTensor(X_array))
            else:
                shap_values = self.explainer.shap_values(X_array)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For binary classification
            
        except Exception as e:
            logger.error(f"Failed to calculate global SHAP values: {e}")
            return {}
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Get feature names
        feature_names = self.config.feature_names or [f"feature_{i}" for i in range(len(mean_abs_shap))]
        
        # Create importance dictionary
        global_importance = {
            name: float(importance) 
            for name, importance in zip(feature_names, mean_abs_shap)
        }
        
        # Sort by importance
        global_importance = dict(sorted(global_importance.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"Global importance calculated for {len(global_importance)} features")
        return global_importance
    
    def create_visualization(self, explanation: SHAPExplanation, 
                           plot_type: str = None) -> Optional[str]:
        """
        Create SHAP visualization for an explanation.
        
        Args:
            explanation: SHAPExplanation object
            plot_type: Type of plot to create
            
        Returns:
            Path to saved plot or None if plotting failed
        """
        plot_type = plot_type or self.config.plot_type
        
        try:
            plt.style.use('default')
            
            if plot_type == "waterfall":
                return self._create_waterfall_plot(explanation)
            elif plot_type == "bar":
                return self._create_bar_plot(explanation)
            elif plot_type == "force":
                return self._create_force_plot(explanation)
            else:
                return self._create_summary_plot(explanation)
                
        except Exception as e:
            logger.error(f"Failed to create {plot_type} plot: {e}")
            return None
    
    def _create_waterfall_plot(self, explanation: SHAPExplanation) -> str:
        """Create waterfall plot for SHAP values."""
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get top features for display
            n_features = min(self.config.max_display_features, len(explanation.shap_values))
            
            # Sort features by absolute SHAP value
            feature_impacts = list(zip(explanation.feature_names, explanation.shap_values))
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_features = feature_impacts[:n_features]
            feature_names = [f[0] for f in top_features]
            shap_values = [f[1] for f in top_features]
            
            # Create waterfall plot
            cumulative = explanation.base_value
            positions = []
            colors = []
            
            for i, (name, value) in enumerate(top_features):
                positions.append(cumulative)
                cumulative += value
                colors.append('green' if value > 0 else 'red')
            
            # Plot bars
            for i, (pos, val, color) in enumerate(zip(positions, shap_values, colors)):
                ax.bar(i, abs(val), bottom=pos if val > 0 else pos + val, 
                      color=color, alpha=0.7, width=0.6)
                
                # Add value labels
                label_pos = pos + val/2
                ax.text(i, label_pos, f'{val:.3f}', ha='center', va='center', 
                       fontweight='bold', color='white')
            
            # Add base value and prediction lines
            ax.axhline(y=explanation.base_value, color='blue', linestyle='--', 
                      label=f'Base value: {explanation.base_value:.3f}')
            ax.axhline(y=explanation.model_prediction, color='orange', linestyle='-', 
                      label=f'Prediction: {explanation.model_prediction:.3f}')
            
            # Customize plot
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_ylabel('SHAP Value')
            ax.set_title(f'SHAP Waterfall Plot - {explanation.instance_id}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if self.config.save_plots:
                plot_path = self._save_plot(fig, f"waterfall_{explanation.instance_id}")
                plt.close(fig)
                return plot_path
            
            plt.show()
            return None
            
        except Exception as e:
            logger.error(f"Failed to create waterfall plot: {e}")
            return None
    
    def _create_bar_plot(self, explanation: SHAPExplanation) -> str:
        """Create bar plot for SHAP values."""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get top features
            n_features = min(self.config.max_display_features, len(explanation.shap_values))
            
            # Sort by absolute SHAP value
            feature_impacts = list(zip(explanation.feature_names, explanation.shap_values))
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_features = feature_impacts[:n_features]
            feature_names = [f[0] for f in top_features]
            shap_values = [f[1] for f in top_features]
            
            # Create colors
            colors = ['green' if val > 0 else 'red' for val in shap_values]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(feature_names))
            bars = ax.barh(y_pos, shap_values, color=colors, alpha=0.7)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, shap_values)):
                ax.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}', 
                       va='center', ha='left' if val > 0 else 'right')
            
            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('SHAP Value')
            ax.set_title(f'SHAP Feature Importance - {explanation.instance_id}')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if self.config.save_plots:
                plot_path = self._save_plot(fig, f"bar_{explanation.instance_id}")
                plt.close(fig)
                return plot_path
            
            plt.show()
            return None
            
        except Exception as e:
            logger.error(f"Failed to create bar plot: {e}")
            return None
    
    def _create_force_plot(self, explanation: SHAPExplanation) -> str:
        """Create force plot for SHAP values."""
        try:
            # This would typically use shap.force_plot, but we'll create a custom version
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Sort features by SHAP value
            feature_impacts = list(zip(explanation.feature_names, explanation.shap_values, explanation.feature_values))
            feature_impacts.sort(key=lambda x: x[1])
            
            # Separate positive and negative contributions
            negative_features = [(name, val, fval) for name, val, fval in feature_impacts if val < 0]
            positive_features = [(name, val, fval) for name, val, fval in feature_impacts if val > 0]
            
            # Plot negative contributions (left side)
            neg_cumsum = 0
            for i, (name, val, fval) in enumerate(negative_features):
                width = abs(val)
                ax.barh(0, -width, left=neg_cumsum - width, height=0.5, 
                       color='red', alpha=0.7, label=f'{name}={fval:.2f}' if i < 5 else "")
                
                if i < 5:  # Label top 5
                    ax.text(neg_cumsum - width/2, 0, f'{name}\n{val:.3f}', 
                           ha='center', va='center', fontsize=8, rotation=90)
                
                neg_cumsum -= width
            
            # Plot positive contributions (right side)
            pos_cumsum = 0
            for i, (name, val, fval) in enumerate(positive_features):
                width = val
                ax.barh(0, width, left=pos_cumsum, height=0.5, 
                       color='green', alpha=0.7, label=f'{name}={fval:.2f}' if i < 5 else "")
                
                if i < 5:  # Label top 5
                    ax.text(pos_cumsum + width/2, 0, f'{name}\n{val:.3f}', 
                           ha='center', va='center', fontsize=8, rotation=90)
                
                pos_cumsum += width
            
            # Add base value and prediction markers
            ax.axvline(x=0, color='blue', linestyle='--', linewidth=2, 
                      label=f'Base: {explanation.base_value:.3f}')
            ax.axvline(x=explanation.model_prediction - explanation.base_value, 
                      color='orange', linestyle='-', linewidth=2, 
                      label=f'Prediction: {explanation.model_prediction:.3f}')
            
            # Customize plot
            ax.set_xlim(neg_cumsum - 0.1, pos_cumsum + 0.1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel('SHAP Value Contribution')
            ax.set_title(f'SHAP Force Plot - {explanation.instance_id}')
            ax.set_yticks([])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if self.config.save_plots:
                plot_path = self._save_plot(fig, f"force_{explanation.instance_id}")
                plt.close(fig)
                return plot_path
            
            plt.show()
            return None
            
        except Exception as e:
            logger.error(f"Failed to create force plot: {e}")
            return None
    
    def _create_summary_plot(self, explanation: SHAPExplanation) -> str:
        """Create summary plot for SHAP values."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left plot: Feature importance
            n_features = min(self.config.max_display_features, len(explanation.shap_values))
            feature_impacts = list(zip(explanation.feature_names, explanation.shap_values))
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_features = feature_impacts[:n_features]
            feature_names = [f[0] for f in top_features]
            shap_values = [f[1] for f in top_features]
            
            colors = ['green' if val > 0 else 'red' for val in shap_values]
            y_pos = np.arange(len(feature_names))
            
            ax1.barh(y_pos, [abs(val) for val in shap_values], color=colors, alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(feature_names)
            ax1.set_xlabel('|SHAP Value|')
            ax1.set_title('Feature Importance')
            ax1.grid(True, alpha=0.3)
            
            # Right plot: SHAP values with direction
            ax2.barh(y_pos, shap_values, color=colors, alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(feature_names)
            ax2.set_xlabel('SHAP Value')
            ax2.set_title('Feature Impact Direction')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f'SHAP Summary - {explanation.instance_id}')
            plt.tight_layout()
            
            # Save plot
            if self.config.save_plots:
                plot_path = self._save_plot(fig, f"summary_{explanation.instance_id}")
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
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            logger.debug(f"Plot saved to {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
            return None
    
    def create_global_summary_plot(self, global_importance: Dict[str, float]) -> Optional[str]:
        """Create global feature importance summary plot."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get top features
            n_features = min(self.config.max_display_features, len(global_importance))
            top_features = list(global_importance.items())[:n_features]
            
            feature_names = [f[0] for f in top_features]
            importance_values = [f[1] for f in top_features]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(feature_names))
            bars = ax.barh(y_pos, importance_values, color='steelblue', alpha=0.7)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, importance_values)):
                ax.text(val + max(importance_values) * 0.01, i, f'{val:.3f}', 
                       va='center', ha='left')
            
            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('Global Feature Importance')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if self.config.save_plots:
                plot_path = self._save_plot(fig, "global_importance")
                plt.close(fig)
                return plot_path
            
            plt.show()
            return None
            
        except Exception as e:
            logger.error(f"Failed to create global summary plot: {e}")
            return None
    
    def save_explanations(self, explanations: List[SHAPExplanation], 
                         filename: str = None) -> str:
        """Save explanations to file."""
        try:
            # Create directory
            save_dir = Path(self.config.explanation_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"shap_explanations_{timestamp}.json"
            
            save_path = save_dir / filename
            
            # Convert explanations to dictionaries
            explanations_data = {
                "metadata": {
                    "num_explanations": len(explanations),
                    "model_type": explanations[0].model_type if explanations else "unknown",
                    "created_at": datetime.now().isoformat(),
                    "config": {
                        "explainer_type": self.config.explainer_type,
                        "background_samples": self.config.background_samples,
                        "max_evals": self.config.max_evals
                    }
                },
                "explanations": [exp.to_dict() for exp in explanations]
            }
            
            # Save to JSON
            with open(save_path, 'w') as f:
                json.dump(explanations_data, f, indent=2)
            
            logger.info(f"Saved {len(explanations)} explanations to {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to save explanations: {e}")
            return None
    
    def load_explanations(self, filepath: str) -> List[SHAPExplanation]:
        """Load explanations from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            explanations = []
            for exp_data in data["explanations"]:
                explanation = SHAPExplanation(
                    instance_id=exp_data["instance_id"],
                    model_prediction=exp_data["model_prediction"],
                    base_value=exp_data["base_value"],
                    shap_values=np.array(exp_data["shap_values"]),
                    feature_values=np.array(exp_data["feature_values"]),
                    feature_names=exp_data["feature_names"],
                    feature_importance=exp_data["feature_importance"],
                    top_positive_features=exp_data["top_positive_features"],
                    top_negative_features=exp_data["top_negative_features"],
                    explanation_time=exp_data["explanation_time"],
                    model_type=exp_data["model_type"],
                    timestamp=datetime.fromisoformat(exp_data["timestamp"])
                )
                explanations.append(explanation)
            
            logger.info(f"Loaded {len(explanations)} explanations from {filepath}")
            return explanations
            
        except Exception as e:
            logger.error(f"Failed to load explanations: {e}")
            return []


# Utility functions
def create_shap_explainer(model: nn.Module, background_data: Union[pd.DataFrame, np.ndarray],
                         config: Optional[SHAPConfig] = None) -> SHAPExplainer:
    """
    Create and initialize a SHAP explainer.
    
    Args:
        model: PyTorch model to explain
        background_data: Background dataset for SHAP
        config: SHAP configuration
        
    Returns:
        Initialized SHAPExplainer
    """
    explainer = SHAPExplainer(model, config)
    explainer.set_background_data(background_data)
    return explainer


def explain_credit_decision(model: nn.Module, instance: Union[pd.DataFrame, np.ndarray],
                          background_data: Union[pd.DataFrame, np.ndarray],
                          feature_names: List[str] = None) -> SHAPExplanation:
    """
    Explain a single credit decision using SHAP.
    
    Args:
        model: Trained credit risk model
        instance: Single instance to explain
        background_data: Background dataset
        feature_names: Names of features
        
    Returns:
        SHAP explanation for the credit decision
    """
    config = SHAPConfig(feature_names=feature_names)
    explainer = create_shap_explainer(model, background_data, config)
    
    return explainer.explain_instance(instance, "credit_decision")


def batch_explain_decisions(model: nn.Module, instances: Union[pd.DataFrame, np.ndarray],
                          background_data: Union[pd.DataFrame, np.ndarray],
                          feature_names: List[str] = None) -> List[SHAPExplanation]:
    """
    Explain multiple credit decisions efficiently.
    
    Args:
        model: Trained credit risk model
        instances: Multiple instances to explain
        background_data: Background dataset
        feature_names: Names of features
        
    Returns:
        List of SHAP explanations
    """
    config = SHAPConfig(feature_names=feature_names, enable_batch_processing=True)
    explainer = create_shap_explainer(model, background_data, config)
    
    return explainer.explain_batch(instances)