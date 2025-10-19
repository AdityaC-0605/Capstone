"""
Attention Mechanism Visualization for Neural Network Models.

This module implements comprehensive attention weight extraction and visualization
for LSTM and GNN models, including temporal attention heatmaps, feature attention
visualization, and attention-based explanation reports.
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

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
class AttentionConfig:
    """Configuration for attention visualization."""

    # Visualization settings
    save_plots: bool = True
    plot_format: str = "png"  # "png", "svg", "html"
    explanation_path: str = "explanations/attention"

    # Heatmap settings
    heatmap_cmap: str = "viridis"  # "viridis", "plasma", "Blues", "Reds"
    heatmap_figsize: Tuple[int, int] = (12, 8)
    show_values: bool = True

    # Feature names and labels
    feature_names: Optional[List[str]] = None
    timestep_labels: Optional[List[str]] = None

    # Filtering and processing
    attention_threshold: float = 0.01  # Minimum attention weight to display
    top_k_features: int = 10  # Number of top features to highlight

    # Report settings
    include_statistics: bool = True
    generate_summary: bool = True


@dataclass
class AttentionExplanation:
    """Container for attention-based explanation results."""

    # Basic information
    instance_id: str
    model_type: str
    model_prediction: float

    # Attention weights
    attention_weights: np.ndarray
    attention_statistics: Dict[str, float]

    # Feature analysis
    top_attended_features: List[Tuple[str, float]]
    attention_distribution: Dict[str, float]

    # Metadata
    explanation_time: float

    # Temporal analysis (for LSTM)
    temporal_attention: Optional[np.ndarray] = None
    peak_attention_timesteps: Optional[List[int]] = None

    # Graph analysis (for GNN)
    node_attention: Optional[np.ndarray] = None
    edge_attention: Optional[np.ndarray] = None

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to dictionary."""
        return {
            "instance_id": self.instance_id,
            "model_type": self.model_type,
            "model_prediction": float(self.model_prediction),
            "attention_weights": self.attention_weights.tolist(),
            "attention_statistics": self.attention_statistics,
            "top_attended_features": self.top_attended_features,
            "attention_distribution": self.attention_distribution,
            "temporal_attention": (
                self.temporal_attention.tolist()
                if self.temporal_attention is not None
                else None
            ),
            "peak_attention_timesteps": self.peak_attention_timesteps,
            "node_attention": (
                self.node_attention.tolist()
                if self.node_attention is not None
                else None
            ),
            "edge_attention": (
                self.edge_attention.tolist()
                if self.edge_attention is not None
                else None
            ),
            "explanation_time": self.explanation_time,
            "timestamp": self.timestamp.isoformat(),
        }


class AttentionExtractor:
    """Base class for extracting attention weights from models."""

    def __init__(
        self, model: nn.Module, config: Optional[AttentionConfig] = None
    ):
        self.model = model
        self.config = config or AttentionConfig()
        self.model.eval()

    def extract_attention(
        self, X: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Extract attention weights from model. To be implemented by subclasses."""
        raise NotImplementedError(
            "Subclasses must implement extract_attention method"
        )

    def get_model_prediction(self, X: torch.Tensor, **kwargs) -> float:
        """Get model prediction for the input."""
        with torch.no_grad():
            # Handle different input types
            if hasattr(X, "x"):  # Graph data
                output = self.model(X)
            else:  # Tensor data - check if LSTM model needs lengths
                if (
                    hasattr(self.model, "forward")
                    and "lengths" in self.model.forward.__code__.co_varnames
                ):
                    lengths = kwargs.get("lengths")
                    if lengths is None:
                        # Create default lengths if not provided
                        lengths = torch.full(
                            (X.shape[0],), X.shape[1], device=X.device
                        )
                    output = self.model(X, lengths)
                else:
                    output = self.model(X)

            if hasattr(output, "squeeze"):
                output = output.squeeze()

            # Convert to probability
            if len(output.shape) == 0 or (
                len(output.shape) == 1 and output.shape[0] == 1
            ):
                prob = torch.sigmoid(output)
            else:
                prob = torch.softmax(output, dim=-1)
                prob = (
                    prob[1] if len(prob) > 1 else prob[0]
                )  # Get positive class probability

            return float(prob)


class LSTMAttentionExtractor(AttentionExtractor):
    """Extract attention weights from LSTM models."""

    def extract_attention(
        self, X: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Extract attention weights from LSTM model."""

        # Ensure model is in eval mode
        self.model.eval()

        with torch.no_grad():
            # Forward pass to get attention weights
            _ = self.model(X, lengths)

            # Get attention weights from the model
            if (
                hasattr(self.model, "last_attention_weights")
                and self.model.last_attention_weights is not None
            ):
                attention_weights = self.model.last_attention_weights
            elif hasattr(self.model, "get_attention_weights"):
                attention_weights = self.model.get_attention_weights()
            elif hasattr(self.model, "attention") and hasattr(
                self.model.attention, "last_attention_weights"
            ):
                attention_weights = self.model.attention.last_attention_weights
            else:
                # Try to extract attention from forward hooks
                attention_weights = self._extract_attention_with_hooks(
                    X, lengths
                )

                if attention_weights is None:
                    # If no attention weights available, create uniform weights
                    batch_size, seq_len = X.shape[:2]
                    attention_weights = (
                        torch.ones(batch_size, seq_len) / seq_len
                    )
                    logger.warning(
                        "No attention weights found in LSTM model, using uniform weights"
                    )

            return {
                "temporal_attention": attention_weights,
                "sequence_lengths": (
                    lengths
                    if lengths is not None
                    else torch.full((X.shape[0],), X.shape[1])
                ),
            }

    def _extract_attention_with_hooks(
        self, X: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Extract attention weights using forward hooks."""

        attention_weights = None

        def attention_hook(module, input, output):
            nonlocal attention_weights
            if isinstance(output, tuple) and len(output) >= 2:
                # Assume second output is attention weights
                attention_weights = output[1]
            elif hasattr(module, "attention_weights"):
                attention_weights = module.attention_weights

        # Register hooks on attention layers
        hooks = []
        for name, module in self.model.named_modules():
            if "attention" in name.lower():
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)

        try:
            # Forward pass
            _ = self.model(X, lengths)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return attention_weights


class GNNAttentionExtractor(AttentionExtractor):
    """Extract attention weights from GNN models."""

    def extract_attention(self, data, **kwargs) -> Dict[str, torch.Tensor]:
        """Extract attention weights from GNN model."""

        # Ensure model is in eval mode
        self.model.eval()

        with torch.no_grad():
            # Forward pass
            if hasattr(data, "batch"):
                _ = self.model(data)
            else:
                # Handle different data formats
                _ = self.model(
                    data.x, data.edge_index, getattr(data, "batch", None)
                )

            attention_weights = {}

            # Extract GAT attention weights if available
            if hasattr(self.model, "conv_layers"):
                for i, layer in enumerate(self.model.conv_layers):
                    if hasattr(layer, "attention_weights"):
                        attention_weights[f"layer_{i}_attention"] = (
                            layer.attention_weights
                        )
                    elif hasattr(
                        layer, "_alpha"
                    ):  # GAT attention coefficients
                        attention_weights[f"layer_{i}_attention"] = (
                            layer._alpha
                        )

            # Extract pooling attention weights if available
            if hasattr(self.model, "pooling"):
                if hasattr(self.model.pooling, "last_attention_weights"):
                    attention_weights["pooling_attention"] = (
                        self.model.pooling.last_attention_weights
                    )
                elif isinstance(self.model.pooling, AttentionPooling):
                    # Try to get attention from pooling layer
                    pooling_attention = self._extract_pooling_attention(data)
                    if pooling_attention is not None:
                        attention_weights["pooling_attention"] = (
                            pooling_attention
                        )

            # Try to extract attention using hooks if nothing found
            if not attention_weights:
                hook_attention = self._extract_attention_with_hooks(data)
                if hook_attention:
                    attention_weights.update(hook_attention)

            # If still no attention weights found, create uniform weights
            if not attention_weights:
                num_nodes = data.x.shape[0]
                attention_weights["uniform_attention"] = (
                    torch.ones(num_nodes) / num_nodes
                )
                logger.warning(
                    "No attention weights found in GNN model, using uniform weights"
                )

            return attention_weights

    def _extract_pooling_attention(self, data) -> Optional[torch.Tensor]:
        """Extract attention weights from pooling layer."""

        try:
            if hasattr(self.model, "pooling") and hasattr(
                self.model.pooling, "attention"
            ):
                # Calculate attention scores manually
                x = data.x
                attention_scores = self.model.pooling.attention(x)
                attention_weights = torch.softmax(
                    attention_scores.squeeze(-1), dim=0
                )
                return attention_weights
        except Exception as e:
            logger.debug(f"Could not extract pooling attention: {e}")

        return None

    def _extract_attention_with_hooks(self, data) -> Dict[str, torch.Tensor]:
        """Extract attention weights using forward hooks."""

        attention_data = {}

        def gat_hook(module, input, output):
            if hasattr(module, "_alpha") and module._alpha is not None:
                attention_data[f"gat_{id(module)}"] = module._alpha

        def attention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attention_data[f"attention_{id(module)}"] = output[1]

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if "gat" in name.lower() or "attention" in name.lower():
                if "gat" in str(type(module)).lower():
                    hook = module.register_forward_hook(gat_hook)
                else:
                    hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)

        try:
            # Forward pass
            if hasattr(data, "batch"):
                _ = self.model(data)
            else:
                _ = self.model(
                    data.x, data.edge_index, getattr(data, "batch", None)
                )
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return attention_data


class AttentionVisualizer:
    """Main class for attention mechanism visualization."""

    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig()

        logger.info("Attention visualizer initialized")

    def explain_lstm_attention(
        self,
        model: nn.Module,
        X: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        instance_id: str = None,
    ) -> AttentionExplanation:
        """Generate attention-based explanation for LSTM model."""

        start_time = datetime.now()

        # Ensure tensors are on the same device as the model
        model_device = next(model.parameters()).device
        X = X.to(model_device)
        if lengths is not None:
            lengths = lengths.to(model_device)

        # Create extractor
        extractor = LSTMAttentionExtractor(model, self.config)

        # Extract attention weights
        attention_data = extractor.extract_attention(X, lengths)
        temporal_attention = attention_data["temporal_attention"]

        # Get model prediction
        prediction = extractor.get_model_prediction(X, lengths=lengths)

        # Process attention weights (use first instance if batch)
        if len(temporal_attention.shape) > 1:
            attention_weights = temporal_attention[0].cpu().numpy()
        else:
            attention_weights = temporal_attention.cpu().numpy()

        # Calculate attention statistics
        attention_stats = self._calculate_attention_statistics(
            attention_weights
        )

        # Get top attended timesteps
        top_indices = np.argsort(attention_weights)[
            -self.config.top_k_features :
        ][::-1]
        top_attended_features = [
            (f"timestep_{i}", float(attention_weights[i])) for i in top_indices
        ]

        # Create attention distribution
        attention_distribution = {
            f"timestep_{i}": float(weight)
            for i, weight in enumerate(attention_weights)
        }

        # Find peak attention timesteps
        threshold = np.mean(attention_weights) + np.std(attention_weights)
        peak_timesteps = np.where(attention_weights > threshold)[0].tolist()

        explanation_time = (datetime.now() - start_time).total_seconds()

        explanation = AttentionExplanation(
            instance_id=instance_id
            or f"lstm_attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_type="LSTM",
            model_prediction=prediction,
            attention_weights=attention_weights,
            attention_statistics=attention_stats,
            top_attended_features=top_attended_features,
            attention_distribution=attention_distribution,
            temporal_attention=attention_weights,
            peak_attention_timesteps=peak_timesteps,
            explanation_time=explanation_time,
        )

        # Log explanation
        audit_logger.log_model_operation(
            user_id="system",
            model_id="attention_visualizer",
            operation="lstm_attention_explanation",
            success=True,
            details={
                "instance_id": explanation.instance_id,
                "explanation_time": explanation_time,
                "num_timesteps": len(attention_weights),
                "peak_timesteps": len(peak_timesteps),
            },
        )

        return explanation

    def explain_gnn_attention(
        self, model: nn.Module, data, instance_id: str = None
    ) -> AttentionExplanation:
        """Generate attention-based explanation for GNN model."""

        start_time = datetime.now()

        # Ensure data is on the same device as the model
        model_device = next(model.parameters()).device
        if hasattr(data, "x"):
            data.x = data.x.to(model_device)
        if hasattr(data, "edge_index"):
            data.edge_index = data.edge_index.to(model_device)
        if hasattr(data, "edge_attr"):
            data.edge_attr = data.edge_attr.to(model_device)
        if hasattr(data, "batch"):
            data.batch = data.batch.to(model_device)

        # Create extractor
        extractor = GNNAttentionExtractor(model, self.config)

        # Extract attention weights
        attention_data = extractor.extract_attention(data)

        # Get model prediction
        prediction = extractor.get_model_prediction(data)

        # Process attention weights
        if "pooling_attention" in attention_data:
            attention_weights = (
                attention_data["pooling_attention"].cpu().numpy()
            )
        elif "layer_0_attention" in attention_data:
            attention_weights = (
                attention_data["layer_0_attention"].cpu().numpy()
            )
        else:
            attention_weights = list(attention_data.values())[0].cpu().numpy()

        # Flatten if needed
        if len(attention_weights.shape) > 1:
            attention_weights = attention_weights.flatten()

        # Calculate attention statistics
        attention_stats = self._calculate_attention_statistics(
            attention_weights
        )

        # Get top attended nodes
        top_indices = np.argsort(attention_weights)[
            -self.config.top_k_features :
        ][::-1]
        top_attended_features = [
            (f"node_{i}", float(attention_weights[i])) for i in top_indices
        ]

        # Create attention distribution
        attention_distribution = {
            f"node_{i}": float(weight)
            for i, weight in enumerate(attention_weights)
        }

        explanation_time = (datetime.now() - start_time).total_seconds()

        explanation = AttentionExplanation(
            instance_id=instance_id
            or f"gnn_attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_type="GNN",
            model_prediction=prediction,
            attention_weights=attention_weights,
            attention_statistics=attention_stats,
            top_attended_features=top_attended_features,
            attention_distribution=attention_distribution,
            node_attention=attention_weights,
            explanation_time=explanation_time,
        )

        # Log explanation
        audit_logger.log_model_operation(
            user_id="system",
            model_id="attention_visualizer",
            operation="gnn_attention_explanation",
            success=True,
            details={
                "instance_id": explanation.instance_id,
                "explanation_time": explanation_time,
                "num_nodes": len(attention_weights),
            },
        )

        return explanation

    def _calculate_attention_statistics(
        self, attention_weights: np.ndarray
    ) -> Dict[str, float]:
        """Calculate statistical measures of attention distribution."""

        stats = {
            "mean": float(np.mean(attention_weights)),
            "std": float(np.std(attention_weights)),
            "min": float(np.min(attention_weights)),
            "max": float(np.max(attention_weights)),
            "entropy": float(
                -np.sum(attention_weights * np.log(attention_weights + 1e-10))
            ),
            "gini": float(self._calculate_gini_coefficient(attention_weights)),
            "concentration_ratio": float(
                np.sum(np.sort(attention_weights)[-3:])
            ),  # Top 3 concentration
        }

        return stats

    def _calculate_gini_coefficient(
        self, attention_weights: np.ndarray
    ) -> float:
        """Calculate Gini coefficient to measure attention concentration."""
        sorted_weights = np.sort(attention_weights)
        n = len(sorted_weights)
        cumsum = np.cumsum(sorted_weights)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def create_temporal_heatmap(
        self,
        explanation: AttentionExplanation,
        feature_names: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Create temporal attention heatmap for LSTM explanations."""

        if (
            explanation.model_type != "LSTM"
            or explanation.temporal_attention is None
        ):
            logger.warning(
                "Temporal heatmap only available for LSTM models with temporal attention"
            )
            return None

        try:
            fig, ax = plt.subplots(figsize=self.config.heatmap_figsize)

            # Prepare data for heatmap
            attention_data = explanation.temporal_attention.reshape(1, -1)

            # Create heatmap
            im = ax.imshow(
                attention_data, cmap=self.config.heatmap_cmap, aspect="auto"
            )

            # Customize plot
            ax.set_title(
                f"Temporal Attention Heatmap - {explanation.instance_id}"
            )
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Sequence")

            # Set ticks
            num_timesteps = attention_data.shape[1]
            tick_positions = np.linspace(
                0, num_timesteps - 1, min(10, num_timesteps), dtype=int
            )
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"t{i}" for i in tick_positions])
            ax.set_yticks([0])
            ax.set_yticklabels(["Input Sequence"])

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Attention Weight")

            # Add value annotations if requested
            if self.config.show_values and num_timesteps <= 20:
                for i in range(num_timesteps):
                    text = ax.text(
                        i,
                        0,
                        f"{attention_data[0, i]:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

            plt.tight_layout()

            # Save plot
            if self.config.save_plots:
                plot_path = self._save_plot(
                    fig, f"temporal_heatmap_{explanation.instance_id}"
                )
                plt.close(fig)
                return plot_path

            plt.show()
            return None

        except Exception as e:
            logger.error(f"Failed to create temporal heatmap: {e}")
            return None

    def create_feature_attention_plot(
        self, explanation: AttentionExplanation
    ) -> Optional[str]:
        """Create feature attention visualization."""

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Plot 1: Top attended features
            top_features = explanation.top_attended_features[
                : self.config.top_k_features
            ]
            features = [f[0] for f in top_features]
            weights = [f[1] for f in top_features]

            bars = ax1.barh(
                range(len(features)), weights, color="skyblue", alpha=0.7
            )
            ax1.set_yticks(range(len(features)))
            ax1.set_yticklabels(features)
            ax1.set_xlabel("Attention Weight")
            ax1.set_title("Top Attended Features")
            ax1.grid(True, alpha=0.3)

            # Add value labels
            for i, (bar, weight) in enumerate(zip(bars, weights)):
                ax1.text(
                    weight + 0.001,
                    i,
                    f"{weight:.3f}",
                    va="center",
                    ha="left",
                    fontsize=9,
                )

            # Plot 2: Attention distribution histogram
            ax2.hist(
                explanation.attention_weights,
                bins=20,
                alpha=0.7,
                color="lightcoral",
            )
            ax2.set_xlabel("Attention Weight")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Attention Weight Distribution")
            ax2.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f"""Statistics:
Mean: {explanation.attention_statistics['mean']:.4f}
Std: {explanation.attention_statistics['std']:.4f}
Entropy: {explanation.attention_statistics['entropy']:.4f}
Gini: {explanation.attention_statistics['gini']:.4f}"""

            ax2.text(
                0.02,
                0.98,
                stats_text,
                transform=ax2.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            plt.suptitle(f"Attention Analysis - {explanation.instance_id}")
            plt.tight_layout()

            # Save plot
            if self.config.save_plots:
                plot_path = self._save_plot(
                    fig, f"feature_attention_{explanation.instance_id}"
                )
                plt.close(fig)
                return plot_path

            plt.show()
            return None

        except Exception as e:
            logger.error(f"Failed to create feature attention plot: {e}")
            return None

    def create_attention_flow_plot(
        self, explanation: AttentionExplanation
    ) -> Optional[str]:
        """Create attention flow visualization showing how attention changes over time/nodes."""

        try:
            if (
                explanation.model_type == "LSTM"
                and explanation.temporal_attention is not None
            ):
                return self._create_temporal_flow_plot(explanation)
            elif (
                explanation.model_type == "GNN"
                and explanation.node_attention is not None
            ):
                return self._create_node_flow_plot(explanation)
            else:
                logger.warning(
                    "Attention flow plot requires temporal or node attention data"
                )
                return None

        except Exception as e:
            logger.error(f"Failed to create attention flow plot: {e}")
            return None

    def _create_temporal_flow_plot(
        self, explanation: AttentionExplanation
    ) -> Optional[str]:
        """Create temporal attention flow plot for LSTM models."""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        attention_weights = explanation.temporal_attention
        timesteps = np.arange(len(attention_weights))

        # Plot 1: Attention flow over time
        ax1.plot(
            timesteps,
            attention_weights,
            "b-",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax1.fill_between(timesteps, attention_weights, alpha=0.3)
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Attention Weight")
        ax1.set_title("Temporal Attention Flow")
        ax1.grid(True, alpha=0.3)

        # Highlight peak attention timesteps
        if explanation.peak_attention_timesteps:
            peak_weights = [
                attention_weights[i]
                for i in explanation.peak_attention_timesteps
            ]
            ax1.scatter(
                explanation.peak_attention_timesteps,
                peak_weights,
                color="red",
                s=100,
                zorder=5,
                label="Peak Attention",
            )
            ax1.legend()

        # Plot 2: Cumulative attention
        cumulative_attention = np.cumsum(attention_weights)
        ax2.plot(timesteps, cumulative_attention, "g-", linewidth=2)
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Cumulative Attention")
        ax2.set_title("Cumulative Attention Over Time")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"Attention Flow Analysis - {explanation.instance_id}")
        plt.tight_layout()

        # Save plot
        if self.config.save_plots:
            plot_path = self._save_plot(
                fig, f"attention_flow_{explanation.instance_id}"
            )
            plt.close(fig)
            return plot_path

        plt.show()
        return None

    def _create_node_flow_plot(
        self, explanation: AttentionExplanation
    ) -> Optional[str]:
        """Create node attention flow plot for GNN models."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        attention_weights = explanation.node_attention
        nodes = np.arange(len(attention_weights))

        # Plot 1: Node attention weights
        bars = ax1.bar(nodes, attention_weights, color="lightblue", alpha=0.7)
        ax1.set_xlabel("Node Index")
        ax1.set_ylabel("Attention Weight")
        ax1.set_title("Node Attention Weights")
        ax1.grid(True, alpha=0.3)

        # Highlight top nodes
        top_indices = np.argsort(attention_weights)[-3:]
        for idx in top_indices:
            bars[idx].set_color("orange")

        # Plot 2: Attention weight ranking
        sorted_indices = np.argsort(attention_weights)[::-1]
        sorted_weights = attention_weights[sorted_indices]

        ax2.plot(
            range(len(sorted_weights)),
            sorted_weights,
            "ro-",
            linewidth=2,
            markersize=4,
        )
        ax2.set_xlabel("Node Rank")
        ax2.set_ylabel("Attention Weight")
        ax2.set_title("Attention Weight Ranking")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"Node Attention Analysis - {explanation.instance_id}")
        plt.tight_layout()

        # Save plot
        if self.config.save_plots:
            plot_path = self._save_plot(
                fig, f"node_flow_{explanation.instance_id}"
            )
            plt.close(fig)
            return plot_path

        plt.show()
        return None

    def generate_attention_report(
        self, explanation: AttentionExplanation
    ) -> str:
        """Generate comprehensive attention-based explanation report."""

        report_lines = [
            f"# Attention-Based Explanation Report",
            f"",
            f"**Instance ID:** {explanation.instance_id}",
            f"**Model Type:** {explanation.model_type}",
            f"**Prediction:** {explanation.model_prediction:.4f}",
            f"**Generated:** {explanation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Processing Time:** {explanation.explanation_time:.4f} seconds",
            f"",
            f"## Attention Analysis Summary",
            f"",
        ]

        # Add model-specific analysis
        if explanation.model_type == "LSTM":
            report_lines.extend(
                [
                    f"### Temporal Attention Pattern",
                    f"",
                    f"The model focused on **{len(explanation.peak_attention_timesteps or [])} key timesteps** "
                    f"out of {len(explanation.attention_weights)} total timesteps.",
                    f"",
                ]
            )

            if explanation.peak_attention_timesteps:
                report_lines.extend(
                    [
                        f"**Peak Attention Timesteps:**",
                        f"",
                    ]
                )
                for timestep in explanation.peak_attention_timesteps:
                    weight = explanation.attention_weights[timestep]
                    report_lines.append(f"- Timestep {timestep}: {weight:.4f}")
                report_lines.append("")

                # Add temporal analysis
                report_lines.extend(
                    [
                        f"**Temporal Analysis:**",
                        f"",
                        f"- Early attention (first 25%): {np.mean(explanation.attention_weights[:len(explanation.attention_weights)//4]):.4f}",
                        f"- Mid attention (middle 50%): {np.mean(explanation.attention_weights[len(explanation.attention_weights)//4:3*len(explanation.attention_weights)//4]):.4f}",
                        f"- Late attention (last 25%): {np.mean(explanation.attention_weights[3*len(explanation.attention_weights)//4:]):.4f}",
                        f"",
                    ]
                )

        else:  # GNN
            report_lines.extend(
                [
                    f"### Graph Attention Pattern",
                    f"",
                    f"The model analyzed **{len(explanation.attention_weights)} nodes** "
                    f"with varying attention weights.",
                    f"",
                ]
            )

            # Add graph-specific analysis
            if explanation.node_attention is not None:
                top_nodes = np.argsort(explanation.node_attention)[-3:][::-1]
                report_lines.extend(
                    [
                        f"**Most Important Nodes:**",
                        f"",
                    ]
                )
                for i, node_idx in enumerate(top_nodes, 1):
                    weight = explanation.node_attention[node_idx]
                    report_lines.append(f"{i}. Node {node_idx}: {weight:.4f}")
                report_lines.append("")

        # Add top attended features
        report_lines.extend(
            [
                f"### Top Attended Elements",
                f"",
            ]
        )

        for i, (feature, weight) in enumerate(
            explanation.top_attended_features[:5], 1
        ):
            percentage = (weight / np.sum(explanation.attention_weights)) * 100
            report_lines.append(
                f"{i}. **{feature}**: {weight:.4f} ({percentage:.1f}% of total attention)"
            )

        report_lines.extend(
            [
                f"",
                f"### Attention Statistics",
                f"",
                f"- **Mean Attention:** {explanation.attention_statistics['mean']:.4f}",
                f"- **Standard Deviation:** {explanation.attention_statistics['std']:.4f}",
                f"- **Min/Max Attention:** {explanation.attention_statistics['min']:.4f} / {explanation.attention_statistics['max']:.4f}",
                f"- **Entropy:** {explanation.attention_statistics['entropy']:.4f} "
                f"(lower = more concentrated)",
                f"- **Gini Coefficient:** {explanation.attention_statistics['gini']:.4f} "
                f"(higher = more concentrated)",
                f"- **Top-3 Concentration:** {explanation.attention_statistics['concentration_ratio']:.4f}",
                f"",
            ]
        )

        # Add interpretation
        entropy = explanation.attention_statistics["entropy"]
        gini = explanation.attention_statistics["gini"]

        if entropy < 2.0:
            attention_pattern = "highly concentrated"
        elif entropy < 3.0:
            attention_pattern = "moderately concentrated"
        else:
            attention_pattern = "distributed"

        report_lines.extend(
            [
                f"### Interpretation",
                f"",
                f"The attention pattern is **{attention_pattern}**, indicating that the model ",
            ]
        )

        if attention_pattern == "highly concentrated":
            report_lines.append(
                f"focuses on a small number of key elements for its decision."
            )
        elif attention_pattern == "moderately concentrated":
            report_lines.append(
                f"balances focus between several important elements."
            )
        else:
            report_lines.append(
                f"considers many elements with relatively equal importance."
            )

        # Add decision confidence analysis
        max_attention = explanation.attention_statistics["max"]
        if max_attention > 0.5:
            confidence_level = "high"
        elif max_attention > 0.3:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        report_lines.extend(
            [
                f"",
                f"The model shows **{confidence_level} confidence** in its attention allocation, "
                f"with the highest attention weight being {max_attention:.4f}.",
                f"",
            ]
        )

        # Add regulatory compliance notes
        report_lines.extend(
            [
                f"### Regulatory Compliance Notes",
                f"",
                f"- **Explainability:** This report provides detailed attention weights for model transparency",
                f"- **Feature Importance:** Top contributing elements are clearly identified and ranked",
                f"- **Statistical Analysis:** Comprehensive attention distribution statistics provided",
                f"- **Audit Trail:** Complete processing metadata included for compliance reviews",
                f"",
                f"---",
                f"*Report generated by Attention Visualizer v1.0*",
                f"*Compliant with explainable AI requirements for financial services*",
            ]
        )

        return "\n".join(report_lines)

    def create_comprehensive_attention_analysis(
        self, explanation: AttentionExplanation
    ) -> Dict[str, Any]:
        """Create comprehensive attention analysis with multiple visualizations and insights."""

        analysis = {
            "explanation": explanation,
            "visualizations": {},
            "insights": {},
            "compliance_data": {},
        }

        try:
            # Generate visualizations
            analysis["visualizations"]["temporal_heatmap"] = (
                self.create_temporal_heatmap(explanation)
            )
            analysis["visualizations"]["feature_attention"] = (
                self.create_feature_attention_plot(explanation)
            )
            analysis["visualizations"]["attention_flow"] = (
                self.create_attention_flow_plot(explanation)
            )

            # Generate insights
            analysis["insights"] = self._generate_attention_insights(
                explanation
            )

            # Generate compliance data
            analysis["compliance_data"] = self._generate_compliance_data(
                explanation
            )

            # Generate report
            analysis["report"] = self.generate_attention_report(explanation)

            logger.info(
                f"Comprehensive attention analysis completed for {explanation.instance_id}"
            )

        except Exception as e:
            logger.error(f"Failed to create comprehensive analysis: {e}")
            analysis["error"] = str(e)

        return analysis

    def _generate_attention_insights(
        self, explanation: AttentionExplanation
    ) -> Dict[str, Any]:
        """Generate detailed insights from attention patterns."""

        insights = {
            "attention_concentration": {},
            "temporal_patterns": {},
            "decision_factors": {},
            "model_behavior": {},
        }

        # Attention concentration analysis
        entropy = explanation.attention_statistics["entropy"]
        gini = explanation.attention_statistics["gini"]

        insights["attention_concentration"] = {
            "level": (
                "high"
                if entropy < 2.0
                else "moderate" if entropy < 3.0 else "low"
            ),
            "entropy_score": entropy,
            "gini_coefficient": gini,
            "interpretation": (
                "Focused decision-making"
                if entropy < 2.0
                else (
                    "Balanced consideration"
                    if entropy < 3.0
                    else "Distributed analysis"
                )
            ),
        }

        # Temporal patterns (for LSTM)
        if (
            explanation.model_type == "LSTM"
            and explanation.temporal_attention is not None
        ):
            attention_weights = explanation.temporal_attention
            seq_len = len(attention_weights)

            # Identify attention phases
            early_attention = np.mean(attention_weights[: seq_len // 3])
            mid_attention = np.mean(
                attention_weights[seq_len // 3 : 2 * seq_len // 3]
            )
            late_attention = np.mean(attention_weights[2 * seq_len // 3 :])

            dominant_phase = (
                "early"
                if early_attention > max(mid_attention, late_attention)
                else "middle" if mid_attention > late_attention else "late"
            )

            insights["temporal_patterns"] = {
                "dominant_phase": dominant_phase,
                "early_attention": early_attention,
                "mid_attention": mid_attention,
                "late_attention": late_attention,
                "attention_trend": (
                    "increasing"
                    if late_attention > early_attention
                    else "decreasing"
                ),
            }

        # Decision factors
        top_features = explanation.top_attended_features[:3]
        total_top_attention = sum(weight for _, weight in top_features)

        insights["decision_factors"] = {
            "primary_factors": len([w for _, w in top_features if w > 0.1]),
            "top_3_contribution": total_top_attention,
            "decision_complexity": (
                "simple"
                if total_top_attention > 0.7
                else "moderate" if total_top_attention > 0.5 else "complex"
            ),
        }

        # Model behavior analysis
        max_attention = explanation.attention_statistics["max"]
        mean_attention = explanation.attention_statistics["mean"]

        insights["model_behavior"] = {
            "confidence_level": (
                "high"
                if max_attention > 0.5
                else "moderate" if max_attention > 0.3 else "low"
            ),
            "attention_variance": explanation.attention_statistics["std"],
            "uniformity": (
                "uniform"
                if explanation.attention_statistics["std"] < 0.1
                else "varied"
            ),
        }

        return insights

    def _generate_compliance_data(
        self, explanation: AttentionExplanation
    ) -> Dict[str, Any]:
        """Generate compliance-relevant data for regulatory requirements."""

        compliance_data = {
            "explainability_score": 0.0,
            "transparency_metrics": {},
            "audit_information": {},
            "regulatory_notes": [],
        }

        # Calculate explainability score (0-1)
        entropy = explanation.attention_statistics["entropy"]
        max_attention = explanation.attention_statistics["max"]

        # Higher explainability for more concentrated attention
        concentration_score = min(1.0, max_attention * 2)
        # Higher explainability for lower entropy (more focused)
        entropy_score = max(0.0, 1.0 - entropy / 5.0)

        compliance_data["explainability_score"] = (
            concentration_score + entropy_score
        ) / 2

        # Transparency metrics
        compliance_data["transparency_metrics"] = {
            "attention_coverage": len(
                [w for w in explanation.attention_weights if w > 0.01]
            ),
            "primary_factor_count": len(
                [w for _, w in explanation.top_attended_features if w > 0.1]
            ),
            "decision_interpretability": (
                "high"
                if compliance_data["explainability_score"] > 0.7
                else (
                    "moderate"
                    if compliance_data["explainability_score"] > 0.5
                    else "low"
                )
            ),
        }

        # Audit information
        compliance_data["audit_information"] = {
            "model_type": explanation.model_type,
            "prediction_value": explanation.model_prediction,
            "processing_timestamp": explanation.timestamp.isoformat(),
            "attention_statistics": explanation.attention_statistics,
            "top_contributing_factors": explanation.top_attended_features[:5],
        }

        # Regulatory notes
        if compliance_data["explainability_score"] > 0.7:
            compliance_data["regulatory_notes"].append(
                "High explainability - suitable for regulatory review"
            )
        elif compliance_data["explainability_score"] > 0.5:
            compliance_data["regulatory_notes"].append(
                "Moderate explainability - additional documentation may be required"
            )
        else:
            compliance_data["regulatory_notes"].append(
                "Low explainability - detailed review recommended"
            )

        if explanation.model_type == "LSTM":
            compliance_data["regulatory_notes"].append(
                "Temporal attention patterns available for sequence analysis"
            )
        elif explanation.model_type == "GNN":
            compliance_data["regulatory_notes"].append(
                "Graph attention patterns available for relationship analysis"
            )

        return compliance_data

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
        self, explanations: List[AttentionExplanation], filename: str = None
    ) -> str:
        """Save explanations to file."""
        try:
            # Create directory if it doesn't exist
            save_dir = Path(self.config.explanation_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"attention_explanations_{timestamp}.json"

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


# Utility functions for easy integration


def visualize_lstm_attention(
    model: nn.Module,
    X: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    config: Optional[AttentionConfig] = None,
) -> AttentionExplanation:
    """
    Visualize attention for LSTM model.

    Args:
        model: LSTM model with attention mechanism
        X: Input tensor
        lengths: Sequence lengths (optional)
        config: Visualization configuration

    Returns:
        AttentionExplanation object
    """
    visualizer = AttentionVisualizer(config)
    return visualizer.explain_lstm_attention(model, X, lengths)


def visualize_gnn_attention(
    model: nn.Module, data, config: Optional[AttentionConfig] = None
) -> AttentionExplanation:
    """
    Visualize attention for GNN model.

    Args:
        model: GNN model with attention mechanism
        data: Graph data object
        config: Visualization configuration

    Returns:
        AttentionExplanation object
    """
    visualizer = AttentionVisualizer(config)
    return visualizer.explain_gnn_attention(model, data)


def create_attention_dashboard(
    explanations: List[AttentionExplanation],
    config: Optional[AttentionConfig] = None,
) -> str:
    """
    Create comprehensive attention analysis dashboard.

    Args:
        explanations: List of attention explanations
        config: Visualization configuration

    Returns:
        Path to generated dashboard HTML file
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly required for dashboard creation")
        return None

    config = config or AttentionConfig()

    # Create dashboard with multiple subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Attention Distribution",
            "Model Predictions",
            "Attention Statistics",
            "Temporal Patterns",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Add plots for each explanation
    for i, exp in enumerate(explanations):
        # Attention distribution
        fig.add_trace(
            go.Histogram(
                x=exp.attention_weights,
                name=f"{exp.instance_id}",
                opacity=0.7,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Model predictions
        fig.add_trace(
            go.Scatter(
                x=[i],
                y=[exp.model_prediction],
                mode="markers",
                name=f"{exp.instance_id}",
                marker=dict(size=10),
            ),
            row=1,
            col=2,
        )

    # Update layout
    fig.update_layout(
        title="Attention Analysis Dashboard", height=800, showlegend=True
    )

    # Save dashboard
    try:
        dashboard_dir = Path(config.explanation_path) / "dashboards"
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        dashboard_path = (
            dashboard_dir
            / f"attention_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        fig.write_html(str(dashboard_path))
        logger.info(f"Dashboard saved to {dashboard_path}")
        return str(dashboard_path)
    except Exception as e:
        logger.error(f"Failed to save dashboard: {e}")
        return None


def batch_explain_attention(
    model: nn.Module,
    data_list: List,
    model_type: str = "auto",
    config: Optional[AttentionConfig] = None,
) -> List[AttentionExplanation]:
    """
    Generate attention explanations for multiple instances in batch.

    Args:
        model: Neural network model with attention mechanism
        data_list: List of input data (tensors for LSTM, graph data for GNN)
        model_type: Model type ("LSTM", "GNN", or "auto" for auto-detection)
        config: Visualization configuration

    Returns:
        List of attention explanations
    """
    config = config or AttentionConfig()
    visualizer = AttentionVisualizer(config)
    explanations = []

    # Auto-detect model type if needed
    if model_type == "auto":
        model_type = "LSTM" if "lstm" in str(type(model)).lower() else "GNN"

    logger.info(
        f"Processing {len(data_list)} instances for {model_type} attention analysis"
    )

    for i, data in enumerate(data_list):
        try:
            instance_id = f"batch_{model_type.lower()}_{i}"

            if model_type.upper() == "LSTM":
                if isinstance(data, tuple):
                    X, lengths = data
                else:
                    X, lengths = data, None
                explanation = visualizer.explain_lstm_attention(
                    model, X, lengths, instance_id
                )
            else:  # GNN
                explanation = visualizer.explain_gnn_attention(
                    model, data, instance_id
                )

            explanations.append(explanation)

        except Exception as e:
            logger.error(f"Failed to explain instance {i}: {e}")
            continue

    logger.info(
        f"Successfully generated {len(explanations)} attention explanations"
    )
    return explanations


def compare_attention_patterns(
    explanations: List[AttentionExplanation],
    config: Optional[AttentionConfig] = None,
) -> Dict[str, Any]:
    """
    Compare attention patterns across multiple explanations.

    Args:
        explanations: List of attention explanations to compare
        config: Visualization configuration

    Returns:
        Dictionary containing comparison results and insights
    """
    if not explanations:
        return {"error": "No explanations provided for comparison"}

    config = config or AttentionConfig()

    comparison = {
        "summary": {},
        "statistics": {},
        "patterns": {},
        "insights": [],
    }

    # Basic summary
    comparison["summary"] = {
        "num_explanations": len(explanations),
        "model_types": list(set(exp.model_type for exp in explanations)),
        "avg_prediction": np.mean(
            [exp.model_prediction for exp in explanations]
        ),
        "prediction_std": np.std(
            [exp.model_prediction for exp in explanations]
        ),
    }

    # Statistical comparison
    all_entropies = [
        exp.attention_statistics["entropy"] for exp in explanations
    ]
    all_ginis = [exp.attention_statistics["gini"] for exp in explanations]
    all_max_attentions = [
        exp.attention_statistics["max"] for exp in explanations
    ]

    comparison["statistics"] = {
        "entropy": {
            "mean": np.mean(all_entropies),
            "std": np.std(all_entropies),
        },
        "gini": {"mean": np.mean(all_ginis), "std": np.std(all_ginis)},
        "max_attention": {
            "mean": np.mean(all_max_attentions),
            "std": np.std(all_max_attentions),
        },
    }

    # Pattern analysis
    concentrated_count = sum(1 for entropy in all_entropies if entropy < 2.0)
    distributed_count = sum(1 for entropy in all_entropies if entropy > 3.0)

    comparison["patterns"] = {
        "concentrated_attention": concentrated_count / len(explanations),
        "distributed_attention": distributed_count / len(explanations),
        "consistent_patterns": np.std(all_entropies) < 0.5,
    }

    # Generate insights
    if comparison["patterns"]["concentrated_attention"] > 0.7:
        comparison["insights"].append(
            "Most instances show concentrated attention patterns"
        )
    elif comparison["patterns"]["distributed_attention"] > 0.7:
        comparison["insights"].append(
            "Most instances show distributed attention patterns"
        )
    else:
        comparison["insights"].append(
            "Mixed attention patterns across instances"
        )

    if comparison["patterns"]["consistent_patterns"]:
        comparison["insights"].append(
            "Attention patterns are consistent across instances"
        )
    else:
        comparison["insights"].append(
            "Attention patterns vary significantly across instances"
        )

    if comparison["summary"]["prediction_std"] < 0.1:
        comparison["insights"].append("Model predictions are consistent")
    else:
        comparison["insights"].append("Model predictions show high variance")

    return comparison
