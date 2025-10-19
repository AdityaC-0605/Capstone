"""
Temporal Convolutional Network (TCN) implementation for credit risk prediction.
Uses dilated causal convolutions with residual connections for efficient sequence processing.
"""

import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

try:
    from ..core.interfaces import BaseModel, TrainingMetrics
    from ..core.logging import get_audit_logger, get_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.interfaces import BaseModel, TrainingMetrics
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
class TCNConfig:
    """Configuration for Temporal Convolutional Network model."""

    # Architecture parameters
    input_size: int = 14  # Number of features per timestep
    num_channels: List[int] = field(
        default_factory=lambda: [64, 64, 64, 64]
    )  # Hidden channels per layer
    kernel_size: int = 3  # Convolution kernel size
    dropout_rate: float = 0.2

    # TCN-specific parameters
    dilation_base: int = 2  # Base for exponential dilation
    use_residual: bool = True
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    activation: str = "relu"  # 'relu', 'gelu', 'swish'

    # Sequence parameters
    max_sequence_length: int = 50
    receptive_field_size: Optional[int] = None  # Auto-calculated if None

    # Output layers
    output_hidden_layers: List[int] = field(default_factory=lambda: [64, 32])

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 15
    min_delta: float = 1e-4

    # Optimization parameters
    optimizer: str = "adam"  # 'adam', 'adamw', 'sgd', 'rmsprop'
    weight_decay: float = 1e-4
    gradient_clip_value: float = 1.0

    # Loss function parameters
    loss_function: str = "focal"  # 'bce', 'focal', 'weighted_bce'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # 'onecycle', 'cosine', 'step', 'plateau'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # Mixed precision training
    use_mixed_precision: bool = True

    # Regularization
    l1_lambda: float = 0.0
    l2_lambda: float = 0.0

    # Model saving
    save_model: bool = True
    model_path: str = "models/tcn"
    save_best_only: bool = True

    # Device configuration
    device: str = "auto"  # 'auto', 'cpu', 'cuda', 'mps'


@dataclass
class TCNResult:
    """Result of TCN training and evaluation."""

    success: bool
    model: Optional["TCNModel"]
    config: TCNConfig
    training_metrics: List[TrainingMetrics]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    receptive_field_size: int
    training_time_seconds: float
    model_path: Optional[str]
    best_epoch: int
    message: str


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class Chomp1d(nn.Module):
    """Removes rightmost elements from the temporal dimension to ensure causality."""

    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)

        Returns:
            Chomped tensor of shape (batch_size, channels, seq_len - chomp_size)
        """
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Temporal block with dilated causal convolution, residual connection, and normalization.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        use_batch_norm: bool = False,
        activation: str = "relu",
    ):
        super(TemporalBlock, self).__init__()

        # First dilated causal convolution
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)

        # Normalization
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(n_outputs)
        elif use_batch_norm:
            self.norm1 = nn.BatchNorm1d(n_outputs)
        else:
            self.norm1 = nn.Identity()

        # Activation
        if activation == "relu":
            self.activation1 = nn.ReLU()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
        elif activation == "swish":
            self.activation1 = nn.SiLU()  # Swish activation
        else:
            self.activation1 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)

        # Second dilated causal convolution
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)

        # Normalization
        if use_layer_norm:
            self.norm2 = nn.LayerNorm(n_outputs)
        elif use_batch_norm:
            self.norm2 = nn.BatchNorm1d(n_outputs)
        else:
            self.norm2 = nn.Identity()

        self.activation2 = (
            self.activation1.__class__()
        )  # Same activation as first
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        if self.downsample is not None:
            nn.init.xavier_uniform_(self.downsample.weight)
            if self.downsample.bias is not None:
                nn.init.constant_(self.downsample.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, n_inputs, seq_len)

        Returns:
            Output tensor of shape (batch_size, n_outputs, seq_len)
        """
        # First convolution block
        out = self.conv1(x)
        out = self.chomp1(out)

        # Apply normalization (handle LayerNorm which expects (batch, seq, features))
        if isinstance(self.norm1, nn.LayerNorm):
            out = out.transpose(1, 2)  # (batch, seq, channels)
            out = self.norm1(out)
            out = out.transpose(1, 2)  # (batch, channels, seq)
        else:
            out = self.norm1(out)

        out = self.activation1(out)
        out = self.dropout1(out)

        # Second convolution block
        out = self.conv2(out)
        out = self.chomp2(out)

        # Apply normalization
        if isinstance(self.norm2, nn.LayerNorm):
            out = out.transpose(1, 2)  # (batch, seq, channels)
            out = self.norm2(out)
            out = out.transpose(1, 2)  # (batch, channels, seq)
        else:
            out = self.norm2(out)

        out = self.activation2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)

        # Ensure same sequence length for residual connection
        if res.size(2) != out.size(2):
            # Trim residual to match output length
            res = res[:, :, : out.size(2)]

        return self.activation1(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with multiple temporal blocks.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
        dilation_base: int = 2,
        use_layer_norm: bool = True,
        use_batch_norm: bool = False,
        activation: str = "relu",
    ):
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = dilation_base**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Calculate padding for causal convolution
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                    use_batch_norm=use_batch_norm,
                    activation=activation,
                )
            )

        self.network = nn.Sequential(*layers)

        # Calculate receptive field size
        self.receptive_field = self._calculate_receptive_field(
            kernel_size, num_levels, dilation_base
        )

    def _calculate_receptive_field(
        self, kernel_size: int, num_levels: int, dilation_base: int
    ) -> int:
        """Calculate the receptive field size of the TCN."""
        receptive_field = 1
        for i in range(num_levels):
            dilation = dilation_base**i
            receptive_field += (kernel_size - 1) * dilation
        return receptive_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_inputs, seq_len)

        Returns:
            Output tensor of shape (batch_size, num_channels[-1], seq_len)
        """
        return self.network(x)


class TCNModel(BaseModel):
    """Temporal Convolutional Network model implementation."""

    def __init__(self, config: Optional[TCNConfig] = None):
        super(TCNModel, self).__init__()
        self.config = config or TCNConfig()
        self.device = self._get_device()

        # Build network architecture
        self._build_network()

        # Initialize weights
        self._initialize_weights()

        # Move to device
        self.to(self.device)

        # Training state
        self.is_trained = False
        self.feature_names = None
        self.scaler = StandardScaler()
        self.best_state_dict = None
        self.training_history = []

    def _get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)

    def _build_network(self):
        """Build the TCN architecture."""
        # Temporal Convolutional Network
        self.tcn = TemporalConvNet(
            num_inputs=self.config.input_size,
            num_channels=self.config.num_channels,
            kernel_size=self.config.kernel_size,
            dropout=self.config.dropout_rate,
            dilation_base=self.config.dilation_base,
            use_layer_norm=self.config.use_layer_norm,
            use_batch_norm=self.config.use_batch_norm,
            activation=self.config.activation,
        )

        # Store receptive field size
        self.receptive_field_size = self.tcn.receptive_field

        # Global pooling (take the last timestep)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Output layers
        layers = []
        prev_dim = self.config.num_channels[-1]

        for hidden_dim in self.config.output_hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif self.config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            if self.config.activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif self.config.activation == "gelu":
                layers.append(nn.GELU())
            elif self.config.activation == "swish":
                layers.append(nn.SiLU())

            layers.append(nn.Dropout(self.config.dropout_rate))

            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.output_layers = nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.output_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TCN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            output: Predictions of shape (batch_size,)
        """
        # Transpose for TCN: (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)

        # Apply TCN
        tcn_out = self.tcn(x)  # (batch_size, num_channels[-1], seq_len)

        # Global pooling to get fixed-size representation
        pooled = self.global_pool(tcn_out)  # (batch_size, num_channels[-1], 1)
        pooled = pooled.squeeze(-1)  # (batch_size, num_channels[-1])

        # Apply output layers
        output = self.output_layers(pooled)

        return output.squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(self.device)
            elif not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x).to(self.device)
            else:
                x = x.to(self.device)

            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)

            # Return as 2D array for binary classification
            neg_probs = 1 - probabilities
            return torch.stack([neg_probs, probabilities], dim=1)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Make binary predictions."""
        probs = self.predict_proba(x)
        return (probs[:, 1] > threshold).long()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance using gradient-based method."""
        if not self.is_trained:
            return {}

        try:
            self.eval()

            # Create dummy input
            dummy_seq_len = min(10, self.config.max_sequence_length)
            dummy_input = torch.randn(
                1,
                dummy_seq_len,
                self.config.input_size,
                requires_grad=True,
                device=self.device,
            )

            output = self.forward(dummy_input)

            # Compute gradients
            gradients = torch.autograd.grad(
                output, dummy_input, create_graph=False
            )[0]
            importance_scores = (
                torch.abs(gradients).mean(dim=(0, 1)).cpu().numpy()
            )

            # Create feature importance dictionary
            feature_importance = {}
            for i, score in enumerate(importance_scores):
                feature_name = (
                    self.feature_names[i]
                    if self.feature_names
                    else f"feature_{i}"
                )
                feature_importance[feature_name] = float(score)

            # Sort by importance
            feature_importance = dict(
                sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )

            return feature_importance

        except Exception as e:
            logger.warning(f"Could not compute feature importance: {e}")
            # Fallback: uniform importance
            if self.feature_names:
                return {
                    name: 1.0 / len(self.feature_names)
                    for name in self.feature_names
                }
            else:
                return {
                    f"feature_{i}": 1.0 / self.config.input_size
                    for i in range(self.config.input_size)
                }

    def get_receptive_field_size(self) -> int:
        """Get the receptive field size of the TCN."""
        return self.receptive_field_size

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        # Create save path
        save_path = path or self.config.model_path
        model_dir = Path(save_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_file = model_dir / "tcn_model.pth"
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "best_state_dict": self.best_state_dict,
                "config": self.config,
                "feature_names": self.feature_names,
                "scaler_mean": (
                    self.scaler.mean_.tolist()
                    if hasattr(self.scaler, "mean_")
                    else None
                ),
                "scaler_scale": (
                    self.scaler.scale_.tolist()
                    if hasattr(self.scaler, "scale_")
                    else None
                ),
                "training_history": self.training_history,
                "receptive_field_size": self.receptive_field_size,
                "device": str(self.device),
            },
            model_file,
        )

        # Save metadata
        metadata = {
            "model_type": "tcn",
            "input_size": self.config.input_size,
            "num_channels": self.config.num_channels,
            "kernel_size": self.config.kernel_size,
            "receptive_field_size": self.receptive_field_size,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "config": self.config.__dict__,
            "saved_at": datetime.now().isoformat(),
        }

        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {model_file}")
        return str(model_file)

    def load_model(self, path: str) -> "TCNModel":
        """Load a trained model."""
        model_path = Path(path)

        if model_path.is_file():
            model_file = model_path
        else:
            model_file = model_path / "tcn_model.pth"

        # Load model
        checkpoint = torch.load(
            model_file, map_location=self.device, weights_only=False
        )

        # Restore configuration and architecture
        self.config = checkpoint["config"]
        self.feature_names = checkpoint.get("feature_names")
        self.training_history = checkpoint.get("training_history", [])
        self.receptive_field_size = checkpoint.get("receptive_field_size", 0)

        # Rebuild network with loaded config
        self._build_network()

        # Load state dict
        self.load_state_dict(checkpoint["model_state_dict"])
        self.best_state_dict = checkpoint.get("best_state_dict")

        # Restore scaler
        if "scaler_mean" in checkpoint and "scaler_scale" in checkpoint:
            if checkpoint["scaler_mean"] is not None:
                self.scaler.mean_ = np.array(checkpoint["scaler_mean"])
            if checkpoint["scaler_scale"] is not None:
                self.scaler.scale_ = np.array(checkpoint["scaler_scale"])

        self.is_trained = True
        self.to(self.device)

        logger.info(f"Model loaded from {model_file}")
        return self


class SequenceDataProcessor:
    """Processor for converting tabular data to sequences for TCN."""

    def __init__(self, sequence_length: int = 10, overlap: float = 0.5):
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.step_size = max(1, int(sequence_length * (1 - overlap)))

    def create_sequences(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from tabular data.

        Args:
            X: Feature dataframe
            y: Target series

        Returns:
            sequences: Array of sequences (num_sequences, seq_len, n_features)
            targets: Array of targets for each sequence
        """
        sequences = []
        targets = []

        # Convert to numpy for easier manipulation
        X_np = X.values
        y_np = y.values

        # Create sequences using sliding window
        for i in range(
            0, len(X_np) - self.sequence_length + 1, self.step_size
        ):
            end_idx = min(i + self.sequence_length, len(X_np))
            seq_len = end_idx - i

            if seq_len >= 3:  # Minimum sequence length
                sequence = X_np[i:end_idx]
                target = y_np[
                    end_idx - 1
                ]  # Use the last target in the sequence

                # Pad sequence if necessary
                if seq_len < self.sequence_length:
                    padding = np.zeros(
                        (self.sequence_length - seq_len, X_np.shape[1])
                    )
                    sequence = np.vstack([padding, sequence])

                sequences.append(sequence)
                targets.append(target)

        return np.array(sequences), np.array(targets)


class TCNTrainer:
    """Trainer for Temporal Convolutional Network models."""

    def __init__(self, config: Optional[TCNConfig] = None):
        self.config = config or TCNConfig()
        self.sequence_processor = SequenceDataProcessor(
            sequence_length=self.config.max_sequence_length, overlap=0.3
        )

    def train_and_evaluate(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
    ) -> TCNResult:
        """Train and evaluate TCN model."""
        start_time = datetime.now()

        try:
            logger.info("Starting TCN training and evaluation")

            # Create sequences from tabular data
            sequences, targets = self.sequence_processor.create_sequences(X, y)

            if len(sequences) == 0:
                raise ValueError("No sequences could be created from the data")

            logger.info(
                f"Created {len(sequences)} sequences with length {sequences.shape[1]}"
            )

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                sequences,
                targets,
                test_size=test_size,
                random_state=42,
                stratify=targets,
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=42,
                stratify=y_train,
            )

            logger.info(
                f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
            )

            # Update config with actual input size
            self.config.input_size = sequences.shape[2]

            # Create and train model
            model = TCNModel(self.config)
            model.feature_names = list(X.columns)

            # Train model
            training_metrics = self._train_model(
                model, X_train, y_train, X_val, y_val
            )

            # Evaluate model
            validation_metrics = self._evaluate_model(
                model, X_val, y_val, "Validation"
            )
            test_metrics = self._evaluate_model(model, X_test, y_test, "Test")

            # Get feature importance
            feature_importance = model.get_feature_importance()

            # Save model if requested
            model_path = None
            if self.config.save_model:
                model_path = model.save_model()

            training_time = (datetime.now() - start_time).total_seconds()

            # Find best epoch
            best_epoch = 0
            if training_metrics:
                try:
                    best_auc = max(training_metrics, key=lambda x: x.auc_roc)
                    best_epoch = best_auc.epoch
                except (ValueError, AttributeError):
                    best_epoch = len(training_metrics) - 1

            # Log training completion
            audit_logger.log_model_operation(
                user_id="system",
                model_id="tcn_baseline",
                operation="training_completed",
                success=True,
                details={
                    "training_time_seconds": training_time,
                    "test_auc": test_metrics.get("roc_auc", 0.0),
                    "best_epoch": best_epoch,
                    "num_parameters": sum(
                        p.numel() for p in model.parameters()
                    ),
                    "receptive_field_size": model.get_receptive_field_size(),
                    "num_sequences": len(sequences),
                },
            )

            logger.info(
                f"TCN training completed in {training_time:.2f} seconds"
            )

            return TCNResult(
                success=True,
                model=model,
                config=self.config,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                feature_importance=feature_importance,
                receptive_field_size=model.get_receptive_field_size(),
                training_time_seconds=training_time,
                model_path=model_path,
                best_epoch=best_epoch,
                message="TCN training completed successfully",
            )

        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            error_message = f"TCN training failed: {str(e)}"
            logger.error(error_message)

            return TCNResult(
                success=False,
                model=None,
                config=self.config,
                training_metrics=[],
                validation_metrics={},
                test_metrics={},
                feature_importance={},
                receptive_field_size=0,
                training_time_seconds=training_time,
                model_path=None,
                best_epoch=0,
                message=error_message,
            )

    def _train_model(
        self,
        model: TCNModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> List[TrainingMetrics]:
        """Train the TCN model."""

        # Scale features
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        model.scaler.fit(X_train_reshaped)

        X_train_scaled = model.scaler.transform(X_train_reshaped).reshape(
            n_samples, n_timesteps, n_features
        )

        n_val_samples = X_val.shape[0]
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_val_scaled = model.scaler.transform(X_val_reshaped).reshape(
            n_val_samples, n_timesteps, n_features
        )

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(model.device)
        y_train_tensor = torch.FloatTensor(y_train).to(model.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(model.device)
        y_val_tensor = torch.FloatTensor(y_val).to(model.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # Setup loss function
        if self.config.loss_function == "focal":
            criterion = FocalLoss(
                alpha=self.config.focal_alpha, gamma=self.config.focal_gamma
            )
        elif self.config.loss_function == "weighted_bce":
            pos_weight = torch.tensor([len(y_train) / (2 * sum(y_train))]).to(
                model.device
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # Setup optimizer
        if self.config.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        # Setup scheduler
        scheduler = None
        if self.config.use_scheduler:
            if self.config.scheduler_type == "onecycle":
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.config.learning_rate,
                    steps_per_epoch=len(train_loader),
                    epochs=self.config.epochs,
                )
            elif self.config.scheduler_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.config.epochs
                )
            elif self.config.scheduler_type == "step":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=30, gamma=self.config.scheduler_factor
                )
            elif self.config.scheduler_type == "plateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=self.config.scheduler_patience,
                    factor=self.config.scheduler_factor,
                )

        # Mixed precision scaler
        scaler = GradScaler() if self.config.use_mixed_precision else None

        # Training loop
        training_metrics = []
        best_val_auc = 0.0
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                if self.config.use_mixed_precision and scaler is not None:
                    with autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)

                        # Add L1/L2 regularization
                        if self.config.l1_lambda > 0:
                            l1_reg = sum(
                                p.abs().sum() for p in model.parameters()
                            )
                            loss += self.config.l1_lambda * l1_reg

                        if self.config.l2_lambda > 0:
                            l2_reg = sum(
                                p.pow(2).sum() for p in model.parameters()
                            )
                            loss += self.config.l2_lambda * l2_reg

                    scaler.scale(loss).backward()

                    # Gradient clipping
                    if self.config.gradient_clip_value > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.gradient_clip_value
                        )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    # Add L1/L2 regularization
                    if self.config.l1_lambda > 0:
                        l1_reg = sum(p.abs().sum() for p in model.parameters())
                        loss += self.config.l1_lambda * l1_reg

                    if self.config.l2_lambda > 0:
                        l2_reg = sum(
                            p.pow(2).sum() for p in model.parameters()
                        )
                        loss += self.config.l2_lambda * l2_reg

                    loss.backward()

                    # Gradient clipping
                    if self.config.gradient_clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.gradient_clip_value
                        )

                    optimizer.step()

                train_loss += loss.item()

                # Update scheduler (for OneCycleLR)
                if scheduler and self.config.scheduler_type == "onecycle":
                    scheduler.step()

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

                val_probs = torch.sigmoid(val_outputs).cpu().numpy()
                val_predictions.extend(val_probs)
                val_targets.extend(y_val_tensor.cpu().numpy())

            # Calculate metrics
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            val_pred_binary = (val_predictions > 0.5).astype(int)

            val_auc = roc_auc_score(val_targets, val_predictions)
            val_f1 = f1_score(val_targets, val_pred_binary, average="weighted")
            val_precision = precision_score(
                val_targets,
                val_pred_binary,
                average="weighted",
                zero_division=0,
            )
            val_recall = recall_score(
                val_targets,
                val_pred_binary,
                average="weighted",
                zero_division=0,
            )

            # Create training metrics
            metrics = TrainingMetrics(
                experiment_id=f"tcn_epoch_{epoch}",
                model_type="tcn",
                epoch=epoch,
                train_loss=train_loss / len(train_loader),
                val_loss=val_loss,
                auc_roc=val_auc,
                f1_score=val_f1,
                precision=val_precision,
                recall=val_recall,
                energy_consumed_kwh=0.0,  # TODO: Integrate with sustainability monitor
                carbon_emissions_kg=0.0,
                training_time_seconds=0.0,
            )
            training_metrics.append(metrics)

            # Update scheduler (except OneCycleLR)
            if scheduler and self.config.scheduler_type != "onecycle":
                if self.config.scheduler_type == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Early stopping and best model saving
            if val_auc > best_val_auc + self.config.min_delta:
                best_val_auc = val_auc
                patience_counter = 0
                if self.config.save_best_only:
                    model.best_state_dict = model.state_dict().copy()
            else:
                patience_counter += 1

            # Log progress
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                logger.info(
                    f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
                )

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model if saved
        if model.best_state_dict is not None:
            model.load_state_dict(model.best_state_dict)

        model.is_trained = True
        model.training_history = training_metrics

        return training_metrics

    def _evaluate_model(
        self, model: TCNModel, X: np.ndarray, y: np.ndarray, dataset_name: str
    ) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        try:
            # Prepare data
            n_samples, n_timesteps, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            X_scaled = model.scaler.transform(X_reshaped).reshape(
                n_samples, n_timesteps, n_features
            )
            X_tensor = torch.FloatTensor(X_scaled).to(model.device)

            # Make predictions
            model.eval()
            with torch.no_grad():
                probs = model.predict_proba(X_tensor)
                predictions = model.predict(X_tensor)

            # Convert to numpy
            probs_np = probs.cpu().numpy()
            predictions_np = predictions.cpu().numpy()

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y, predictions_np),
                "precision": precision_score(
                    y, predictions_np, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y, predictions_np, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    y, predictions_np, average="weighted", zero_division=0
                ),
                "roc_auc": roc_auc_score(y, probs_np[:, 1]),
            }

            logger.info(
                f"{dataset_name} Metrics - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed for {dataset_name}: {e}")
            return {}


# Factory functions and utilities
def create_tcn_model(config: Optional[TCNConfig] = None) -> TCNModel:
    """Create a TCN model instance."""
    return TCNModel(config)


def train_tcn_baseline(
    X: pd.DataFrame, y: pd.Series, config: Optional[TCNConfig] = None
) -> TCNResult:
    """Convenience function to train TCN baseline."""
    trainer = TCNTrainer(config)
    return trainer.train_and_evaluate(X, y)


def get_default_tcn_config() -> TCNConfig:
    """Get default TCN configuration."""
    return TCNConfig()


def get_fast_tcn_config() -> TCNConfig:
    """Get fast TCN configuration for testing."""
    return TCNConfig(
        num_channels=[32, 32],
        epochs=20,
        early_stopping_patience=5,
        batch_size=32,
        use_mixed_precision=False,
        max_sequence_length=20,
    )


def get_optimized_tcn_config() -> TCNConfig:
    """Get optimized TCN configuration for production."""
    return TCNConfig(
        num_channels=[128, 128, 64, 64, 32],
        epochs=200,
        early_stopping_patience=25,
        batch_size=128,
        learning_rate=0.0005,
        use_mixed_precision=True,
        use_scheduler=True,
        scheduler_type="cosine",
        dropout_rate=0.3,
        weight_decay=1e-3,
        kernel_size=5,
        dilation_base=2,
        max_sequence_length=100,
        activation="gelu",
    )
