"""
LSTM (Long Short-Term Memory) network for temporal credit risk prediction.
Handles sequential data like payment histories, spending patterns, and temporal features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import json
from pathlib import Path
import time

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    from ..core.interfaces import BaseModel, TrainingMetrics
    from ..core.logging import get_logger, get_audit_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

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
class LSTMConfig:
    """Configuration for LSTM model."""

    # Architecture parameters
    input_size: int = 14  # Number of features per timestep
    hidden_size: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.3
    bidirectional: bool = True

    # Attention mechanism
    use_attention: bool = True
    attention_dim: int = 64

    # Sequence parameters
    max_sequence_length: int = 50
    min_sequence_length: int = 5

    # Output layers
    output_hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    use_batch_norm: bool = True

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
    model_path: str = "models/lstm"
    save_best_only: bool = True

    # Device configuration
    device: str = "auto"  # 'auto', 'cpu', 'cuda', 'mps'


@dataclass
class LSTMResult:
    """Result of LSTM training and evaluation."""

    success: bool
    model: Optional["LSTMModel"]
    config: LSTMConfig
    training_metrics: List[TrainingMetrics]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    attention_weights: Optional[torch.Tensor]
    training_time_seconds: float
    model_path: Optional[str]
    best_epoch: int
    message: str


class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM outputs."""

    def __init__(self, hidden_size: int, attention_dim: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(
        self, lstm_outputs: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: (batch_size, max_seq_len, hidden_size)
            lengths: (batch_size,) actual sequence lengths

        Returns:
            context: (batch_size, hidden_size) weighted sum of outputs
            attention_weights: (batch_size, max_seq_len) attention weights
        """
        batch_size, max_seq_len, hidden_size = lstm_outputs.shape

        # Calculate attention scores
        attention_scores = self.attention(lstm_outputs)  # (batch_size, max_seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, max_seq_len)

        # Create mask for padding
        mask = torch.arange(max_seq_len, device=lstm_outputs.device).expand(
            batch_size, max_seq_len
        ) < lengths.unsqueeze(1)

        # Apply mask (set padded positions to large negative value)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(
            attention_scores, dim=1
        )  # (batch_size, max_seq_len)

        # Calculate weighted sum
        context = torch.sum(
            lstm_outputs * attention_weights.unsqueeze(-1), dim=1
        )  # (batch_size, hidden_size)

        return context, attention_weights


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LSTMModel(BaseModel):
    """LSTM model for temporal credit risk prediction."""

    def __init__(self, config: Optional[LSTMConfig] = None):
        super(LSTMModel, self).__init__()
        self.config = config or LSTMConfig()
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
        self.last_attention_weights = None

    def _get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)

    def _build_network(self):
        """Build the LSTM network architecture."""
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout_rate if self.config.num_layers > 1 else 0,
            bidirectional=self.config.bidirectional,
            batch_first=True,
        )

        # Calculate LSTM output size
        lstm_output_size = self.config.hidden_size * (
            2 if self.config.bidirectional else 1
        )

        # Attention mechanism
        if self.config.use_attention:
            self.attention = AttentionLayer(lstm_output_size, self.config.attention_dim)
            final_lstm_size = lstm_output_size
        else:
            self.attention = None
            final_lstm_size = lstm_output_size

        # Output layers
        layers = []
        prev_dim = final_lstm_size

        for hidden_dim in self.config.output_hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.config.dropout_rate))

            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.output_layers = nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                # Input-to-hidden weights
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                # Hidden-to-hidden weights
                nn.init.orthogonal_(param)
            elif "bias" in name:
                # Biases
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1 (LSTM best practice)
                if "bias_ih" in name:
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

        # Initialize other layers
        for module in self.output_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: (batch_size, max_seq_len, input_size) padded sequences
            lengths: (batch_size,) actual sequence lengths

        Returns:
            output: (batch_size,) predictions
        """
        batch_size = x.size(0)

        # Pack padded sequences for efficient LSTM processing
        packed_x = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_x)

        # Unpack sequences
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply attention or use last output
        if self.config.use_attention:
            context, attention_weights = self.attention(lstm_output, lengths)
            self.last_attention_weights = attention_weights.detach()
        else:
            # Use the last valid output for each sequence
            idx = (lengths - 1).long().unsqueeze(-1).unsqueeze(-1)
            idx = idx.expand(batch_size, 1, lstm_output.size(-1))
            context = lstm_output.gather(1, idx).squeeze(1)
            self.last_attention_weights = None

        # Pass through output layers
        output = self.output_layers(context)

        return output.squeeze(-1)

    def predict_proba(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x).to(self.device)
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.LongTensor(lengths).to(self.device)

            x = x.to(self.device)
            lengths = lengths.to(self.device)

            logits = self.forward(x, lengths)
            probabilities = torch.sigmoid(logits)

            # Return as 2D array for binary classification
            neg_probs = 1 - probabilities
            return torch.stack([neg_probs, probabilities], dim=1)

    def predict(
        self, x: torch.Tensor, lengths: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Make binary predictions."""
        probs = self.predict_proba(x, lengths)
        return (probs[:, 1] > threshold).long()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance using attention weights and gradient analysis."""
        if not self.is_trained:
            return {}

        feature_importance = {}

        # If attention is used, use attention weights as importance
        if self.config.use_attention and self.last_attention_weights is not None:
            # Average attention weights across batch and time steps
            avg_attention = self.last_attention_weights.mean(dim=0).cpu().numpy()

            # Map attention weights to timesteps, not features
            for i, weight in enumerate(avg_attention):
                feature_name = f"timestep_{i}"
                feature_importance[feature_name] = float(weight)

            # Also add feature-based importance using gradients
            try:
                self.eval()
                dummy_seq_len = min(5, self.config.max_sequence_length)
                dummy_input = torch.randn(
                    1,
                    dummy_seq_len,
                    self.config.input_size,
                    requires_grad=True,
                    device=self.device,
                )
                dummy_lengths = torch.tensor([dummy_seq_len], device=self.device)

                output = self.forward(dummy_input, dummy_lengths)
                gradients = torch.autograd.grad(
                    output, dummy_input, create_graph=False
                )[0]
                importance_scores = torch.abs(gradients).mean(dim=(0, 1)).cpu().numpy()

                for i, score in enumerate(importance_scores):
                    if self.feature_names and i < len(self.feature_names):
                        feature_name = self.feature_names[i]
                        feature_importance[feature_name] = float(score)
            except Exception:
                pass
        else:
            # Use gradient-based importance for input features
            try:
                self.eval()

                # Create dummy input
                dummy_seq_len = min(5, self.config.max_sequence_length)
                dummy_input = torch.randn(
                    1,
                    dummy_seq_len,
                    self.config.input_size,
                    requires_grad=True,
                    device=self.device,
                )
                dummy_lengths = torch.tensor([dummy_seq_len], device=self.device)

                output = self.forward(dummy_input, dummy_lengths)

                # Compute gradients
                gradients = torch.autograd.grad(
                    output, dummy_input, create_graph=False
                )[0]
                importance_scores = torch.abs(gradients).mean(dim=(0, 1)).cpu().numpy()

                for i, score in enumerate(importance_scores):
                    if self.feature_names and i < len(self.feature_names):
                        feature_name = self.feature_names[i]
                    else:
                        feature_name = f"feature_{i}"
                    feature_importance[feature_name] = float(score)
            except Exception as e:
                # Fallback: create dummy importance based on feature names
                if self.feature_names:
                    for i, name in enumerate(self.feature_names):
                        feature_importance[name] = 1.0 / len(self.feature_names)
                else:
                    for i in range(self.config.input_size):
                        feature_importance[f"feature_{i}"] = (
                            1.0 / self.config.input_size
                        )

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return feature_importance

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the last computed attention weights."""
        return self.last_attention_weights

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        # Create save path
        save_path = path or self.config.model_path
        model_dir = Path(save_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_file = model_dir / "lstm_model.pth"
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
                "device": str(self.device),
            },
            model_file,
        )

        # Save metadata
        metadata = {
            "model_type": "lstm",
            "input_size": self.config.input_size,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "bidirectional": self.config.bidirectional,
            "use_attention": self.config.use_attention,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "config": self.config.__dict__,
            "saved_at": datetime.now().isoformat(),
        }

        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {model_file}")
        return str(model_file)

    def load_model(self, path: str) -> "LSTMModel":
        """Load a trained model."""
        model_path = Path(path)

        if model_path.is_file():
            model_file = model_path
        else:
            model_file = model_path / "lstm_model.pth"

        # Load model
        checkpoint = torch.load(
            model_file, map_location=self.device, weights_only=False
        )

        # Restore configuration and architecture
        self.config = checkpoint["config"]
        self.feature_names = checkpoint.get("feature_names")
        self.training_history = checkpoint.get("training_history", [])

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
    """Processor for converting tabular data to sequences for LSTM."""

    def __init__(self, sequence_length: int = 10, overlap: float = 0.5):
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.step_size = max(1, int(sequence_length * (1 - overlap)))

    def create_sequences(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[List[np.ndarray], List[int], np.ndarray]:
        """
        Create sequences from tabular data.

        Args:
            X: Feature dataframe
            y: Target series

        Returns:
            sequences: List of sequences (each sequence is array of shape [seq_len, n_features])
            lengths: List of actual sequence lengths
            targets: Array of targets for each sequence
        """
        sequences = []
        lengths = []
        targets = []

        # Convert to numpy for easier manipulation
        X_np = X.values
        y_np = y.values

        # Create sequences using sliding window
        for i in range(0, len(X_np) - self.sequence_length + 1, self.step_size):
            end_idx = min(i + self.sequence_length, len(X_np))
            seq_len = end_idx - i

            if seq_len >= 3:  # Minimum sequence length
                sequence = X_np[i:end_idx]
                target = y_np[end_idx - 1]  # Use the last target in the sequence

                sequences.append(sequence)
                lengths.append(seq_len)
                targets.append(target)

        return sequences, lengths, np.array(targets)

    def pad_sequences(
        self, sequences: List[np.ndarray], max_length: Optional[int] = None
    ) -> np.ndarray:
        """Pad sequences to the same length."""
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        padded = np.zeros((len(sequences), max_length, sequences[0].shape[1]))

        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_length)
            padded[i, :seq_len] = seq[:seq_len]

        return padded


class LSTMTrainer:
    """Trainer for LSTM models."""

    def __init__(self, config: Optional[LSTMConfig] = None):
        self.config = config or LSTMConfig()
        self.sequence_processor = SequenceDataProcessor(
            sequence_length=self.config.max_sequence_length, overlap=0.3
        )

    def train_and_evaluate(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
    ) -> LSTMResult:
        """Train and evaluate LSTM model."""
        start_time = datetime.now()

        try:
            logger.info("Starting LSTM training and evaluation")

            # Create sequences from tabular data
            sequences, lengths, targets = self.sequence_processor.create_sequences(X, y)

            if len(sequences) == 0:
                raise ValueError("No sequences could be created from the data")

            logger.info(
                f"Created {len(sequences)} sequences with max length {max(lengths)}"
            )

            # Split data
            seq_train, seq_test, len_train, len_test, y_train, y_test = (
                train_test_split(
                    sequences,
                    lengths,
                    targets,
                    test_size=test_size,
                    random_state=42,
                    stratify=targets,
                )
            )

            seq_train, seq_val, len_train, len_val, y_train, y_val = train_test_split(
                seq_train,
                len_train,
                y_train,
                test_size=0.2,
                random_state=42,
                stratify=y_train,
            )

            logger.info(
                f"Data split - Train: {len(seq_train)}, Val: {len(seq_val)}, Test: {len(seq_test)}"
            )

            # Update config with actual input size
            self.config.input_size = sequences[0].shape[1]

            # Create and train model
            model = LSTMModel(self.config)
            model.feature_names = list(X.columns)

            # Train model
            training_metrics = self._train_model(
                model, seq_train, len_train, y_train, seq_val, len_val, y_val
            )

            # Evaluate model
            validation_metrics = self._evaluate_model(
                model, seq_val, len_val, y_val, "Validation"
            )
            test_metrics = self._evaluate_model(
                model, seq_test, len_test, y_test, "Test"
            )

            # Get feature importance
            feature_importance = model.get_feature_importance()

            # Get attention weights
            attention_weights = model.get_attention_weights()

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
                model_id="lstm_baseline",
                operation="training_completed",
                success=True,
                details={
                    "training_time_seconds": training_time,
                    "test_auc": test_metrics.get("roc_auc", 0.0),
                    "best_epoch": best_epoch,
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "num_sequences": len(sequences),
                },
            )

            logger.info(f"LSTM training completed in {training_time:.2f} seconds")

            return LSTMResult(
                success=True,
                model=model,
                config=self.config,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                feature_importance=feature_importance,
                attention_weights=attention_weights,
                training_time_seconds=training_time,
                model_path=model_path,
                best_epoch=best_epoch,
                message="LSTM training completed successfully",
            )

        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            error_message = f"LSTM training failed: {str(e)}"
            logger.error(error_message)

            return LSTMResult(
                success=False,
                model=None,
                config=self.config,
                training_metrics=[],
                validation_metrics={},
                test_metrics={},
                feature_importance={},
                attention_weights=None,
                training_time_seconds=training_time,
                model_path=None,
                best_epoch=0,
                message=error_message,
            )

    def _train_model(
        self,
        model: LSTMModel,
        seq_train: List[np.ndarray],
        len_train: List[int],
        y_train: np.ndarray,
        seq_val: List[np.ndarray],
        len_val: List[int],
        y_val: np.ndarray,
    ) -> List[TrainingMetrics]:
        """Train the LSTM model."""

        # Prepare data
        X_train_padded = self.sequence_processor.pad_sequences(
            seq_train, self.config.max_sequence_length
        )
        X_val_padded = self.sequence_processor.pad_sequences(
            seq_val, self.config.max_sequence_length
        )

        # Scale features
        n_samples, n_timesteps, n_features = X_train_padded.shape
        X_train_reshaped = X_train_padded.reshape(-1, n_features)
        model.scaler.fit(X_train_reshaped)

        X_train_scaled = model.scaler.transform(X_train_reshaped).reshape(
            n_samples, n_timesteps, n_features
        )

        n_val_samples = X_val_padded.shape[0]
        X_val_reshaped = X_val_padded.reshape(-1, n_features)
        X_val_scaled = model.scaler.transform(X_val_reshaped).reshape(
            n_val_samples, n_timesteps, n_features
        )

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(model.device)
        y_train_tensor = torch.FloatTensor(y_train).to(model.device)
        len_train_tensor = torch.LongTensor(len_train).to(model.device)

        X_val_tensor = torch.FloatTensor(X_val_scaled).to(model.device)
        y_val_tensor = torch.FloatTensor(y_val).to(model.device)
        len_val_tensor = torch.LongTensor(len_val).to(model.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, len_train_tensor, y_train_tensor)
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
            if self.config.scheduler_type == "cosine":
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

            for batch_X, batch_lengths, batch_y in train_loader:
                optimizer.zero_grad()

                if self.config.use_mixed_precision and scaler is not None:
                    with autocast():
                        outputs = model(batch_X, batch_lengths)
                        loss = criterion(outputs, batch_y)

                        # Add L1/L2 regularization
                        if self.config.l1_lambda > 0:
                            l1_reg = sum(p.abs().sum() for p in model.parameters())
                            loss += self.config.l1_lambda * l1_reg

                        if self.config.l2_lambda > 0:
                            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
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
                    outputs = model(batch_X, batch_lengths)
                    loss = criterion(outputs, batch_y)

                    # Add L1/L2 regularization
                    if self.config.l1_lambda > 0:
                        l1_reg = sum(p.abs().sum() for p in model.parameters())
                        loss += self.config.l1_lambda * l1_reg

                    if self.config.l2_lambda > 0:
                        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
                        loss += self.config.l2_lambda * l2_reg

                    loss.backward()

                    # Gradient clipping
                    if self.config.gradient_clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.gradient_clip_value
                        )

                    optimizer.step()

                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                val_outputs = model(X_val_tensor, len_val_tensor)
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
                val_targets, val_pred_binary, average="weighted", zero_division=0
            )
            val_recall = recall_score(
                val_targets, val_pred_binary, average="weighted", zero_division=0
            )

            # Create training metrics
            metrics = TrainingMetrics(
                experiment_id=f"lstm_epoch_{epoch}",
                model_type="lstm",
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

            # Update scheduler (except plateau which needs validation loss)
            if scheduler and self.config.scheduler_type != "plateau":
                scheduler.step()
            elif scheduler and self.config.scheduler_type == "plateau":
                scheduler.step(val_loss)

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
        self,
        model: LSTMModel,
        sequences: List[np.ndarray],
        lengths: List[int],
        targets: np.ndarray,
        dataset_name: str,
    ) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        try:
            # Prepare data
            X_padded = self.sequence_processor.pad_sequences(
                sequences, self.config.max_sequence_length
            )

            # Scale features
            n_samples, n_timesteps, n_features = X_padded.shape
            X_reshaped = X_padded.reshape(-1, n_features)
            X_scaled = model.scaler.transform(X_reshaped).reshape(
                n_samples, n_timesteps, n_features
            )

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled).to(model.device)
            lengths_tensor = torch.LongTensor(lengths).to(model.device)

            # Make predictions
            model.eval()
            with torch.no_grad():
                probs = model.predict_proba(X_tensor, lengths_tensor)
                predictions = model.predict(X_tensor, lengths_tensor)

            # Convert to numpy
            probs_np = probs.cpu().numpy()
            predictions_np = predictions.cpu().numpy()

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(targets, predictions_np),
                "precision": precision_score(
                    targets, predictions_np, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    targets, predictions_np, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    targets, predictions_np, average="weighted", zero_division=0
                ),
                "roc_auc": roc_auc_score(targets, probs_np[:, 1]),
            }

            logger.info(
                f"{dataset_name} Metrics - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed for {dataset_name}: {e}")
            return {}


# Factory functions and utilities
def create_lstm_model(config: Optional[LSTMConfig] = None) -> LSTMModel:
    """Create an LSTM model instance."""
    return LSTMModel(config)


def train_lstm_baseline(
    X: pd.DataFrame, y: pd.Series, config: Optional[LSTMConfig] = None
) -> LSTMResult:
    """Convenience function to train LSTM baseline."""
    trainer = LSTMTrainer(config)
    return trainer.train_and_evaluate(X, y)


def get_default_lstm_config() -> LSTMConfig:
    """Get default LSTM configuration."""
    return LSTMConfig()


def get_fast_lstm_config() -> LSTMConfig:
    """Get fast LSTM configuration for testing."""
    return LSTMConfig(
        hidden_size=64,
        num_layers=1,
        max_sequence_length=20,
        epochs=20,
        early_stopping_patience=5,
        batch_size=32,
        use_mixed_precision=False,
    )


def get_optimized_lstm_config() -> LSTMConfig:
    """Get optimized LSTM configuration for production."""
    return LSTMConfig(
        hidden_size=256,
        num_layers=3,
        max_sequence_length=100,
        epochs=200,
        early_stopping_patience=25,
        batch_size=128,
        learning_rate=0.0005,
        use_mixed_precision=True,
        use_scheduler=True,
        scheduler_type="cosine",
        dropout_rate=0.4,
        weight_decay=1e-3,
        use_attention=True,
        attention_dim=128,
    )
