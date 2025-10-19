"""
Model pruning pipeline for neural network compression.
Implements magnitude-based weight pruning, structured pruning for neurons and channels,
and iterative pruning during fine-tuning with impact measurement and validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import copy
import warnings
from abc import ABC, abstractmethod

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

try:
    from ..models.dnn_model import DNNModel, DNNTrainer, DNNConfig
    from ..core.interfaces import BaseModel, TrainingMetrics
    from ..core.logging import get_logger, get_audit_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from models.dnn_model import DNNModel, DNNTrainer, DNNConfig
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
class PruningConfig:
    """Configuration for model pruning."""

    # Pruning strategy
    pruning_method: str = "magnitude"  # 'magnitude', 'structured', 'gradual'

    # Magnitude-based pruning
    sparsity_level: float = 0.5  # Target sparsity (0.0 to 1.0)
    magnitude_threshold: Optional[float] = None  # Absolute threshold for pruning

    # Structured pruning
    structured_type: str = "neuron"  # 'neuron', 'channel', 'filter'
    structured_ratio: float = 0.3  # Ratio of structures to prune
    importance_metric: str = "l2_norm"  # 'l2_norm', 'l1_norm', 'gradient'

    # Gradual pruning
    initial_sparsity: float = 0.0
    final_sparsity: float = 0.8
    pruning_frequency: int = 10  # Prune every N epochs
    pruning_schedule: str = "polynomial"  # 'linear', 'polynomial', 'exponential'

    # Fine-tuning
    fine_tune_epochs: int = 20
    fine_tune_lr: float = 0.0001
    recovery_epochs: int = 5  # Epochs to recover after each pruning step

    # Layer selection
    layers_to_prune: Optional[List[str]] = None  # None means all eligible layers
    exclude_layers: List[str] = field(default_factory=list)  # Layers to exclude

    # Validation and impact measurement
    validate_pruning: bool = True
    accuracy_threshold: float = 0.02  # Max acceptable accuracy drop
    measure_flops: bool = True
    measure_memory: bool = True

    # Iterative pruning
    iterative_steps: int = 5
    iterative_recovery: bool = True

    # Results storage
    save_pruned_model: bool = True
    pruned_model_path: str = "models/pruned"


@dataclass
class PruningResult:
    """Result of model pruning operation."""

    success: bool
    original_model: Optional[nn.Module]
    pruned_model: Optional[nn.Module]
    config: PruningConfig

    # Compression metrics
    original_params: int = 0
    pruned_params: int = 0
    compression_ratio: float = 0.0
    sparsity_achieved: float = 0.0

    # Performance metrics
    original_performance: Dict[str, float] = field(default_factory=dict)
    pruned_performance: Dict[str, float] = field(default_factory=dict)
    performance_drop: Dict[str, float] = field(default_factory=dict)

    # Resource metrics
    original_flops: int = 0
    pruned_flops: int = 0
    flops_reduction: float = 0.0
    original_memory_mb: float = 0.0
    pruned_memory_mb: float = 0.0
    memory_reduction: float = 0.0

    # Pruning details
    layers_pruned: List[str] = field(default_factory=list)
    pruning_ratios: Dict[str, float] = field(default_factory=dict)
    pruning_time_seconds: float = 0.0
    fine_tuning_time_seconds: float = 0.0

    # Model paths
    pruned_model_path: Optional[str] = None

    message: str = ""


class BasePruner(ABC):
    """Abstract base class for pruning methods."""

    @abstractmethod
    def prune_model(self, model: nn.Module, config: PruningConfig) -> nn.Module:
        """Prune the model according to the configuration."""
        pass

    @abstractmethod
    def calculate_importance(self, layer: nn.Module) -> torch.Tensor:
        """Calculate importance scores for pruning decisions."""
        pass


class MagnitudePruner(BasePruner):
    """Magnitude-based weight pruning."""

    def prune_model(self, model: nn.Module, config: PruningConfig) -> nn.Module:
        """Prune model based on weight magnitudes."""
        pruned_model = copy.deepcopy(model)

        # Collect all weights
        all_weights = []
        weight_info = []

        for name, module in pruned_model.named_modules():
            if self._should_prune_layer(name, module, config):
                if hasattr(module, "weight") and module.weight is not None:
                    weights = module.weight.data.abs().flatten()
                    all_weights.append(weights)
                    weight_info.append((name, module, weights.shape[0]))

        if not all_weights:
            logger.warning("No weights found for pruning")
            return pruned_model

        # Calculate global threshold
        all_weights_tensor = torch.cat(all_weights)

        if config.magnitude_threshold is not None:
            threshold = config.magnitude_threshold
        else:
            # Use percentile-based threshold
            threshold = torch.quantile(all_weights_tensor, config.sparsity_level)

        logger.info(f"Pruning threshold: {threshold:.6f}")

        # Apply pruning
        total_params = 0
        pruned_params = 0

        for name, module, _ in weight_info:
            if hasattr(module, "weight") and module.weight is not None:
                weight_mask = module.weight.data.abs() > threshold

                # Count parameters
                total_params += module.weight.numel()
                pruned_params += weight_mask.sum().item()

                # Apply mask
                module.weight.data *= weight_mask.float()

                # Store mask for future use
                module.register_buffer("weight_mask", weight_mask)

        actual_sparsity = (
            1.0 - (pruned_params / total_params) if total_params > 0 else 0.0
        )
        logger.info(f"Achieved sparsity: {actual_sparsity:.4f}")

        return pruned_model

    def calculate_importance(self, layer: nn.Module) -> torch.Tensor:
        """Calculate importance based on weight magnitudes."""
        if hasattr(layer, "weight") and layer.weight is not None:
            return layer.weight.data.abs()
        return torch.tensor([])

    def _should_prune_layer(
        self, name: str, module: nn.Module, config: PruningConfig
    ) -> bool:
        """Check if layer should be pruned."""
        # Skip excluded layers
        if any(exclude in name for exclude in config.exclude_layers):
            return False

        # Only prune specific layers if specified
        if config.layers_to_prune is not None:
            if not any(layer_name in name for layer_name in config.layers_to_prune):
                return False

        # Only prune linear and conv layers
        return isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d))


class StructuredPruner(BasePruner):
    """Structured pruning for neurons, channels, or filters."""

    def prune_model(self, model: nn.Module, config: PruningConfig) -> nn.Module:
        """Prune model using structured pruning."""
        pruned_model = copy.deepcopy(model)

        if config.structured_type == "neuron":
            return self._prune_neurons(pruned_model, config)
        elif config.structured_type == "channel":
            return self._prune_channels(pruned_model, config)
        else:
            logger.warning(f"Structured type {config.structured_type} not implemented")
            return pruned_model

    def _prune_neurons(self, model: nn.Module, config: PruningConfig) -> nn.Module:
        """Prune entire neurons from linear layers."""
        layers_to_modify = []

        # Find linear layers to prune
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and self._should_prune_layer(
                name, module, config
            ):
                layers_to_modify.append((name, module))

        # Prune neurons
        for name, layer in layers_to_modify:
            importance_scores = self.calculate_importance(layer)

            if importance_scores.numel() == 0:
                continue

            # Calculate number of neurons to prune
            num_neurons = layer.out_features
            num_to_prune = int(num_neurons * config.structured_ratio)

            if num_to_prune == 0:
                continue

            # Find least important neurons
            if config.importance_metric == "l2_norm":
                neuron_importance = torch.norm(layer.weight.data, dim=1)
            elif config.importance_metric == "l1_norm":
                neuron_importance = torch.norm(layer.weight.data, dim=1, p=1)
            else:
                neuron_importance = torch.norm(layer.weight.data, dim=1)

            _, indices_to_prune = torch.topk(
                neuron_importance, num_to_prune, largest=False
            )

            # Create mask
            keep_mask = torch.ones(num_neurons, dtype=torch.bool)
            keep_mask[indices_to_prune] = False

            # Apply pruning by zeroing out weights
            layer.weight.data[indices_to_prune] = 0
            if layer.bias is not None:
                layer.bias.data[indices_to_prune] = 0

            # Store mask
            layer.register_buffer("neuron_mask", keep_mask)

            logger.info(f"Pruned {num_to_prune}/{num_neurons} neurons from {name}")

        return model

    def _prune_channels(self, model: nn.Module, config: PruningConfig) -> nn.Module:
        """Prune channels from convolutional layers."""
        # This is a simplified implementation
        # In practice, channel pruning requires careful handling of dependencies
        logger.warning("Channel pruning not fully implemented")
        return model

    def calculate_importance(self, layer: nn.Module) -> torch.Tensor:
        """Calculate importance scores for structured elements."""
        if isinstance(layer, nn.Linear):
            # For neurons, use L2 norm of outgoing weights
            return torch.norm(layer.weight.data, dim=1)
        elif isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            # For channels, use L2 norm of filters
            return torch.norm(layer.weight.data.view(layer.weight.size(0), -1), dim=1)
        return torch.tensor([])

    def _should_prune_layer(
        self, name: str, module: nn.Module, config: PruningConfig
    ) -> bool:
        """Check if layer should be pruned."""
        if any(exclude in name for exclude in config.exclude_layers):
            return False

        if config.layers_to_prune is not None:
            if not any(layer_name in name for layer_name in config.layers_to_prune):
                return False

        return isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d))


class GradualPruner(BasePruner):
    """Gradual pruning during training."""

    def __init__(self):
        self.current_sparsity = 0.0
        self.pruning_step = 0

    def prune_model(self, model: nn.Module, config: PruningConfig) -> nn.Module:
        """Apply gradual pruning (this is called during training)."""
        # This method is called during training to gradually increase sparsity
        target_sparsity = self._calculate_target_sparsity(config)

        if target_sparsity > self.current_sparsity:
            # Apply magnitude pruning with current target sparsity
            magnitude_config = copy.deepcopy(config)
            magnitude_config.sparsity_level = target_sparsity

            pruner = MagnitudePruner()
            pruned_model = pruner.prune_model(model, magnitude_config)

            self.current_sparsity = target_sparsity
            self.pruning_step += 1

            logger.info(
                f"Gradual pruning step {self.pruning_step}: sparsity = {target_sparsity:.4f}"
            )

            return pruned_model

        return model

    def _calculate_target_sparsity(self, config: PruningConfig) -> float:
        """Calculate target sparsity for current step."""
        if config.pruning_schedule == "linear":
            progress = self.pruning_step / config.iterative_steps
            return config.initial_sparsity + progress * (
                config.final_sparsity - config.initial_sparsity
            )
        elif config.pruning_schedule == "polynomial":
            progress = self.pruning_step / config.iterative_steps
            return config.initial_sparsity + (
                config.final_sparsity - config.initial_sparsity
            ) * (progress**3)
        else:
            # Exponential
            progress = self.pruning_step / config.iterative_steps
            return config.initial_sparsity + (
                config.final_sparsity - config.initial_sparsity
            ) * (1 - np.exp(-3 * progress))

    def calculate_importance(self, layer: nn.Module) -> torch.Tensor:
        """Calculate importance based on magnitudes."""
        if hasattr(layer, "weight") and layer.weight is not None:
            return layer.weight.data.abs()
        return torch.tensor([])


class ModelPruner:
    """Main model pruning class."""

    def __init__(self, config: Optional[PruningConfig] = None):
        self.config = config or PruningConfig()

        # Initialize pruner based on method
        if self.config.pruning_method == "magnitude":
            self.pruner = MagnitudePruner()
        elif self.config.pruning_method == "structured":
            self.pruner = StructuredPruner()
        elif self.config.pruning_method == "gradual":
            self.pruner = GradualPruner()
        else:
            raise ValueError(f"Unknown pruning method: {self.config.pruning_method}")

    def prune_and_fine_tune(
        self, model: nn.Module, X: pd.DataFrame, y: pd.Series
    ) -> PruningResult:
        """
        Prune model and fine-tune with validation.

        Args:
            model: Model to prune
            X: Training features
            y: Training targets

        Returns:
            PruningResult with pruning results
        """
        start_time = datetime.now()

        try:
            logger.info("Starting model pruning and fine-tuning")

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Evaluate original model
            original_performance = self._evaluate_model(model, X_val, y_val)
            original_stats = self._calculate_model_stats(model)

            logger.info(
                f"Original model - AUC: {original_performance.get('roc_auc', 0.0):.4f}, "
                f"Params: {original_stats['params']}"
            )

            # Perform pruning
            pruning_start = datetime.now()

            if self.config.pruning_method == "gradual":
                pruned_model = self._gradual_prune_and_fine_tune(
                    model, X_train, y_train, X_val, y_val
                )
            else:
                # Apply pruning
                pruned_model = self.pruner.prune_model(model, self.config)

                # Fine-tune pruned model
                if self.config.fine_tune_epochs > 0:
                    pruned_model = self._fine_tune_model(
                        pruned_model, X_train, y_train, X_val, y_val
                    )

            pruning_time = (datetime.now() - pruning_start).total_seconds()

            # Evaluate pruned model
            pruned_performance = self._evaluate_model(pruned_model, X_val, y_val)
            pruned_stats = self._calculate_model_stats(pruned_model)

            # Calculate metrics
            compression_ratio = (
                original_stats["params"] / pruned_stats["params"]
                if pruned_stats["params"] > 0
                else 1.0
            )
            sparsity_achieved = self._calculate_sparsity(pruned_model)

            performance_drop = {
                key: original_performance[key] - pruned_performance.get(key, 0.0)
                for key in original_performance.keys()
            }

            # Check if pruning meets criteria
            auc_drop = performance_drop.get("roc_auc", 0.0)
            success = auc_drop <= self.config.accuracy_threshold

            if not success:
                logger.warning(
                    f"Pruning failed: AUC drop ({auc_drop:.4f}) exceeds threshold ({self.config.accuracy_threshold})"
                )

            # Save pruned model
            pruned_model_path = None
            if self.config.save_pruned_model and success:
                pruned_model_path = self._save_pruned_model(pruned_model)

            total_time = (datetime.now() - start_time).total_seconds()

            # Log completion
            audit_logger.log_model_operation(
                user_id="system",
                model_id="model_pruning",
                operation="pruning_completed",
                success=success,
                details={
                    "pruning_time_seconds": total_time,
                    "compression_ratio": compression_ratio,
                    "sparsity_achieved": sparsity_achieved,
                    "auc_drop": auc_drop,
                    "pruning_method": self.config.pruning_method,
                },
            )

            logger.info(f"Pruning completed in {total_time:.2f} seconds")
            logger.info(f"Compression ratio: {compression_ratio:.2f}x")
            logger.info(f"Sparsity achieved: {sparsity_achieved:.4f}")
            logger.info(f"AUC drop: {auc_drop:.4f}")

            return PruningResult(
                success=success,
                original_model=model,
                pruned_model=pruned_model,
                config=self.config,
                original_params=original_stats["params"],
                pruned_params=pruned_stats["params"],
                compression_ratio=compression_ratio,
                sparsity_achieved=sparsity_achieved,
                original_performance=original_performance,
                pruned_performance=pruned_performance,
                performance_drop=performance_drop,
                original_flops=original_stats.get("flops", 0),
                pruned_flops=pruned_stats.get("flops", 0),
                flops_reduction=1.0
                - (pruned_stats.get("flops", 0) / original_stats.get("flops", 1)),
                original_memory_mb=original_stats.get("memory_mb", 0.0),
                pruned_memory_mb=pruned_stats.get("memory_mb", 0.0),
                memory_reduction=1.0
                - (
                    pruned_stats.get("memory_mb", 0.0)
                    / original_stats.get("memory_mb", 1.0)
                ),
                layers_pruned=self._get_pruned_layers(pruned_model),
                pruning_ratios=self._get_pruning_ratios(pruned_model),
                pruning_time_seconds=pruning_time,
                fine_tuning_time_seconds=total_time - pruning_time,
                pruned_model_path=pruned_model_path,
                message=(
                    "Pruning completed successfully"
                    if success
                    else f"Pruning failed: AUC drop too large ({auc_drop:.4f})"
                ),
            )

        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Pruning failed: {str(e)}"
            logger.error(error_message)

            return PruningResult(
                success=False,
                original_model=model,
                pruned_model=None,
                config=self.config,
                pruning_time_seconds=total_time,
                message=error_message,
            )

    def _gradual_prune_and_fine_tune(
        self,
        model: nn.Module,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> nn.Module:
        """Perform gradual pruning with fine-tuning."""
        current_model = copy.deepcopy(model)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(y_val.values)

        # Setup training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(current_model.parameters(), lr=self.config.fine_tune_lr)

        # Gradual pruning loop
        for step in range(self.config.iterative_steps):
            logger.info(
                f"Gradual pruning step {step + 1}/{self.config.iterative_steps}"
            )

            # Apply pruning
            current_model = self.pruner.prune_model(current_model, self.config)

            # Fine-tune for recovery
            if self.config.iterative_recovery:
                current_model = self._fine_tune_model_tensors(
                    current_model,
                    X_train_tensor,
                    y_train_tensor,
                    X_val_tensor,
                    y_val_tensor,
                    self.config.recovery_epochs,
                )

        return current_model

    def _fine_tune_model(
        self,
        model: nn.Module,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> nn.Module:
        """Fine-tune pruned model."""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(y_val.values)

        return self._fine_tune_model_tensors(
            model,
            X_train_tensor,
            y_train_tensor,
            X_val_tensor,
            y_val_tensor,
            self.config.fine_tune_epochs,
        )

    def _fine_tune_model_tensors(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
    ) -> nn.Module:
        """Fine-tune model with tensor inputs."""
        model.train()

        # Setup training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.fine_tune_lr)

        best_val_auc = 0.0
        best_model_state = None

        for epoch in range(epochs):
            # Training
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            loss.backward()

            # Apply pruning masks during backprop
            self._apply_pruning_masks(model)

            optimizer.step()

            # Validation
            if epoch % 5 == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_probs = torch.sigmoid(val_outputs.squeeze()).numpy()
                    val_auc = roc_auc_score(y_val.numpy(), val_probs)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = copy.deepcopy(model.state_dict())

                model.train()

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model

    def _apply_pruning_masks(self, model: nn.Module):
        """Apply pruning masks to maintain sparsity during training."""
        for module in model.modules():
            if hasattr(module, "weight_mask"):
                module.weight.data *= module.weight_mask.float()
            if hasattr(module, "neuron_mask"):
                # For structured pruning, zero out pruned neurons
                mask = module.neuron_mask
                if module.weight.size(0) == len(mask):
                    module.weight.data[~mask] = 0
                    if module.bias is not None:
                        module.bias.data[~mask] = 0

    def _evaluate_model(
        self, model: nn.Module, X_val: pd.DataFrame, y_val: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            model.eval()
            X_tensor = torch.FloatTensor(X_val.values)

            with torch.no_grad():
                outputs = model(X_tensor)
                probs = torch.sigmoid(outputs.squeeze()).numpy()
                preds = (probs > 0.5).astype(int)

            metrics = {
                "accuracy": accuracy_score(y_val.values, preds),
                "roc_auc": roc_auc_score(y_val.values, probs),
                "f1_score": f1_score(y_val.values, preds, average="weighted"),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"accuracy": 0.0, "roc_auc": 0.0, "f1_score": 0.0}

    def _calculate_model_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Calculate model statistics."""
        total_params = sum(p.numel() for p in model.parameters())

        # Estimate memory usage
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        memory_mb = (param_size + buffer_size) / (1024 * 1024)

        # Estimate FLOPs (simplified)
        flops = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                flops += module.in_features * module.out_features
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                # Simplified FLOP calculation
                flops += (
                    module.in_channels
                    * module.out_channels
                    * np.prod(module.kernel_size)
                )

        return {"params": total_params, "memory_mb": memory_mb, "flops": flops}

    def _calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate overall sparsity of the model."""
        total_params = 0
        zero_params = 0

        for module in model.modules():
            if hasattr(module, "weight") and module.weight is not None:
                weight_data = module.weight.data
                total_params += weight_data.numel()
                zero_params += (weight_data == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0

    def _get_pruned_layers(self, model: nn.Module) -> List[str]:
        """Get list of layers that were pruned."""
        pruned_layers = []

        for name, module in model.named_modules():
            if hasattr(module, "weight_mask") or hasattr(module, "neuron_mask"):
                pruned_layers.append(name)

        return pruned_layers

    def _get_pruning_ratios(self, model: nn.Module) -> Dict[str, float]:
        """Get pruning ratios for each layer."""
        ratios = {}

        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                total_weights = module.weight.numel()
                zero_weights = (module.weight.data == 0).sum().item()
                ratio = zero_weights / total_weights if total_weights > 0 else 0.0

                if ratio > 0:
                    ratios[name] = ratio

        return ratios

    def _save_pruned_model(self, model: nn.Module) -> str:
        """Save pruned model to disk."""
        save_path = Path(self.config.pruned_model_path)
        save_path.mkdir(parents=True, exist_ok=True)

        model_file = save_path / "pruned_model.pth"

        # Save model state
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": self.config,
                "sparsity": self._calculate_sparsity(model),
                "pruned_layers": self._get_pruned_layers(model),
                "pruning_ratios": self._get_pruning_ratios(model),
                "saved_at": datetime.now().isoformat(),
            },
            model_file,
        )

        logger.info(f"Pruned model saved to {model_file}")
        return str(model_file)


# Utility functions
def prune_model(
    model: nn.Module,
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[PruningConfig] = None,
) -> PruningResult:
    """
    Convenience function to prune a model.

    Args:
        model: Model to prune
        X: Training features
        y: Training targets
        config: Pruning configuration

    Returns:
        PruningResult with pruning results
    """
    pruner = ModelPruner(config)
    return pruner.prune_and_fine_tune(model, X, y)


def get_default_pruning_config() -> PruningConfig:
    """Get default pruning configuration."""
    return PruningConfig()


def get_magnitude_pruning_config(sparsity: float = 0.5) -> PruningConfig:
    """Get magnitude-based pruning configuration."""
    return PruningConfig(
        pruning_method="magnitude",
        sparsity_level=sparsity,
        fine_tune_epochs=20,
        validate_pruning=True,
    )


def get_structured_pruning_config(ratio: float = 0.3) -> PruningConfig:
    """Get structured pruning configuration."""
    return PruningConfig(
        pruning_method="structured",
        structured_type="neuron",
        structured_ratio=ratio,
        importance_metric="l2_norm",
        fine_tune_epochs=30,
        validate_pruning=True,
    )


def get_gradual_pruning_config(final_sparsity: float = 0.8) -> PruningConfig:
    """Get gradual pruning configuration."""
    return PruningConfig(
        pruning_method="gradual",
        initial_sparsity=0.0,
        final_sparsity=final_sparsity,
        iterative_steps=5,
        pruning_schedule="polynomial",
        recovery_epochs=5,
        fine_tune_epochs=10,
        validate_pruning=True,
    )


def analyze_pruning_impact(
    original_model: nn.Module,
    pruned_model: nn.Module,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Analyze the impact of pruning on model performance and efficiency.

    Args:
        original_model: Original unpruned model
        pruned_model: Pruned model
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary with analysis results
    """
    # Evaluate both models
    original_perf = _evaluate_model_simple(original_model, X_test, y_test)
    pruned_perf = _evaluate_model_simple(pruned_model, X_test, y_test)

    # Calculate model statistics
    original_stats = _calculate_model_stats_simple(original_model)
    pruned_stats = _calculate_model_stats_simple(pruned_model)

    # Calculate sparsity
    sparsity = _calculate_sparsity_simple(pruned_model)

    # Compression metrics
    compression_ratio = (
        original_stats["params"] / pruned_stats["params"]
        if pruned_stats["params"] > 0
        else 1.0
    )
    memory_reduction = (
        1.0 - (pruned_stats["memory_mb"] / original_stats["memory_mb"])
        if original_stats["memory_mb"] > 0
        else 0.0
    )

    # Performance impact
    performance_drop = {
        key: original_perf[key] - pruned_perf.get(key, 0.0)
        for key in original_perf.keys()
    }

    analysis = {
        "compression_metrics": {
            "compression_ratio": compression_ratio,
            "sparsity": sparsity,
            "parameter_reduction": 1.0
            - (pruned_stats["params"] / original_stats["params"]),
            "memory_reduction": memory_reduction,
        },
        "performance_metrics": {
            "original_performance": original_perf,
            "pruned_performance": pruned_perf,
            "performance_drop": performance_drop,
        },
        "model_statistics": {
            "original_params": original_stats["params"],
            "pruned_params": pruned_stats["params"],
            "original_memory_mb": original_stats["memory_mb"],
            "pruned_memory_mb": pruned_stats["memory_mb"],
        },
    }

    return analysis


def load_pruned_model(
    model_path: str, model_class: type
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a pruned model from disk.

    Args:
        model_path: Path to saved pruned model
        model_class: Class of the model to instantiate

    Returns:
        Tuple of (loaded_model, metadata)
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # This is a simplified loader - in practice, you'd need to reconstruct the model architecture
    # based on the saved configuration and apply the pruning masks

    metadata = {
        "config": checkpoint.get("config"),
        "sparsity": checkpoint.get("sparsity", 0.0),
        "pruned_layers": checkpoint.get("pruned_layers", []),
        "pruning_ratios": checkpoint.get("pruning_ratios", {}),
        "saved_at": checkpoint.get("saved_at"),
    }

    logger.info(f"Loaded pruned model with sparsity: {metadata['sparsity']:.4f}")

    # Note: This is a placeholder - actual implementation would need to reconstruct the model
    # and apply pruning masks based on the saved configuration
    return None, metadata


def compare_pruning_methods(
    model: nn.Module,
    X: pd.DataFrame,
    y: pd.Series,
    sparsity_levels: List[float] = [0.3, 0.5, 0.7, 0.9],
) -> Dict[str, List[PruningResult]]:
    """
    Compare different pruning methods across multiple sparsity levels.

    Args:
        model: Model to prune
        X: Training features
        y: Training targets
        sparsity_levels: List of sparsity levels to test

    Returns:
        Dictionary mapping method names to lists of results
    """
    results = {"magnitude": [], "structured": [], "gradual": []}

    for sparsity in sparsity_levels:
        logger.info(f"Testing sparsity level: {sparsity}")

        # Magnitude pruning
        mag_config = get_magnitude_pruning_config(sparsity)
        mag_result = prune_model(copy.deepcopy(model), X, y, mag_config)
        results["magnitude"].append(mag_result)

        # Structured pruning
        struct_config = get_structured_pruning_config(
            sparsity * 0.5
        )  # Lower ratio for structured
        struct_result = prune_model(copy.deepcopy(model), X, y, struct_config)
        results["structured"].append(struct_result)

        # Gradual pruning
        grad_config = get_gradual_pruning_config(sparsity)
        grad_result = prune_model(copy.deepcopy(model), X, y, grad_config)
        results["gradual"].append(grad_result)

    return results


# Helper functions
def _evaluate_model_simple(
    model: nn.Module, X: pd.DataFrame, y: pd.Series
) -> Dict[str, float]:
    """Simple model evaluation."""
    try:
        model.eval()
        X_tensor = torch.FloatTensor(X.values)

        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.sigmoid(outputs.squeeze()).numpy()
            preds = (probs > 0.5).astype(int)

        return {
            "accuracy": accuracy_score(y.values, preds),
            "roc_auc": roc_auc_score(y.values, probs),
            "f1_score": f1_score(y.values, preds, average="weighted"),
        }
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return {"accuracy": 0.0, "roc_auc": 0.0, "f1_score": 0.0}


def _calculate_model_stats_simple(model: nn.Module) -> Dict[str, Any]:
    """Simple model statistics calculation."""
    total_params = sum(p.numel() for p in model.parameters())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    memory_mb = param_size / (1024 * 1024)

    return {"params": total_params, "memory_mb": memory_mb}


def _calculate_sparsity_simple(model: nn.Module) -> float:
    """Simple sparsity calculation."""
    total_params = 0
    zero_params = 0

    for module in model.modules():
        if hasattr(module, "weight") and module.weight is not None:
            weight_data = module.weight.data
            total_params += weight_data.numel()
            zero_params += (weight_data == 0).sum().item()

    return zero_params / total_params if total_params > 0 else 0.0
