"""
Knowledge distillation framework for model compression.
Implements teacher-student training pipeline with temperature-scaled softmax,
compressed model generation and validation, and distillation loss optimization.
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
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ML imports
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

try:
    from ..core.interfaces import BaseModel, TrainingMetrics
    from ..core.logging import get_audit_logger, get_logger
    from ..models.dnn_model import DNNConfig, DNNModel, DNNTrainer
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.interfaces import BaseModel, TrainingMetrics
    from core.logging import get_audit_logger, get_logger
    from models.dnn_model import DNNConfig, DNNModel, DNNTrainer

    # Create minimal implementations for testing
    class MockAuditLogger:
        def log_model_operation(self, **kwargs):
            pass

    def get_audit_logger():
        return MockAuditLogger()


logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    # Temperature scaling
    temperature: float = 4.0  # Temperature for softmax scaling

    # Loss configuration
    distillation_loss_weight: float = 0.7  # Weight for distillation loss
    student_loss_weight: float = 0.3  # Weight for student's own loss
    loss_function: str = "kl_divergence"  # 'kl_divergence', 'mse', 'cosine'

    # Training parameters
    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 64
    early_stopping_patience: int = 10
    min_delta: float = 1e-4

    # Optimization
    optimizer: str = "adam"  # 'adam', 'adamw', 'sgd'
    weight_decay: float = 1e-4
    gradient_clip_value: float = 1.0
    use_mixed_precision: bool = True

    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # 'cosine', 'step', 'plateau'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Feature matching (optional)
    use_feature_matching: bool = False
    feature_matching_weight: float = 0.1
    feature_matching_layers: List[str] = field(default_factory=list)

    # Attention transfer (optional)
    use_attention_transfer: bool = False
    attention_transfer_weight: float = 0.1

    # Progressive distillation
    use_progressive_distillation: bool = False
    progressive_stages: List[float] = field(
        default_factory=lambda: [0.8, 0.6, 0.4]
    )  # Temperature stages

    # Validation
    validate_distillation: bool = True
    accuracy_threshold: float = 0.05  # Max acceptable accuracy drop from teacher

    # Model saving
    save_student_model: bool = True
    student_model_path: str = "models/distilled"


@dataclass
class DistillationResult:
    """Result of knowledge distillation."""

    success: bool
    teacher_model: Optional[nn.Module]
    student_model: Optional[nn.Module]
    config: DistillationConfig

    # Compression metrics
    teacher_params: int = 0
    student_params: int = 0
    compression_ratio: float = 0.0
    parameter_reduction: float = 0.0

    # Performance metrics
    teacher_performance: Dict[str, float] = field(default_factory=dict)
    student_performance: Dict[str, float] = field(default_factory=dict)
    performance_drop: Dict[str, float] = field(default_factory=dict)

    # Training metrics
    training_history: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    distillation_time_seconds: float = 0.0

    # Model paths
    student_model_path: Optional[str] = None

    message: str = ""


class DistillationLoss(nn.Module):
    """Knowledge distillation loss functions."""

    def __init__(self, config: DistillationConfig):
        super(DistillationLoss, self).__init__()
        self.config = config
        self.temperature = config.temperature
        self.alpha = config.distillation_loss_weight
        self.beta = config.student_loss_weight

        # Student loss (standard task loss)
        self.student_criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate distillation loss.

        Args:
            student_logits: Student model outputs (before sigmoid)
            teacher_logits: Teacher model outputs (before sigmoid)
            targets: Ground truth targets

        Returns:
            Total loss and loss components
        """
        # Student loss (standard task loss)
        student_loss = self.student_criterion(student_logits.squeeze(), targets)

        # Distillation loss
        if self.config.loss_function == "kl_divergence":
            distillation_loss = self._kl_divergence_loss(student_logits, teacher_logits)
        elif self.config.loss_function == "mse":
            distillation_loss = self._mse_loss(student_logits, teacher_logits)
        elif self.config.loss_function == "cosine":
            distillation_loss = self._cosine_loss(student_logits, teacher_logits)
        else:
            distillation_loss = self._kl_divergence_loss(student_logits, teacher_logits)

        # Combined loss
        total_loss = self.alpha * distillation_loss + self.beta * student_loss

        loss_components = {
            "total_loss": total_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "student_loss": student_loss.item(),
        }

        return total_loss, loss_components

    def _kl_divergence_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence loss with temperature scaling."""
        # Temperature scaling
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        # For binary classification, we need to handle the single output case
        if student_logits.dim() == 1 or student_logits.size(-1) == 1:
            # Convert to binary probabilities
            student_probs = torch.sigmoid(student_logits / self.temperature)
            teacher_probs = torch.sigmoid(teacher_logits / self.temperature)

            # Stack to create [neg_prob, pos_prob] format
            student_probs_2d = torch.stack([1 - student_probs, student_probs], dim=-1)
            teacher_probs_2d = torch.stack([1 - teacher_probs, teacher_probs], dim=-1)

            student_log_probs = torch.log(student_probs_2d + 1e-8)

            kl_loss = F.kl_div(
                student_log_probs, teacher_probs_2d, reduction="batchmean"
            )
        else:
            kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")

        # Scale by temperature squared (standard practice)
        return kl_loss * (self.temperature**2)

    def _mse_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Mean squared error loss between logits."""
        return F.mse_loss(student_logits, teacher_logits)

    def _cosine_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Cosine similarity loss between logits."""
        cosine_sim = F.cosine_similarity(student_logits, teacher_logits, dim=-1)
        return 1.0 - cosine_sim.mean()


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for intermediate layer distillation."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super(FeatureMatchingLoss, self).__init__()
        # Projection layer if dimensions don't match
        if student_dim != teacher_dim:
            self.projection = nn.Linear(student_dim, teacher_dim)
        else:
            self.projection = nn.Identity()

    def forward(
        self, student_features: torch.Tensor, teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Calculate feature matching loss."""
        projected_student = self.projection(student_features)
        return F.mse_loss(projected_student, teacher_features.detach())


class AttentionTransferLoss(nn.Module):
    """Attention transfer loss for spatial attention distillation."""

    def __init__(self):
        super(AttentionTransferLoss, self).__init__()

    def forward(
        self, student_attention: torch.Tensor, teacher_attention: torch.Tensor
    ) -> torch.Tensor:
        """Calculate attention transfer loss."""
        # Normalize attention maps
        student_norm = F.normalize(
            student_attention.view(student_attention.size(0), -1), p=2, dim=1
        )
        teacher_norm = F.normalize(
            teacher_attention.view(teacher_attention.size(0), -1), p=2, dim=1
        )

        return F.mse_loss(student_norm, teacher_norm.detach())


class KnowledgeDistiller:
    """Main knowledge distillation class."""

    def __init__(self, config: Optional[DistillationConfig] = None):
        self.config = config or DistillationConfig()
        self.distillation_loss = DistillationLoss(self.config)

        # Feature matching components
        self.feature_matching_losses = nn.ModuleDict()

        # Attention transfer components
        self.attention_transfer_loss = AttentionTransferLoss()

    def distill_knowledge(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> DistillationResult:
        """
        Perform knowledge distillation from teacher to student.

        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            X: Training features
            y: Training targets

        Returns:
            DistillationResult with distillation results
        """
        start_time = datetime.now()

        try:
            logger.info("Starting knowledge distillation")

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train.values)
            y_train_tensor = torch.FloatTensor(y_train.values)
            X_val_tensor = torch.FloatTensor(X_val.values)
            y_val_tensor = torch.FloatTensor(y_val.values)

            # Evaluate teacher model
            teacher_performance = self._evaluate_model(
                teacher_model, X_val_tensor, y_val_tensor
            )
            teacher_params = sum(p.numel() for p in teacher_model.parameters())

            logger.info(
                f"Teacher model - AUC: {teacher_performance.get('roc_auc', 0.0):.4f}, "
                f"Params: {teacher_params}"
            )

            # Setup feature matching if requested
            if self.config.use_feature_matching:
                self._setup_feature_matching(teacher_model, student_model)

            # Train student model
            if self.config.use_progressive_distillation:
                trained_student, training_history = self._progressive_distillation(
                    teacher_model,
                    student_model,
                    X_train_tensor,
                    y_train_tensor,
                    X_val_tensor,
                    y_val_tensor,
                )
            else:
                trained_student, training_history = self._standard_distillation(
                    teacher_model,
                    student_model,
                    X_train_tensor,
                    y_train_tensor,
                    X_val_tensor,
                    y_val_tensor,
                )

            # Evaluate student model
            student_performance = self._evaluate_model(
                trained_student, X_val_tensor, y_val_tensor
            )
            student_params = sum(p.numel() for p in trained_student.parameters())

            # Calculate metrics
            compression_ratio = (
                teacher_params / student_params if student_params > 0 else 1.0
            )
            parameter_reduction = (
                1.0 - (student_params / teacher_params) if teacher_params > 0 else 0.0
            )

            performance_drop = {
                key: teacher_performance[key] - student_performance.get(key, 0.0)
                for key in teacher_performance.keys()
            }

            # Check if distillation meets criteria
            auc_drop = performance_drop.get("roc_auc", 0.0)
            success = auc_drop <= self.config.accuracy_threshold

            if not success:
                logger.warning(
                    f"Distillation failed: AUC drop ({auc_drop:.4f}) exceeds threshold ({self.config.accuracy_threshold})"
                )

            # Save student model
            student_model_path = None
            if self.config.save_student_model and success:
                student_model_path = self._save_student_model(trained_student)

            # Find best epoch
            best_epoch = 0
            if training_history:
                best_epoch = max(
                    range(len(training_history)),
                    key=lambda i: training_history[i].get("val_auc", 0.0),
                )

            distillation_time = (datetime.now() - start_time).total_seconds()

            # Log completion
            audit_logger.log_model_operation(
                user_id="system",
                model_id="knowledge_distillation",
                operation="distillation_completed",
                success=success,
                details={
                    "distillation_time_seconds": distillation_time,
                    "compression_ratio": compression_ratio,
                    "parameter_reduction": parameter_reduction,
                    "auc_drop": auc_drop,
                    "temperature": self.config.temperature,
                },
            )

            logger.info(
                f"Knowledge distillation completed in {distillation_time:.2f} seconds"
            )
            logger.info(f"Compression ratio: {compression_ratio:.2f}x")
            logger.info(f"Parameter reduction: {parameter_reduction:.4f}")
            logger.info(f"AUC drop: {auc_drop:.4f}")

            return DistillationResult(
                success=success,
                teacher_model=teacher_model,
                student_model=trained_student,
                config=self.config,
                teacher_params=teacher_params,
                student_params=student_params,
                compression_ratio=compression_ratio,
                parameter_reduction=parameter_reduction,
                teacher_performance=teacher_performance,
                student_performance=student_performance,
                performance_drop=performance_drop,
                training_history=training_history,
                best_epoch=best_epoch,
                distillation_time_seconds=distillation_time,
                student_model_path=student_model_path,
                message=(
                    "Knowledge distillation completed successfully"
                    if success
                    else f"Distillation failed: AUC drop too large ({auc_drop:.4f})"
                ),
            )

        except Exception as e:
            distillation_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Knowledge distillation failed: {str(e)}"
            logger.error(error_message)

            return DistillationResult(
                success=False,
                teacher_model=teacher_model,
                student_model=student_model,
                config=self.config,
                distillation_time_seconds=distillation_time,
                message=error_message,
            )

    def _standard_distillation(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> Tuple[nn.Module, List[Dict[str, float]]]:
        """Perform standard knowledge distillation."""
        # Set teacher to evaluation mode
        teacher_model.eval()

        # Setup training
        optimizer = self._setup_optimizer(student_model)
        scheduler = (
            self._setup_scheduler(optimizer) if self.config.use_scheduler else None
        )
        scaler = GradScaler() if self.config.use_mixed_precision else None

        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # Training loop
        training_history = []
        best_val_auc = 0.0
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Training phase
            student_model.train()
            epoch_losses = {"total": 0.0, "distillation": 0.0, "student": 0.0}

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                # Get teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_logits = teacher_model(batch_X)

                # Training step with mixed precision
                if self.config.use_mixed_precision and scaler is not None:
                    with autocast():
                        student_logits = student_model(batch_X)
                        loss, loss_components = self.distillation_loss(
                            student_logits, teacher_logits, batch_y
                        )

                    scaler.scale(loss).backward()

                    if self.config.gradient_clip_value > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            student_model.parameters(), self.config.gradient_clip_value
                        )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    student_logits = student_model(batch_X)
                    loss, loss_components = self.distillation_loss(
                        student_logits, teacher_logits, batch_y
                    )

                    loss.backward()

                    if self.config.gradient_clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(
                            student_model.parameters(), self.config.gradient_clip_value
                        )

                    optimizer.step()

                # Accumulate losses
                for key in epoch_losses:
                    if key in loss_components:
                        epoch_losses[key] += loss_components[key]

            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= len(train_loader)

            # Validation phase
            val_metrics = self._evaluate_model(student_model, X_val, y_val)
            val_auc = val_metrics.get("roc_auc", 0.0)

            # Learning rate scheduling
            if scheduler:
                if self.config.scheduler_type == "plateau":
                    scheduler.step(epoch_losses["total"])
                else:
                    scheduler.step()

            # Early stopping and best model tracking
            if val_auc > best_val_auc + self.config.min_delta:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(student_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # Record training history
            history_entry = {
                "epoch": epoch,
                "train_loss": epoch_losses["total"],
                "distillation_loss": epoch_losses["distillation"],
                "student_loss": epoch_losses["student"],
                "val_auc": val_auc,
                "val_f1": val_metrics.get("f1_score", 0.0),
            }
            training_history.append(history_entry)

            # Logging
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                logger.info(
                    f"Epoch {epoch}: Loss: {epoch_losses['total']:.4f}, "
                    f"Val AUC: {val_auc:.4f}"
                )

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            student_model.load_state_dict(best_model_state)

        return student_model, training_history

    def _progressive_distillation(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> Tuple[nn.Module, List[Dict[str, float]]]:
        """Perform progressive distillation with decreasing temperature."""
        all_training_history = []
        current_student = copy.deepcopy(student_model)

        epochs_per_stage = self.config.epochs // len(self.config.progressive_stages)

        for stage, temperature in enumerate(self.config.progressive_stages):
            logger.info(
                f"Progressive distillation stage {stage + 1}/{len(self.config.progressive_stages)}, "
                f"Temperature: {temperature}"
            )

            # Update temperature
            original_temp = self.config.temperature
            self.config.temperature = temperature
            self.distillation_loss.temperature = temperature

            # Train for this stage
            stage_config = copy.deepcopy(self.config)
            stage_config.epochs = epochs_per_stage
            stage_config.early_stopping_patience = max(3, epochs_per_stage // 3)

            # Create temporary distiller for this stage
            temp_distiller = KnowledgeDistiller(stage_config)

            current_student, stage_history = temp_distiller._standard_distillation(
                teacher_model, current_student, X_train, y_train, X_val, y_val
            )

            # Adjust epoch numbers for global history
            for entry in stage_history:
                entry["epoch"] += stage * epochs_per_stage

            all_training_history.extend(stage_history)

            # Restore original temperature
            self.config.temperature = original_temp
            self.distillation_loss.temperature = original_temp

        return current_student, all_training_history

    def _setup_feature_matching(
        self, teacher_model: nn.Module, student_model: nn.Module
    ):
        """Setup feature matching between teacher and student intermediate layers."""
        # This is a simplified implementation
        # In practice, you'd need to identify corresponding layers and their dimensions
        logger.info("Feature matching setup - simplified implementation")

    def _setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Setup optimizer for student model."""
        if self.config.optimizer == "adam":
            return optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            return optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )

    def _setup_scheduler(self, optimizer: optim.Optimizer):
        """Setup learning rate scheduler."""
        if self.config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=self.config.scheduler_factor
            )
        else:
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
            )

    def _evaluate_model(
        self, model: nn.Module, X_val: torch.Tensor, y_val: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            model.eval()

            with torch.no_grad():
                outputs = model(X_val)
                probs = torch.sigmoid(outputs.squeeze()).numpy()
                preds = (probs > 0.5).astype(int)

            metrics = {
                "accuracy": accuracy_score(y_val.numpy(), preds),
                "roc_auc": roc_auc_score(y_val.numpy(), probs),
                "f1_score": f1_score(y_val.numpy(), preds, average="weighted"),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"accuracy": 0.0, "roc_auc": 0.0, "f1_score": 0.0}

    def _save_student_model(self, student_model: nn.Module) -> str:
        """Save distilled student model."""
        save_path = Path(self.config.student_model_path)
        save_path.mkdir(parents=True, exist_ok=True)

        model_file = save_path / "distilled_student_model.pth"

        # Save model state
        torch.save(
            {
                "model_state_dict": student_model.state_dict(),
                "config": self.config,
                "distillation_method": "knowledge_distillation",
                "temperature": self.config.temperature,
                "saved_at": datetime.now().isoformat(),
            },
            model_file,
        )

        logger.info(f"Distilled student model saved to {model_file}")
        return str(model_file)
