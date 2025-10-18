"""
Deep Neural Network (DNN) baseline model for credit risk prediction.
Includes configurable architecture, mixed precision training, and comprehensive optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
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
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

try:
    from ..core.interfaces import BaseModel, TrainingMetrics
    from ..core.config import get_config
    from ..core.logging import get_logger, get_audit_logger
    from ..data.cross_validation import validate_model_cv, get_imbalanced_cv_config
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
class DNNConfig:
    """Configuration for Deep Neural Network model."""
    # Architecture parameters
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    dropout_rate: float = 0.3
    activation: str = 'relu'  # 'relu', 'leaky_relu', 'elu', 'gelu'
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 100
    early_stopping_patience: int = 15
    min_delta: float = 1e-4
    
    # Optimization parameters
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd', 'rmsprop'
    weight_decay: float = 1e-4
    gradient_clip_value: float = 1.0
    
    # Loss function parameters
    loss_function: str = 'focal'  # 'bce', 'focal', 'weighted_bce'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    class_weights: Optional[List[float]] = None
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = 'onecycle'  # 'onecycle', 'cosine', 'step', 'plateau'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Mixed precision training
    use_mixed_precision: bool = True
    
    # Regularization
    l1_lambda: float = 0.0
    l2_lambda: float = 0.0
    
    # Model saving
    save_model: bool = True
    model_path: str = "models/dnn"
    save_best_only: bool = True
    
    # Device configuration
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'


@dataclass
class DNNResult:
    """Result of DNN training and evaluation."""
    success: bool
    model: Optional['DNNModel']
    config: DNNConfig
    training_metrics: List[TrainingMetrics]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    training_time_seconds: float
    model_path: Optional[str]
    best_epoch: int
    message: str


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DNNModel(BaseModel):
    """Deep Neural Network model implementation."""
    
    def __init__(self, input_dim: int, config: Optional[DNNConfig] = None):
        super(DNNModel, self).__init__()
        self.config = config or DNNConfig()
        self.input_dim = input_dim
        self.device = self._get_device()
        
        # Build network architecture
        self.layers = self._build_network()
        
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
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(self.config.device)
    
    def _build_network(self) -> nn.ModuleList:
        """Build the neural network architecture."""
        layers = nn.ModuleList()
        
        # Input layer
        prev_dim = self.input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(self.config.hidden_layers):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Layer normalization (alternative to batch norm)
            if self.config.use_layer_norm and not self.config.use_batch_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation function
            if self.config.activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif self.config.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.01, inplace=True))
            elif self.config.activation == 'elu':
                layers.append(nn.ELU(inplace=True))
            elif self.config.activation == 'gelu':
                layers.append(nn.GELU())
            
            # Dropout
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        return layers
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)  # Remove last dimension for binary classification
    
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
        
        # Use gradient-based feature importance
        self.eval()
        
        # Create a dummy input to compute gradients
        dummy_input = torch.randn(1, self.input_dim, requires_grad=True, device=self.device)
        output = self.forward(dummy_input)
        
        # Compute gradients
        gradients = torch.autograd.grad(output, dummy_input, create_graph=False)[0]
        importance_scores = torch.abs(gradients).squeeze().cpu().numpy()
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, score in enumerate(importance_scores):
            feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"
            feature_importance[feature_name] = float(score)
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create save path
        save_path = path or self.config.model_path
        model_dir = Path(save_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_file = model_dir / "dnn_model.pth"
        torch.save({
            'model_state_dict': self.state_dict(),
            'best_state_dict': self.best_state_dict,
            'config': self.config,
            'input_dim': self.input_dim,
            'feature_names': self.feature_names,
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'training_history': self.training_history,
            'device': str(self.device)
        }, model_file)
        
        # Save metadata
        metadata = {
            'model_type': 'dnn',
            'input_dim': self.input_dim,
            'architecture': self.config.hidden_layers,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'config': self.config.__dict__,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {model_file}")
        return str(model_file)
    
    def load_model(self, path: str) -> 'DNNModel':
        """Load a trained model."""
        model_path = Path(path)
        
        if model_path.is_file():
            # Path is the model file
            model_file = model_path
        else:
            # Path is the directory
            model_file = model_path / "dnn_model.pth"
        
        # Load model
        checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        
        # Restore configuration and architecture
        self.config = checkpoint['config']
        self.input_dim = checkpoint['input_dim']
        self.feature_names = checkpoint.get('feature_names')
        self.training_history = checkpoint.get('training_history', [])
        
        # Rebuild network with loaded config
        self.layers = self._build_network()
        
        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'])
        self.best_state_dict = checkpoint.get('best_state_dict')
        
        # Restore scaler
        if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
            if checkpoint['scaler_mean'] is not None:
                self.scaler.mean_ = np.array(checkpoint['scaler_mean'])
            if checkpoint['scaler_scale'] is not None:
                self.scaler.scale_ = np.array(checkpoint['scaler_scale'])
        
        self.is_trained = True
        self.to(self.device)
        
        logger.info(f"Model loaded from {model_file}")
        return self


class DNNTrainer:
    """Trainer for Deep Neural Network models."""
    
    def __init__(self, config: Optional[DNNConfig] = None):
        self.config = config or DNNConfig()
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.2) -> DNNResult:
        """Train and evaluate DNN model."""
        start_time = datetime.now()
        
        try:
            logger.info("Starting DNN training and evaluation")
            
            # Prepare data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Create and train model
            model = DNNModel(input_dim=X_train.shape[1], config=self.config)
            model.feature_names = list(X_train.columns)
            
            # Train model
            training_metrics = self._train_model(model, X_train, y_train, X_val, y_val)
            
            # Evaluate model
            validation_metrics = self._evaluate_model(model, X_val, y_val, "Validation")
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
                best_auc = max(training_metrics, key=lambda x: x.auc_roc)
                best_epoch = best_auc.epoch
            
            # Log training completion
            audit_logger.log_model_operation(
                user_id="system",
                model_id="dnn_baseline",
                operation="training_completed",
                success=True,
                details={
                    "training_time_seconds": training_time,
                    "test_auc": test_metrics.get('roc_auc', 0.0),
                    "best_epoch": best_epoch,
                    "num_parameters": sum(p.numel() for p in model.parameters())
                }
            )
            
            logger.info(f"DNN training completed in {training_time:.2f} seconds")
            
            return DNNResult(
                success=True,
                model=model,
                config=self.config,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                feature_importance=feature_importance,
                training_time_seconds=training_time,
                model_path=model_path,
                best_epoch=best_epoch,
                message="DNN training completed successfully"
            )
            
        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            error_message = f"DNN training failed: {str(e)}"
            logger.error(error_message)
            
            return DNNResult(
                success=False,
                model=None,
                config=self.config,
                training_metrics=[],
                validation_metrics={},
                test_metrics={},
                feature_importance={},
                training_time_seconds=training_time,
                model_path=None,
                best_epoch=0,
                message=error_message
            )
    
    def _train_model(self, model: DNNModel, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series) -> List[TrainingMetrics]:
        """Train the DNN model."""
        
        # Prepare data
        X_train_scaled = model.scaler.fit_transform(X_train)
        X_val_scaled = model.scaler.transform(X_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(model.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(model.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(model.device)
        y_val_tensor = torch.FloatTensor(y_val.values).to(model.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Setup loss function
        if self.config.loss_function == 'focal':
            criterion = FocalLoss(alpha=self.config.focal_alpha, gamma=self.config.focal_gamma)
        elif self.config.loss_function == 'weighted_bce':
            pos_weight = torch.tensor([len(y_train) / (2 * sum(y_train))]).to(model.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Setup optimizer
        if self.config.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, 
                                 weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate,
                                  weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.config.learning_rate,
                                momentum=0.9, weight_decay=self.config.weight_decay)
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay)
        
        # Setup scheduler
        scheduler = None
        if self.config.use_scheduler:
            if self.config.scheduler_type == 'onecycle':
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=self.config.learning_rate,
                    steps_per_epoch=len(train_loader), epochs=self.config.epochs
                )
            elif self.config.scheduler_type == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.config.epochs
                )
            elif self.config.scheduler_type == 'step':
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=30, gamma=self.config.scheduler_factor
                )
            elif self.config.scheduler_type == 'plateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', patience=self.config.scheduler_patience,
                    factor=self.config.scheduler_factor
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
                            l1_reg = sum(p.abs().sum() for p in model.parameters())
                            loss += self.config.l1_lambda * l1_reg
                        
                        if self.config.l2_lambda > 0:
                            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
                            loss += self.config.l2_lambda * l2_reg
                    
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.config.gradient_clip_value > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_value)
                    
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
                        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
                        loss += self.config.l2_lambda * l2_reg
                    
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config.gradient_clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_value)
                    
                    optimizer.step()
                
                train_loss += loss.item()
                
                # Update scheduler (for OneCycleLR)
                if scheduler and self.config.scheduler_type == 'onecycle':
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
            val_f1 = f1_score(val_targets, val_pred_binary, average='weighted')
            val_precision = precision_score(val_targets, val_pred_binary, average='weighted', zero_division=0)
            val_recall = recall_score(val_targets, val_pred_binary, average='weighted', zero_division=0)
            
            # Create training metrics
            metrics = TrainingMetrics(
                experiment_id=f"dnn_epoch_{epoch}",
                model_type="dnn",
                epoch=epoch,
                train_loss=train_loss / len(train_loader),
                val_loss=val_loss,
                auc_roc=val_auc,
                f1_score=val_f1,
                precision=val_precision,
                recall=val_recall,
                energy_consumed_kwh=0.0,  # TODO: Integrate with sustainability monitor
                carbon_emissions_kg=0.0,
                training_time_seconds=0.0
            )
            training_metrics.append(metrics)
            
            # Update scheduler (except OneCycleLR)
            if scheduler and self.config.scheduler_type != 'onecycle':
                if self.config.scheduler_type == 'plateau':
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
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
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
    
    def _evaluate_model(self, model: DNNModel, X: pd.DataFrame, y: pd.Series,
                       dataset_name: str) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        try:
            # Prepare data
            X_scaled = model.scaler.transform(X)
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
                'accuracy': accuracy_score(y, predictions_np),
                'precision': precision_score(y, predictions_np, average='weighted', zero_division=0),
                'recall': recall_score(y, predictions_np, average='weighted', zero_division=0),
                'f1_score': f1_score(y, predictions_np, average='weighted', zero_division=0),
                'roc_auc': roc_auc_score(y, probs_np[:, 1])
            }
            
            logger.info(f"{dataset_name} Metrics - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed for {dataset_name}: {e}")
            return {}


# Factory functions and utilities
def create_dnn_model(input_dim: int, config: Optional[DNNConfig] = None) -> DNNModel:
    """Create a DNN model instance."""
    return DNNModel(input_dim, config)


def train_dnn_baseline(X: pd.DataFrame, y: pd.Series, 
                      config: Optional[DNNConfig] = None) -> DNNResult:
    """Convenience function to train DNN baseline."""
    trainer = DNNTrainer(config)
    return trainer.train_and_evaluate(X, y)


def get_default_dnn_config() -> DNNConfig:
    """Get default DNN configuration."""
    return DNNConfig()


def get_fast_dnn_config() -> DNNConfig:
    """Get fast DNN configuration for testing."""
    return DNNConfig(
        hidden_layers=[128, 64],
        epochs=20,
        early_stopping_patience=5,
        batch_size=128,
        use_mixed_precision=False
    )


def get_optimized_dnn_config() -> DNNConfig:
    """Get optimized DNN configuration for production."""
    return DNNConfig(
        hidden_layers=[1024, 512, 256, 128, 64],
        epochs=200,
        early_stopping_patience=25,
        batch_size=512,
        learning_rate=0.0005,
        use_mixed_precision=True,
        use_scheduler=True,
        scheduler_type='onecycle',
        dropout_rate=0.4,
        weight_decay=1e-3
    )