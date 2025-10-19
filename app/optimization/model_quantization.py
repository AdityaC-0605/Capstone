"""
Model quantization system for neural network compression.
Implements Quantization-Aware Training (QAT), post-training static and dynamic quantization,
and INT8 model conversion pipeline with accuracy validation.
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
from torch.quantization import (
    DeQuantStub,
    QuantStub,
    convert,
    get_default_qat_qconfig,
    get_default_qconfig,
    prepare,
    prepare_qat,
    quantize_dynamic,
)
from torch.quantization.fake_quantize import FakeQuantize
from torch.quantization.observer import (
    MinMaxObserver,
    MovingAverageMinMaxObserver,
)

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
class QuantizationConfig:
    """Configuration for model quantization."""

    # Quantization method
    quantization_method: str = (
        "qat"  # 'qat', 'post_training_static', 'post_training_dynamic'
    )

    # Quantization-Aware Training (QAT)
    qat_epochs: int = 20
    qat_learning_rate: float = 0.0001
    qat_warmup_epochs: int = 5

    # Post-training quantization
    calibration_dataset_size: int = 1000
    calibration_batch_size: int = 32

    # Quantization configuration - auto-detect best backend
    backend: str = "auto"  # 'auto', 'fbgemm', 'qnnpack', 'onednn'
    qconfig_type: str = "default"  # 'default', 'custom'

    # Custom quantization settings
    weight_observer: str = (
        "MinMaxObserver"  # 'MinMaxObserver', 'MovingAverageMinMaxObserver'
    )
    activation_observer: str = "MovingAverageMinMaxObserver"
    weight_dtype: str = "qint8"  # 'qint8', 'quint8'
    activation_dtype: str = "quint8"

    # Fake quantization for QAT
    weight_fake_quantize: bool = True
    activation_fake_quantize: bool = True

    # Model preparation
    prepare_custom_config: Optional[Dict[str, Any]] = None
    convert_custom_config: Optional[Dict[str, Any]] = None

    # Validation and accuracy
    validate_quantization: bool = True
    accuracy_threshold: float = 0.05  # Max acceptable accuracy drop

    # Model saving
    save_quantized_model: bool = True
    quantized_model_path: str = "models/quantized"

    # Optimization
    fuse_modules: bool = True  # Fuse conv-bn-relu patterns
    optimize_for_mobile: bool = False


@dataclass
class QuantizationResult:
    """Result of model quantization operation."""

    success: bool
    original_model: Optional[nn.Module]
    quantized_model: Optional[nn.Module]
    config: QuantizationConfig

    # Compression metrics
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 0.0
    size_reduction: float = 0.0

    # Performance metrics
    original_performance: Dict[str, float] = field(default_factory=dict)
    quantized_performance: Dict[str, float] = field(default_factory=dict)
    performance_drop: Dict[str, float] = field(default_factory=dict)

    # Inference metrics
    original_inference_time_ms: float = 0.0
    quantized_inference_time_ms: float = 0.0
    speedup_ratio: float = 0.0

    # Quantization details
    quantization_method: str = ""
    backend_used: str = ""
    quantization_time_seconds: float = 0.0
    calibration_time_seconds: float = 0.0

    # Model paths
    quantized_model_path: Optional[str] = None

    message: str = ""


class QuantizableModel(nn.Module):
    """Wrapper to make models quantizable."""

    def __init__(self, model: nn.Module):
        super(QuantizableModel, self).__init__()
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse modules for better quantization performance."""
        # This is a simplified implementation
        # In practice, you'd fuse specific patterns like conv-bn-relu
        pass


class BaseQuantizer(ABC):
    """Abstract base class for quantization methods."""

    @abstractmethod
    def quantize_model(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        X_train: Optional[torch.Tensor] = None,
        y_train: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Quantize the model according to the configuration."""
        pass

    @abstractmethod
    def prepare_model(
        self, model: nn.Module, config: QuantizationConfig
    ) -> nn.Module:
        """Prepare model for quantization."""
        pass


class QATQuantizer(BaseQuantizer):
    """Quantization-Aware Training quantizer."""

    def quantize_model(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        X_train: Optional[torch.Tensor] = None,
        y_train: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Quantize model using QAT."""
        if X_train is None or y_train is None:
            raise ValueError("QAT requires training data")

        # Prepare model for QAT
        prepared_model = self.prepare_model(model, config)

        # Train with fake quantization
        trained_model = self._train_qat_model(
            prepared_model, X_train, y_train, config
        )

        # Convert to quantized model
        quantized_model = convert(trained_model, inplace=False)

        return quantized_model

    def prepare_model(
        self, model: nn.Module, config: QuantizationConfig
    ) -> nn.Module:
        """Prepare model for QAT."""
        try:
            # Wrap model to make it quantizable
            quantizable_model = QuantizableModel(model)

            # Set backend with error handling
            try:
                torch.backends.quantized.engine = config.backend
            except Exception as e:
                logger.warning(f"Failed to set backend {config.backend}: {e}")
                # Try fallback backends
                for fallback in ["qnnpack", "fbgemm", "onednn"]:
                    try:
                        torch.backends.quantized.engine = fallback
                        config.backend = fallback
                        logger.info(f"Using fallback backend: {fallback}")
                        break
                    except:
                        continue

            # Set quantization configuration
            if config.qconfig_type == "default":
                try:
                    quantizable_model.qconfig = get_default_qat_qconfig(
                        config.backend
                    )
                except Exception as e:
                    logger.warning(f"Failed to get default QAT config: {e}")
                    quantizable_model.qconfig = self._create_custom_qconfig(
                        config
                    )
            else:
                quantizable_model.qconfig = self._create_custom_qconfig(config)

            # Fuse modules if requested
            if config.fuse_modules:
                try:
                    quantizable_model.fuse_model()
                except Exception as e:
                    logger.warning(f"Module fusion failed: {e}")

            # Prepare for QAT
            prepared_model = prepare_qat(quantizable_model, inplace=False)

            return prepared_model

        except Exception as e:
            logger.error(f"Failed to prepare model for QAT: {e}")
            raise

    def _create_custom_qconfig(self, config: QuantizationConfig):
        """Create custom quantization configuration."""
        # Weight observer
        if config.weight_observer == "MinMaxObserver":
            weight_observer = MinMaxObserver.with_args(
                dtype=getattr(torch, config.weight_dtype)
            )
        else:
            weight_observer = MovingAverageMinMaxObserver.with_args(
                dtype=getattr(torch, config.weight_dtype)
            )

        # Activation observer
        if config.activation_observer == "MinMaxObserver":
            activation_observer = MinMaxObserver.with_args(
                dtype=getattr(torch, config.activation_dtype)
            )
        else:
            activation_observer = MovingAverageMinMaxObserver.with_args(
                dtype=getattr(torch, config.activation_dtype)
            )

        # Create fake quantize modules
        weight_fake_quantize = FakeQuantize.with_args(
            observer=weight_observer,
            quant_min=-128,
            quant_max=127,
            dtype=getattr(torch, config.weight_dtype),
        )

        activation_fake_quantize = FakeQuantize.with_args(
            observer=activation_observer,
            quant_min=0,
            quant_max=255,
            dtype=getattr(torch, config.activation_dtype),
        )

        from torch.quantization.qconfig import QConfig

        return QConfig(
            activation=activation_fake_quantize, weight=weight_fake_quantize
        )

    def _train_qat_model(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        config: QuantizationConfig,
    ) -> nn.Module:
        """Train model with quantization-aware training."""
        model.train()

        # Setup training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.qat_learning_rate)

        # Training loop
        for epoch in range(config.qat_epochs):
            # Warmup: disable fake quantization for first few epochs
            if epoch < config.qat_warmup_epochs:
                model.apply(torch.quantization.disable_fake_quant)
            else:
                model.apply(torch.quantization.enable_fake_quant)

            # Training step
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                logger.info(
                    f"QAT Epoch {epoch}/{config.qat_epochs}, Loss: {loss.item():.4f}"
                )

        # Enable fake quantization for final model
        model.apply(torch.quantization.enable_fake_quant)

        return model


class PostTrainingStaticQuantizer(BaseQuantizer):
    """Post-training static quantization."""

    def quantize_model(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        X_train: Optional[torch.Tensor] = None,
        y_train: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Quantize model using post-training static quantization."""
        if X_train is None:
            raise ValueError("Static quantization requires calibration data")

        # Prepare model
        prepared_model = self.prepare_model(model, config)

        # Calibrate model
        calibrated_model = self._calibrate_model(
            prepared_model, X_train, config
        )

        # Convert to quantized model
        quantized_model = convert(calibrated_model, inplace=False)

        return quantized_model

    def prepare_model(
        self, model: nn.Module, config: QuantizationConfig
    ) -> nn.Module:
        """Prepare model for static quantization."""
        try:
            # Wrap model to make it quantizable
            quantizable_model = QuantizableModel(model)

            # Set backend with error handling
            try:
                torch.backends.quantized.engine = config.backend
            except Exception as e:
                logger.warning(f"Failed to set backend {config.backend}: {e}")
                # Try fallback backends
                for fallback in ["qnnpack", "fbgemm", "onednn"]:
                    try:
                        torch.backends.quantized.engine = fallback
                        config.backend = fallback
                        logger.info(f"Using fallback backend: {fallback}")
                        break
                    except:
                        continue

            # Set quantization configuration
            if config.qconfig_type == "default":
                try:
                    quantizable_model.qconfig = get_default_qconfig(
                        config.backend
                    )
                except Exception as e:
                    logger.warning(f"Failed to get default config: {e}")
                    quantizable_model.qconfig = self._create_custom_qconfig(
                        config
                    )
            else:
                quantizable_model.qconfig = self._create_custom_qconfig(config)

            # Fuse modules if requested
            if config.fuse_modules:
                try:
                    quantizable_model.fuse_model()
                except Exception as e:
                    logger.warning(f"Module fusion failed: {e}")

            # Prepare for static quantization
            prepared_model = prepare(quantizable_model, inplace=False)

            return prepared_model

        except Exception as e:
            logger.error(
                f"Failed to prepare model for static quantization: {e}"
            )
            raise

    def _create_custom_qconfig(self, config: QuantizationConfig):
        """Create custom quantization configuration for static quantization."""
        # Similar to QAT but without fake quantization
        if config.weight_observer == "MinMaxObserver":
            weight_observer = MinMaxObserver.with_args(
                dtype=getattr(torch, config.weight_dtype)
            )
        else:
            weight_observer = MovingAverageMinMaxObserver.with_args(
                dtype=getattr(torch, config.weight_dtype)
            )

        if config.activation_observer == "MinMaxObserver":
            activation_observer = MinMaxObserver.with_args(
                dtype=getattr(torch, config.activation_dtype)
            )
        else:
            activation_observer = MovingAverageMinMaxObserver.with_args(
                dtype=getattr(torch, config.activation_dtype)
            )

        from torch.quantization.qconfig import QConfig

        return QConfig(activation=activation_observer, weight=weight_observer)

    def _calibrate_model(
        self,
        model: nn.Module,
        X_calibration: torch.Tensor,
        config: QuantizationConfig,
    ) -> nn.Module:
        """Calibrate model with representative data."""
        model.eval()

        # Use subset of data for calibration
        calibration_size = min(
            config.calibration_dataset_size, len(X_calibration)
        )
        X_cal = X_calibration[:calibration_size]

        logger.info(f"Calibrating model with {calibration_size} samples")

        # Run calibration
        with torch.no_grad():
            for i in range(0, len(X_cal), config.calibration_batch_size):
                batch = X_cal[i : i + config.calibration_batch_size]
                _ = model(batch)

        return model


class PostTrainingDynamicQuantizer(BaseQuantizer):
    """Post-training dynamic quantization."""

    def quantize_model(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        X_train: Optional[torch.Tensor] = None,
        y_train: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Quantize model using post-training dynamic quantization."""
        try:
            # Dynamic quantization doesn't need calibration data
            quantized_model = quantize_dynamic(
                model,
                {nn.Linear},  # Quantize linear layers
                dtype=getattr(torch, config.weight_dtype),
            )

            return quantized_model

        except Exception as e:
            logger.warning(f"Dynamic quantization failed: {e}")
            # Fallback: return a manually quantized version
            return self._manual_quantize_model(model, config)

    def prepare_model(
        self, model: nn.Module, config: QuantizationConfig
    ) -> nn.Module:
        """Dynamic quantization doesn't need preparation."""
        return model

    def _manual_quantize_model(
        self, model: nn.Module, config: QuantizationConfig
    ) -> nn.Module:
        """Manual quantization fallback for compatibility."""
        logger.info("Using manual quantization fallback")

        # Create a copy of the model
        quantized_model = copy.deepcopy(model)

        # Manually quantize linear layer weights to simulate quantization
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                # Simulate INT8 quantization by scaling and rounding weights
                weight = module.weight.data

                # Calculate scale and zero point
                weight_min = weight.min()
                weight_max = weight.max()
                scale = (weight_max - weight_min) / 255.0
                zero_point = int(-weight_min / scale)

                # Quantize and dequantize
                quantized_weight = torch.round(weight / scale + zero_point)
                quantized_weight = torch.clamp(quantized_weight, 0, 255)
                dequantized_weight = (quantized_weight - zero_point) * scale

                # Update module weight
                module.weight.data = dequantized_weight

        return quantized_model


class ModelQuantizer:
    """Main model quantization class."""

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

        # Auto-detect and set best backend
        if self.config.backend == "auto":
            self.config.backend = self._detect_best_backend()

        # Initialize quantizer based on method
        if self.config.quantization_method == "qat":
            self.quantizer = QATQuantizer()
        elif self.config.quantization_method == "post_training_static":
            self.quantizer = PostTrainingStaticQuantizer()
        elif self.config.quantization_method == "post_training_dynamic":
            self.quantizer = PostTrainingDynamicQuantizer()
        else:
            raise ValueError(
                f"Unknown quantization method: {self.config.quantization_method}"
            )

    def _detect_best_backend(self) -> str:
        """Auto-detect the best available quantization backend."""
        import platform

        # Check available backends
        available_backends = []

        try:
            torch.backends.quantized.engine = "fbgemm"
            available_backends.append("fbgemm")
        except:
            pass

        try:
            torch.backends.quantized.engine = "qnnpack"
            available_backends.append("qnnpack")
        except:
            pass

        try:
            torch.backends.quantized.engine = "onednn"
            available_backends.append("onednn")
        except:
            pass

        # Platform-specific preferences
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "darwin":  # macOS
            if "qnnpack" in available_backends:
                return "qnnpack"
        elif system == "linux":
            if "fbgemm" in available_backends:
                return "fbgemm"
            elif "onednn" in available_backends:
                return "onednn"

        # Fallback to first available
        if available_backends:
            return available_backends[0]

        # If no backends available, use qnnpack as fallback
        logger.warning(
            "No quantization backends detected, using qnnpack as fallback"
        )
        return "qnnpack"

    def quantize_and_validate(
        self, model: nn.Module, X: pd.DataFrame, y: pd.Series
    ) -> QuantizationResult:
        """
        Quantize model and validate performance.

        Args:
            model: Model to quantize
            X: Training/calibration features
            y: Training/calibration targets

        Returns:
            QuantizationResult with quantization results
        """
        start_time = datetime.now()

        try:
            logger.info(
                f"Starting model quantization using {self.config.quantization_method}"
            )

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train.values)
            y_train_tensor = torch.FloatTensor(y_train.values)
            X_val_tensor = torch.FloatTensor(X_val.values)
            y_val_tensor = torch.FloatTensor(y_val.values)

            # Evaluate original model
            original_performance = self._evaluate_model(
                model, X_val_tensor, y_val_tensor
            )
            original_size = self._calculate_model_size(model)
            original_inference_time = self._measure_inference_time(
                model, X_val_tensor
            )

            logger.info(
                f"Original model - AUC: {original_performance.get('roc_auc', 0.0):.4f}, "
                f"Size: {original_size:.2f}MB"
            )

            # Perform quantization
            quantization_start = datetime.now()

            # Prepare training data for quantization methods that need it
            train_data = None
            train_targets = None
            if self.config.quantization_method in ["qat"]:
                train_data = X_train_tensor
                train_targets = y_train_tensor
            elif self.config.quantization_method == "post_training_static":
                train_data = X_train_tensor

            # Quantize model
            quantized_model = self.quantizer.quantize_model(
                model, self.config, train_data, train_targets
            )

            quantization_time = (
                datetime.now() - quantization_start
            ).total_seconds()

            # Evaluate quantized model
            quantized_performance = self._evaluate_model(
                quantized_model, X_val_tensor, y_val_tensor
            )
            quantized_size = self._calculate_model_size(quantized_model)
            quantized_inference_time = self._measure_inference_time(
                quantized_model, X_val_tensor
            )

            # Calculate metrics
            compression_ratio = (
                original_size / quantized_size if quantized_size > 0 else 1.0
            )
            size_reduction = (
                1.0 - (quantized_size / original_size)
                if original_size > 0
                else 0.0
            )
            speedup_ratio = (
                original_inference_time / quantized_inference_time
                if quantized_inference_time > 0
                else 1.0
            )

            performance_drop = {
                key: original_performance[key]
                - quantized_performance.get(key, 0.0)
                for key in original_performance.keys()
            }

            # Check if quantization meets criteria
            auc_drop = performance_drop.get("roc_auc", 0.0)
            success = auc_drop <= self.config.accuracy_threshold

            if not success:
                logger.warning(
                    f"Quantization failed: AUC drop ({auc_drop:.4f}) exceeds threshold ({self.config.accuracy_threshold})"
                )

            # Save quantized model
            quantized_model_path = None
            if self.config.save_quantized_model and success:
                quantized_model_path = self._save_quantized_model(
                    quantized_model
                )

            total_time = (datetime.now() - start_time).total_seconds()

            # Log completion
            audit_logger.log_model_operation(
                user_id="system",
                model_id="model_quantization",
                operation="quantization_completed",
                success=success,
                details={
                    "quantization_time_seconds": total_time,
                    "compression_ratio": compression_ratio,
                    "size_reduction": size_reduction,
                    "speedup_ratio": speedup_ratio,
                    "auc_drop": auc_drop,
                    "quantization_method": self.config.quantization_method,
                },
            )

            logger.info(f"Quantization completed in {total_time:.2f} seconds")
            logger.info(f"Compression ratio: {compression_ratio:.2f}x")
            logger.info(f"Size reduction: {size_reduction:.4f}")
            logger.info(f"Speedup ratio: {speedup_ratio:.2f}x")
            logger.info(f"AUC drop: {auc_drop:.4f}")

            return QuantizationResult(
                success=success,
                original_model=model,
                quantized_model=quantized_model,
                config=self.config,
                original_size_mb=original_size,
                quantized_size_mb=quantized_size,
                compression_ratio=compression_ratio,
                size_reduction=size_reduction,
                original_performance=original_performance,
                quantized_performance=quantized_performance,
                performance_drop=performance_drop,
                original_inference_time_ms=original_inference_time,
                quantized_inference_time_ms=quantized_inference_time,
                speedup_ratio=speedup_ratio,
                quantization_method=self.config.quantization_method,
                backend_used=self.config.backend,
                quantization_time_seconds=quantization_time,
                quantized_model_path=quantized_model_path,
                message=(
                    "Quantization completed successfully"
                    if success
                    else f"Quantization failed: AUC drop too large ({auc_drop:.4f})"
                ),
            )

        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Quantization failed: {str(e)}"
            logger.error(error_message)

            return QuantizationResult(
                success=False,
                original_model=model,
                quantized_model=None,
                config=self.config,
                quantization_time_seconds=total_time,
                message=error_message,
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

    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        model_size = (param_size + buffer_size) / (
            1024 * 1024
        )  # Convert to MB
        return model_size

    def _measure_inference_time(
        self, model: nn.Module, X_test: torch.Tensor, num_runs: int = 100
    ) -> float:
        """Measure average inference time in milliseconds."""
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(X_test[:1])

        # Measure inference time
        import time

        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(X_test[:1])

        end_time = time.time()
        avg_time_ms = ((end_time - start_time) / num_runs) * 1000

        return avg_time_ms

    def _save_quantized_model(self, model: nn.Module) -> str:
        """Save quantized model to disk."""
        save_path = Path(self.config.quantized_model_path)
        save_path.mkdir(parents=True, exist_ok=True)

        model_file = save_path / "quantized_model.pth"

        # Save model state
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": self.config,
                "quantization_method": self.config.quantization_method,
                "backend": self.config.backend,
                "saved_at": datetime.now().isoformat(),
            },
            model_file,
        )

        # Also save in TorchScript format for deployment
        try:
            scripted_model = torch.jit.script(model)
            script_file = save_path / "quantized_model_scripted.pt"
            scripted_model.save(str(script_file))
            logger.info(f"Scripted quantized model saved to {script_file}")
        except Exception as e:
            logger.warning(f"Could not save scripted model: {e}")

        logger.info(f"Quantized model saved to {model_file}")
        return str(model_file)
