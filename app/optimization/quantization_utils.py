"""
Utility functions for model quantization.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .model_quantization import (
    ModelQuantizer,
    QuantizationConfig,
    QuantizationResult,
)


# Utility functions
def quantize_model(
    model: nn.Module,
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[QuantizationConfig] = None,
) -> QuantizationResult:
    """
    Convenience function to quantize a model.

    Args:
        model: Model to quantize
        X: Training/calibration features
        y: Training/calibration targets
        config: Quantization configuration

    Returns:
        QuantizationResult with quantization results
    """
    quantizer = ModelQuantizer(config)
    return quantizer.quantize_and_validate(model, X, y)


def get_default_quantization_config() -> QuantizationConfig:
    """Get default quantization configuration."""
    return QuantizationConfig()


def get_qat_config(epochs: int = 20) -> QuantizationConfig:
    """Get Quantization-Aware Training configuration."""
    return QuantizationConfig(
        quantization_method="qat",
        qat_epochs=epochs,
        qat_learning_rate=0.0001,
        qat_warmup_epochs=5,
        validate_quantization=True,
        fuse_modules=True,
    )


def get_static_quantization_config(
    calibration_size: int = 1000,
) -> QuantizationConfig:
    """Get post-training static quantization configuration."""
    return QuantizationConfig(
        quantization_method="post_training_static",
        calibration_dataset_size=calibration_size,
        calibration_batch_size=32,
        validate_quantization=True,
        fuse_modules=True,
    )


def get_dynamic_quantization_config() -> QuantizationConfig:
    """Get post-training dynamic quantization configuration."""
    return QuantizationConfig(
        quantization_method="post_training_dynamic", validate_quantization=True
    )


def get_mobile_quantization_config() -> QuantizationConfig:
    """Get quantization configuration optimized for mobile deployment."""
    return QuantizationConfig(
        quantization_method="post_training_static",
        backend="qnnpack",
        calibration_dataset_size=500,
        calibration_batch_size=16,
        optimize_for_mobile=True,
        fuse_modules=True,
        validate_quantization=True,
    )


def analyze_quantization_impact(
    original_model: nn.Module,
    quantized_model: nn.Module,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Analyze the impact of quantization on model performance and efficiency.

    Args:
        original_model: Original unquantized model
        quantized_model: Quantized model
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary with analysis results
    """
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values)

    # Evaluate both models
    original_perf = _evaluate_model_simple(
        original_model, X_test_tensor, y_test_tensor
    )
    quantized_perf = _evaluate_model_simple(
        quantized_model, X_test_tensor, y_test_tensor
    )

    # Calculate model sizes
    original_size = _calculate_model_size_simple(original_model)
    quantized_size = _calculate_model_size_simple(quantized_model)

    # Measure inference times
    original_time = _measure_inference_time_simple(
        original_model, X_test_tensor
    )
    quantized_time = _measure_inference_time_simple(
        quantized_model, X_test_tensor
    )

    # Compression metrics
    compression_ratio = (
        original_size / quantized_size if quantized_size > 0 else 1.0
    )
    size_reduction = (
        1.0 - (quantized_size / original_size) if original_size > 0 else 0.0
    )
    speedup_ratio = (
        original_time / quantized_time if quantized_time > 0 else 1.0
    )

    # Performance impact
    performance_drop = {
        key: original_perf[key] - quantized_perf.get(key, 0.0)
        for key in original_perf.keys()
    }

    analysis = {
        "compression_metrics": {
            "compression_ratio": compression_ratio,
            "size_reduction": size_reduction,
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
        },
        "performance_metrics": {
            "original_performance": original_perf,
            "quantized_performance": quantized_perf,
            "performance_drop": performance_drop,
        },
        "inference_metrics": {
            "speedup_ratio": speedup_ratio,
            "original_inference_time_ms": original_time,
            "quantized_inference_time_ms": quantized_time,
        },
    }

    return analysis


def compare_quantization_methods(
    model: nn.Module, X: pd.DataFrame, y: pd.Series
) -> Dict[str, QuantizationResult]:
    """
    Compare different quantization methods on the same model.

    Args:
        model: Model to quantize
        X: Training/calibration features
        y: Training/calibration targets

    Returns:
        Dictionary mapping method names to results
    """
    results = {}

    # QAT
    try:
        qat_config = get_qat_config(epochs=10)  # Shorter for comparison
        qat_result = quantize_model(copy.deepcopy(model), X, y, qat_config)
        results["qat"] = qat_result
    except Exception as e:
        print(f"QAT failed: {e}")
        results["qat"] = None

    # Static quantization
    try:
        static_config = get_static_quantization_config(calibration_size=500)
        static_result = quantize_model(
            copy.deepcopy(model), X, y, static_config
        )
        results["static"] = static_result
    except Exception as e:
        print(f"Static quantization failed: {e}")
        results["static"] = None

    # Dynamic quantization
    try:
        dynamic_config = get_dynamic_quantization_config()
        dynamic_result = quantize_model(
            copy.deepcopy(model), X, y, dynamic_config
        )
        results["dynamic"] = dynamic_result
    except Exception as e:
        print(f"Dynamic quantization failed: {e}")
        results["dynamic"] = None

    return results


def load_quantized_model(model_path: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a quantized model from disk.

    Args:
        model_path: Path to saved quantized model

    Returns:
        Tuple of (loaded_model, metadata)
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    metadata = {
        "config": checkpoint.get("config"),
        "quantization_method": checkpoint.get("quantization_method"),
        "backend": checkpoint.get("backend"),
        "saved_at": checkpoint.get("saved_at"),
    }

    # Note: This is a placeholder - actual implementation would need to reconstruct
    # the quantized model architecture based on the saved configuration
    print(
        f"Loaded quantized model with method: {metadata['quantization_method']}"
    )

    return None, metadata


def benchmark_quantized_model(
    model: nn.Module, X_test: pd.DataFrame, num_runs: int = 1000
) -> Dict[str, float]:
    """
    Benchmark quantized model inference performance.

    Args:
        model: Quantized model to benchmark
        X_test: Test data
        num_runs: Number of inference runs

    Returns:
        Dictionary with benchmark results
    """
    X_test_tensor = torch.FloatTensor(X_test.values)

    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(X_test_tensor[:1])

    # Benchmark
    import time

    times = []

    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(X_test_tensor[:1])
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms

    times = np.array(times)

    return {
        "mean_inference_time_ms": np.mean(times),
        "std_inference_time_ms": np.std(times),
        "min_inference_time_ms": np.min(times),
        "max_inference_time_ms": np.max(times),
        "p50_inference_time_ms": np.percentile(times, 50),
        "p95_inference_time_ms": np.percentile(times, 95),
        "p99_inference_time_ms": np.percentile(times, 99),
    }


def validate_quantized_model_accuracy(
    original_model: nn.Module,
    quantized_model: nn.Module,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Validate that quantized model maintains acceptable accuracy.

    Args:
        original_model: Original model
        quantized_model: Quantized model
        X_test: Test features
        y_test: Test targets
        threshold: Maximum acceptable accuracy drop

    Returns:
        Validation results
    """
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values)

    # Evaluate both models
    original_perf = _evaluate_model_simple(
        original_model, X_test_tensor, y_test_tensor
    )
    quantized_perf = _evaluate_model_simple(
        quantized_model, X_test_tensor, y_test_tensor
    )

    # Calculate drops
    accuracy_drop = original_perf["accuracy"] - quantized_perf["accuracy"]
    auc_drop = original_perf["roc_auc"] - quantized_perf["roc_auc"]
    f1_drop = original_perf["f1_score"] - quantized_perf["f1_score"]

    # Validation
    accuracy_valid = accuracy_drop <= threshold
    auc_valid = auc_drop <= threshold
    f1_valid = f1_drop <= threshold

    overall_valid = accuracy_valid and auc_valid and f1_valid

    return {
        "validation_passed": overall_valid,
        "accuracy_drop": accuracy_drop,
        "auc_drop": auc_drop,
        "f1_drop": f1_drop,
        "accuracy_valid": accuracy_valid,
        "auc_valid": auc_valid,
        "f1_valid": f1_valid,
        "threshold": threshold,
        "original_performance": original_perf,
        "quantized_performance": quantized_perf,
    }


# Helper functions
def _evaluate_model_simple(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor
) -> Dict[str, float]:
    """Simple model evaluation."""
    try:
        model.eval()

        with torch.no_grad():
            outputs = model(X)
            probs = torch.sigmoid(outputs.squeeze()).numpy()
            preds = (probs > 0.5).astype(int)

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        return {
            "accuracy": accuracy_score(y.numpy(), preds),
            "roc_auc": roc_auc_score(y.numpy(), probs),
            "f1_score": f1_score(y.numpy(), preds, average="weighted"),
        }
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {"accuracy": 0.0, "roc_auc": 0.0, "f1_score": 0.0}


def _calculate_model_size_simple(model: nn.Module) -> float:
    """Simple model size calculation in MB."""
    param_size = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def _measure_inference_time_simple(
    model: nn.Module, X: torch.Tensor, num_runs: int = 100
) -> float:
    """Simple inference time measurement in milliseconds."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(X[:1])

    # Measure
    import time

    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(X[:1])

    end_time = time.time()
    return ((end_time - start_time) / num_runs) * 1000
