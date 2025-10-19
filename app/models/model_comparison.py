"""
Model comparison utilities for benchmarking different model types.
Compares DNN, LightGBM, and other models on performance, efficiency, and sustainability.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Model imports
try:
    from .dnn_model import DNNTrainer, get_fast_dnn_config
    from .lightgbm_model import LightGBMTrainer, get_fast_lightgbm_config
except ImportError:
    # Fallback for direct execution
    import sys

    sys.path.append(str(Path(__file__).parent))
    from dnn_model import DNNTrainer, get_fast_dnn_config
    from lightgbm_model import LightGBMTrainer, get_fast_lightgbm_config

# Core imports
try:
    from ..core.logging import get_logger
except ImportError:
    import logging

    def get_logger(name):
        return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class ModelBenchmarkResult:
    """Results from model benchmarking."""

    model_name: str
    model_type: str

    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0

    # Efficiency metrics
    training_time_seconds: float = 0.0
    inference_time_ms: float = 0.0
    model_size_mb: float = 0.0
    num_parameters: int = 0

    # Resource usage
    peak_memory_mb: float = 0.0
    energy_estimate_kwh: float = 0.0

    # Additional info
    best_epoch: int = 0
    convergence_time: float = 0.0
    success: bool = False
    error_message: str = ""


@dataclass
class ComparisonConfig:
    """Configuration for model comparison."""

    test_size: float = 0.2
    random_state: int = 42

    # Models to compare
    include_lightgbm: bool = True
    include_dnn: bool = True

    # Benchmarking options
    measure_inference_time: bool = True
    inference_samples: int = 1000
    measure_memory: bool = True

    # Output options
    save_results: bool = True
    results_path: str = "benchmarks/model_comparison"
    generate_report: bool = True


class ModelComparison:
    """Comprehensive model comparison and benchmarking system."""

    def __init__(self, config: Optional[ComparisonConfig] = None):
        self.config = config or ComparisonConfig()
        self.results: List[ModelBenchmarkResult] = []

    def compare_models(
        self, X: pd.DataFrame, y: pd.Series
    ) -> List[ModelBenchmarkResult]:
        """Compare multiple models on the given dataset."""
        logger.info("Starting comprehensive model comparison")

        self.results = []

        # Test LightGBM
        if self.config.include_lightgbm:
            lgb_result = self._benchmark_lightgbm(X, y)
            self.results.append(lgb_result)

        # Test DNN
        if self.config.include_dnn:
            dnn_result = self._benchmark_dnn(X, y)
            self.results.append(dnn_result)

        # Save results if requested
        if self.config.save_results:
            self._save_results()

        # Generate report if requested
        if self.config.generate_report:
            report = self._generate_comparison_report()
            print(report)

        logger.info("Model comparison completed")
        return self.results

    def _benchmark_lightgbm(
        self, X: pd.DataFrame, y: pd.Series
    ) -> ModelBenchmarkResult:
        """Benchmark LightGBM model."""
        logger.info("Benchmarking LightGBM...")

        result = ModelBenchmarkResult(
            model_name="LightGBM Baseline", model_type="gradient_boosting"
        )

        try:
            # Configure LightGBM for fair comparison
            lgb_config = get_fast_lightgbm_config()
            lgb_config.enable_hyperopt = False  # Disable for fair comparison

            # Train and evaluate
            trainer = LightGBMTrainer(lgb_config)
            lgb_result = trainer.train_and_evaluate(
                X, y, test_size=self.config.test_size
            )

            if lgb_result.success:
                # Extract metrics
                result.accuracy = lgb_result.test_metrics.get("accuracy", 0.0)
                result.precision = lgb_result.test_metrics.get(
                    "precision", 0.0
                )
                result.recall = lgb_result.test_metrics.get("recall", 0.0)
                result.f1_score = lgb_result.test_metrics.get("f1_score", 0.0)
                result.auc_roc = lgb_result.test_metrics.get("roc_auc", 0.0)

                result.training_time_seconds = lgb_result.training_time_seconds
                result.model_size_mb = self._estimate_lightgbm_size(
                    lgb_result.model
                )
                result.num_parameters = self._estimate_lightgbm_parameters(
                    lgb_result.model
                )

                # Measure inference time
                if self.config.measure_inference_time:
                    result.inference_time_ms = (
                        self._measure_lightgbm_inference(
                            lgb_result.model,
                            X.head(self.config.inference_samples),
                        )
                    )

                result.success = True
                logger.info(
                    f"LightGBM benchmark completed - AUC: {result.auc_roc:.4f}"
                )
            else:
                result.error_message = lgb_result.message
                logger.error(
                    f"LightGBM benchmark failed: {result.error_message}"
                )

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"LightGBM benchmark failed: {e}")

        return result

    def _benchmark_dnn(
        self, X: pd.DataFrame, y: pd.Series
    ) -> ModelBenchmarkResult:
        """Benchmark DNN model."""
        logger.info("Benchmarking DNN...")

        result = ModelBenchmarkResult(
            model_name="Deep Neural Network", model_type="deep_learning"
        )

        try:
            # Configure DNN for fair comparison
            dnn_config = get_fast_dnn_config()

            # Train and evaluate
            trainer = DNNTrainer(dnn_config)
            dnn_result = trainer.train_and_evaluate(
                X, y, test_size=self.config.test_size
            )

            if dnn_result.success:
                # Extract metrics
                result.accuracy = dnn_result.test_metrics.get("accuracy", 0.0)
                result.precision = dnn_result.test_metrics.get(
                    "precision", 0.0
                )
                result.recall = dnn_result.test_metrics.get("recall", 0.0)
                result.f1_score = dnn_result.test_metrics.get("f1_score", 0.0)
                result.auc_roc = dnn_result.test_metrics.get("roc_auc", 0.0)

                result.training_time_seconds = dnn_result.training_time_seconds
                result.best_epoch = dnn_result.best_epoch
                result.model_size_mb = self._estimate_dnn_size(
                    dnn_result.model
                )
                result.num_parameters = sum(
                    p.numel() for p in dnn_result.model.parameters()
                )

                # Measure inference time
                if self.config.measure_inference_time:
                    result.inference_time_ms = self._measure_dnn_inference(
                        dnn_result.model, X.head(self.config.inference_samples)
                    )

                result.success = True
                logger.info(
                    f"DNN benchmark completed - AUC: {result.auc_roc:.4f}"
                )
            else:
                result.error_message = dnn_result.message
                logger.error(f"DNN benchmark failed: {result.error_message}")

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"DNN benchmark failed: {e}")

        return result

    def _estimate_lightgbm_size(self, model) -> float:
        """Estimate LightGBM model size in MB."""
        try:
            if model and model.is_trained:
                # Rough estimation based on number of trees and features
                num_trees = model.model.num_trees()
                num_features = model.model.num_feature()
                # Approximate: each tree node ~8 bytes, each tree ~num_leaves nodes
                estimated_bytes = (
                    num_trees * 31 * 8 * num_features
                )  # 31 is default num_leaves
                return estimated_bytes / (1024 * 1024)  # Convert to MB
            return 0.0
        except:
            return 0.0

    def _estimate_lightgbm_parameters(self, model) -> int:
        """Estimate number of parameters in LightGBM model."""
        try:
            if model and model.is_trained:
                num_trees = model.model.num_trees()
                num_features = model.model.num_feature()
                # Rough estimation: trees * leaves * features
                return num_trees * 31 * num_features
            return 0
        except:
            return 0

    def _estimate_dnn_size(self, model) -> float:
        """Estimate DNN model size in MB."""
        try:
            if model:
                # Calculate size based on parameters
                param_size = sum(
                    p.numel() * p.element_size() for p in model.parameters()
                )
                buffer_size = sum(
                    b.numel() * b.element_size() for b in model.buffers()
                )
                total_size = param_size + buffer_size
                return total_size / (1024 * 1024)  # Convert to MB
            return 0.0
        except:
            return 0.0

    def _measure_lightgbm_inference(
        self, model, X_sample: pd.DataFrame
    ) -> float:
        """Measure LightGBM inference time per sample."""
        try:
            if not model or not model.is_trained:
                return 0.0

            # Warm up
            _ = model.predict_proba(X_sample.head(10))

            # Measure time
            start_time = time.time()
            _ = model.predict_proba(X_sample)
            end_time = time.time()

            # Calculate time per sample in milliseconds
            time_per_sample = ((end_time - start_time) / len(X_sample)) * 1000
            return time_per_sample
        except:
            return 0.0

    def _measure_dnn_inference(self, model, X_sample: pd.DataFrame) -> float:
        """Measure DNN inference time per sample."""
        try:
            if not model or not model.is_trained:
                return 0.0

            import torch

            # Prepare data
            X_scaled = model.scaler.transform(X_sample)
            X_tensor = torch.FloatTensor(X_scaled).to(model.device)

            # Warm up
            model.eval()
            with torch.no_grad():
                _ = model.predict_proba(X_tensor[:10])

            # Measure time
            start_time = time.time()
            with torch.no_grad():
                _ = model.predict_proba(X_tensor)
            end_time = time.time()

            # Calculate time per sample in milliseconds
            time_per_sample = ((end_time - start_time) / len(X_sample)) * 1000
            return time_per_sample
        except:
            return 0.0

    def _save_results(self):
        """Save benchmark results to file."""
        try:
            results_dir = Path(self.config.results_path)
            results_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"model_comparison_{timestamp}.json"

            # Convert results to serializable format
            results_data = {
                "timestamp": timestamp,
                "config": self.config.__dict__,
                "results": [],
            }

            for result in self.results:
                results_data["results"].append(result.__dict__)

            # Save to file
            with open(results_file, "w") as f:
                json.dump(results_data, f, indent=2, default=str)

            logger.info(f"Benchmark results saved to {results_file}")

        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")

    def _generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report."""
        if not self.results:
            return "No benchmark results available."

        report = []
        report.append("🏆 Model Comparison Report")
        report.append("=" * 60)
        report.append("")

        # Performance comparison
        report.append("📊 Performance Metrics:")
        report.append("-" * 40)
        report.append(
            f"{'Model':<20} {'AUC-ROC':<8} {'F1-Score':<8} {'Accuracy':<8}"
        )
        report.append("-" * 40)

        for result in self.results:
            if result.success:
                report.append(
                    f"{result.model_name:<20} {result.auc_roc:<8.4f} "
                    f"{result.f1_score:<8.4f} {result.accuracy:<8.4f}"
                )

        # Efficiency comparison
        report.append("")
        report.append("⚡ Efficiency Metrics:")
        report.append("-" * 50)
        report.append(
            f"{'Model':<20} {'Train Time':<12} {'Inference':<12} {'Size (MB)':<10}"
        )
        report.append("-" * 50)

        for result in self.results:
            if result.success:
                train_time = f"{result.training_time_seconds:.2f}s"
                inference_time = f"{result.inference_time_ms:.2f}ms"
                size = f"{result.model_size_mb:.2f}"
                report.append(
                    f"{result.model_name:<20} {train_time:<12} "
                    f"{inference_time:<12} {size:<10}"
                )

        # Model characteristics
        report.append("")
        report.append("🔍 Model Characteristics:")
        report.append("-" * 40)
        report.append(f"{'Model':<20} {'Parameters':<12} {'Type':<15}")
        report.append("-" * 40)

        for result in self.results:
            if result.success:
                params = f"{result.num_parameters:,}"
                report.append(
                    f"{result.model_name:<20} {params:<12} {result.model_type:<15}"
                )

        # Best performing models
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            report.append("")
            report.append("🥇 Best Performing Models:")
            report.append("-" * 30)

            best_auc = max(successful_results, key=lambda r: r.auc_roc)
            fastest_train = min(
                successful_results, key=lambda r: r.training_time_seconds
            )
            fastest_inference = min(
                successful_results, key=lambda r: r.inference_time_ms
            )
            smallest = min(successful_results, key=lambda r: r.model_size_mb)

            report.append(
                f"Best AUC-ROC: {best_auc.model_name} ({best_auc.auc_roc:.4f})"
            )
            report.append(
                f"Fastest Training: {fastest_train.model_name} ({fastest_train.training_time_seconds:.2f}s)"
            )
            report.append(
                f"Fastest Inference: {fastest_inference.model_name} ({fastest_inference.inference_time_ms:.2f}ms)"
            )
            report.append(
                f"Smallest Model: {smallest.model_name} ({smallest.model_size_mb:.2f} MB)"
            )

        # Recommendations
        report.append("")
        report.append("💡 Recommendations:")
        report.append("-" * 20)

        if successful_results:
            # Find best overall model based on multiple criteria
            best_overall = max(
                successful_results,
                key=lambda r: r.auc_roc * 0.4
                + (1 / max(r.training_time_seconds, 0.1)) * 0.3
                + (1 / max(r.inference_time_ms, 0.1)) * 0.3,
            )

            report.append(
                f"• For production deployment: {best_overall.model_name}"
            )
            report.append(
                f"  (Best balance of accuracy, speed, and efficiency)"
            )

            if len(successful_results) > 1:
                lgb_results = [
                    r
                    for r in successful_results
                    if r.model_type == "gradient_boosting"
                ]
                dnn_results = [
                    r
                    for r in successful_results
                    if r.model_type == "deep_learning"
                ]

                if lgb_results and dnn_results:
                    lgb = lgb_results[0]
                    dnn = dnn_results[0]

                    if lgb.training_time_seconds < dnn.training_time_seconds:
                        report.append(
                            f"• For rapid prototyping: {lgb.model_name}"
                        )
                        report.append(
                            f"  (Faster training: {lgb.training_time_seconds:.2f}s vs {dnn.training_time_seconds:.2f}s)"
                        )

                    if dnn.auc_roc > lgb.auc_roc:
                        report.append(
                            f"• For maximum accuracy: {dnn.model_name}"
                        )
                        report.append(
                            f"  (Higher AUC: {dnn.auc_roc:.4f} vs {lgb.auc_roc:.4f})"
                        )

        return "\n".join(report)


# Convenience functions
def compare_lightgbm_vs_dnn(
    X: pd.DataFrame, y: pd.Series, config: Optional[ComparisonConfig] = None
) -> List[ModelBenchmarkResult]:
    """Convenience function to compare LightGBM vs DNN."""
    if config is None:
        config = ComparisonConfig()

    comparison = ModelComparison(config)
    return comparison.compare_models(X, y)


def quick_model_comparison(X: pd.DataFrame, y: pd.Series) -> str:
    """Quick model comparison with default settings."""
    config = ComparisonConfig(
        include_lightgbm=True,
        include_dnn=True,
        save_results=False,
        generate_report=True,
    )

    comparison = ModelComparison(config)
    comparison.compare_models(X, y)

    return "Model comparison completed. Check the output above for results."
