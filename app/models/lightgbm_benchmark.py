"""
LightGBM benchmarking and performance comparison utilities.
Provides comprehensive benchmarking against neural network models.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from ..data.experiment_tracking import ExperimentTracker

# Core imports
from .lightgbm_model import LightGBMConfig, LightGBMResult, LightGBMTrainer

logger = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""

    # Model configurations
    lightgbm_configs: List[LightGBMConfig]

    # Benchmarking parameters
    test_sizes: List[float] = None
    cv_folds: int = 5
    n_runs: int = 3

    # Performance metrics
    track_energy: bool = True
    track_memory: bool = True
    track_inference_time: bool = True

    # Output
    save_results: bool = True
    results_path: str = "benchmarks/lightgbm"

    def __post_init__(self):
        if self.test_sizes is None:
            self.test_sizes = [0.2]


@dataclass
class BenchmarkResult:
    """Results from benchmarking experiments."""

    config_name: str
    model_type: str = "lightgbm"

    # Performance metrics
    accuracy_scores: List[float] = None
    precision_scores: List[float] = None
    recall_scores: List[float] = None
    f1_scores: List[float] = None
    auc_scores: List[float] = None

    # Timing metrics
    training_times: List[float] = None
    inference_times: List[float] = None

    # Resource metrics
    memory_usage_mb: List[float] = None
    energy_consumption_kwh: List[float] = None

    # Model characteristics
    model_size_mb: float = 0.0
    num_parameters: int = 0
    num_features: int = 0

    # Statistical summary
    mean_metrics: Dict[str, float] = None
    std_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.accuracy_scores is None:
            self.accuracy_scores = []
        if self.precision_scores is None:
            self.precision_scores = []
        if self.recall_scores is None:
            self.recall_scores = []
        if self.f1_scores is None:
            self.f1_scores = []
        if self.auc_scores is None:
            self.auc_scores = []
        if self.training_times is None:
            self.training_times = []
        if self.inference_times is None:
            self.inference_times = []
        if self.memory_usage_mb is None:
            self.memory_usage_mb = []
        if self.energy_consumption_kwh is None:
            self.energy_consumption_kwh = []

    def calculate_summary_stats(self):
        """Calculate mean and standard deviation for all metrics."""
        metrics = {
            "accuracy": self.accuracy_scores,
            "precision": self.precision_scores,
            "recall": self.recall_scores,
            "f1_score": self.f1_scores,
            "auc_roc": self.auc_scores,
            "training_time": self.training_times,
            "inference_time": self.inference_times,
            "memory_usage": self.memory_usage_mb,
            "energy_consumption": self.energy_consumption_kwh,
        }

        self.mean_metrics = {}
        self.std_metrics = {}

        for metric_name, values in metrics.items():
            if values and len(values) > 0:
                self.mean_metrics[metric_name] = np.mean(values)
                self.std_metrics[metric_name] = np.std(values)
            else:
                self.mean_metrics[metric_name] = 0.0
                self.std_metrics[metric_name] = 0.0


class LightGBMBenchmark:
    """Comprehensive benchmarking system for LightGBM models."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.experiment_tracker = ExperimentTracker()
        self.results: List[BenchmarkResult] = []

    def run_benchmark(self, X: pd.DataFrame, y: pd.Series) -> List[BenchmarkResult]:
        """Run comprehensive benchmark on dataset."""
        logger.info(
            f"Starting LightGBM benchmark with {len(self.config.lightgbm_configs)} configurations"
        )

        self.results = []

        for i, lgb_config in enumerate(self.config.lightgbm_configs):
            config_name = f"lightgbm_config_{i}"
            logger.info(f"Benchmarking configuration: {config_name}")

            result = self._benchmark_single_config(X, y, lgb_config, config_name)
            self.results.append(result)

        # Save results if requested
        if self.config.save_results:
            self._save_benchmark_results()

        logger.info("Benchmark completed successfully")
        return self.results

    def _benchmark_single_config(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        lgb_config: LightGBMConfig,
        config_name: str,
    ) -> BenchmarkResult:
        """Benchmark a single LightGBM configuration."""
        result = BenchmarkResult(config_name=config_name)

        # Track experiment
        experiment_id = self.experiment_tracker.start_experiment(
            experiment_name=f"lightgbm_benchmark_{config_name}",
            tags={"model_type": "lightgbm"},
        )

        # Log parameters
        self.experiment_tracker.log_parameters(lgb_config.__dict__)

        try:
            # Run multiple trials
            for run in range(self.config.n_runs):
                logger.info(f"Running trial {run + 1}/{self.config.n_runs}")

                # Run single trial
                trial_result = self._run_single_trial(X, y, lgb_config, run)

                # Collect metrics
                if trial_result.success:
                    result.accuracy_scores.append(
                        trial_result.test_metrics.get("accuracy", 0.0)
                    )
                    result.precision_scores.append(
                        trial_result.test_metrics.get("precision", 0.0)
                    )
                    result.recall_scores.append(
                        trial_result.test_metrics.get("recall", 0.0)
                    )
                    result.f1_scores.append(
                        trial_result.test_metrics.get("f1_score", 0.0)
                    )
                    result.auc_scores.append(
                        trial_result.test_metrics.get("roc_auc", 0.0)
                    )
                    result.training_times.append(trial_result.training_time_seconds)

                    # Measure inference time
                    inference_time = self._measure_inference_time(
                        trial_result.model, X.head(100)
                    )
                    result.inference_times.append(inference_time)

                    # Estimate model size (LightGBM models are typically small)
                    if result.model_size_mb == 0.0:
                        result.model_size_mb = self._estimate_model_size(
                            trial_result.model
                        )
                        result.num_features = len(X.columns)
                        result.num_parameters = self._estimate_num_parameters(
                            trial_result.model
                        )

            # Calculate summary statistics
            result.calculate_summary_stats()

            # Log experiment results
            self.experiment_tracker.log_metrics(result.mean_metrics)

            self.experiment_tracker.end_experiment("FINISHED")

        except Exception as e:
            logger.error(f"Benchmark failed for {config_name}: {e}")
            self.experiment_tracker.end_experiment("FAILED")

        return result

    def _run_single_trial(
        self, X: pd.DataFrame, y: pd.Series, lgb_config: LightGBMConfig, run_id: int
    ) -> LightGBMResult:
        """Run a single training trial."""
        # Add run-specific randomization
        lgb_config_copy = LightGBMConfig(**lgb_config.__dict__)

        # Create trainer and run
        trainer = LightGBMTrainer(lgb_config_copy)
        result = trainer.train_and_evaluate(X, y)

        return result

    def _measure_inference_time(self, model, X_sample: pd.DataFrame) -> float:
        """Measure inference time per sample."""
        if not model or not model.is_trained:
            return 0.0

        try:
            # Warm up
            _ = model.predict_proba(X_sample.head(10))

            # Measure time for batch prediction
            start_time = time.time()
            _ = model.predict_proba(X_sample)
            end_time = time.time()

            # Calculate time per sample in milliseconds
            time_per_sample = ((end_time - start_time) / len(X_sample)) * 1000
            return time_per_sample

        except Exception as e:
            logger.warning(f"Could not measure inference time: {e}")
            return 0.0

    def _estimate_model_size(self, model) -> float:
        """Estimate model size in MB."""
        if not model or not model.is_trained:
            return 0.0

        try:
            # Save model to temporary file to get size
            temp_path = Path("temp_model_size_check.txt")
            model.model.save_model(str(temp_path))

            size_bytes = temp_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)

            # Clean up
            temp_path.unlink(missing_ok=True)

            return size_mb

        except Exception as e:
            logger.warning(f"Could not estimate model size: {e}")
            return 0.0

    def _estimate_num_parameters(self, model) -> int:
        """Estimate number of parameters in LightGBM model."""
        if not model or not model.is_trained:
            return 0

        try:
            # LightGBM parameters are roughly: num_trees * num_leaves * num_features
            num_trees = model.model.num_trees()
            num_features = model.model.num_feature()

            # Rough estimation - each tree has internal nodes and leaf values
            params_per_tree = (
                model.config.num_leaves * 2
            )  # Internal nodes + leaf values
            total_params = num_trees * params_per_tree

            return total_params

        except Exception as e:
            logger.warning(f"Could not estimate parameters: {e}")
            return 0

    def _save_benchmark_results(self):
        """Save benchmark results to file."""
        try:
            results_dir = Path(self.config.results_path)
            results_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"lightgbm_benchmark_{timestamp}.json"

            # Convert results to serializable format
            results_data = {
                "benchmark_config": self.config.__dict__,
                "timestamp": timestamp,
                "results": [],
            }

            for result in self.results:
                result_dict = {
                    "config_name": result.config_name,
                    "model_type": result.model_type,
                    "mean_metrics": result.mean_metrics,
                    "std_metrics": result.std_metrics,
                    "model_size_mb": result.model_size_mb,
                    "num_parameters": result.num_parameters,
                    "num_features": result.num_features,
                    "raw_scores": {
                        "accuracy": result.accuracy_scores,
                        "precision": result.precision_scores,
                        "recall": result.recall_scores,
                        "f1_score": result.f1_scores,
                        "auc_roc": result.auc_scores,
                        "training_times": result.training_times,
                        "inference_times": result.inference_times,
                    },
                }
                results_data["results"].append(result_dict)

            # Save to file
            with open(results_file, "w") as f:
                json.dump(results_data, f, indent=2, default=str)

            logger.info(f"Benchmark results saved to {results_file}")

        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")

    def generate_comparison_report(self) -> str:
        """Generate a comparison report of all benchmarked configurations."""
        if not self.results:
            return "No benchmark results available."

        report = []
        report.append("LightGBM Benchmark Comparison Report")
        report.append("=" * 50)
        report.append("")

        # Summary table
        report.append("Performance Summary:")
        report.append("-" * 30)

        for result in self.results:
            report.append(f"\nConfiguration: {result.config_name}")
            report.append(
                f"  AUC-ROC: {result.mean_metrics.get('auc_roc', 0):.4f} ± {result.std_metrics.get('auc_roc', 0):.4f}"
            )
            report.append(
                f"  F1-Score: {result.mean_metrics.get('f1_score', 0):.4f} ± {result.std_metrics.get('f1_score', 0):.4f}"
            )
            report.append(
                f"  Training Time: {result.mean_metrics.get('training_time', 0):.2f}s ± {result.std_metrics.get('training_time', 0):.2f}s"
            )
            report.append(
                f"  Inference Time: {result.mean_metrics.get('inference_time', 0):.2f}ms ± {result.std_metrics.get('inference_time', 0):.2f}ms"
            )
            report.append(f"  Model Size: {result.model_size_mb:.2f} MB")
            report.append(f"  Parameters: {result.num_parameters:,}")

        # Best performing configuration
        best_auc_result = max(
            self.results, key=lambda r: r.mean_metrics.get("auc_roc", 0)
        )
        fastest_result = min(
            self.results,
            key=lambda r: r.mean_metrics.get("training_time", float("inf")),
        )
        smallest_result = min(self.results, key=lambda r: r.model_size_mb)

        report.append("\n" + "=" * 50)
        report.append("Best Configurations:")
        report.append(
            f"  Best AUC-ROC: {best_auc_result.config_name} ({best_auc_result.mean_metrics.get('auc_roc', 0):.4f})"
        )
        report.append(
            f"  Fastest Training: {fastest_result.config_name} ({fastest_result.mean_metrics.get('training_time', 0):.2f}s)"
        )
        report.append(
            f"  Smallest Model: {smallest_result.config_name} ({smallest_result.model_size_mb:.2f} MB)"
        )

        return "\n".join(report)


# Factory functions for common benchmark configurations
def create_fast_benchmark_config() -> BenchmarkConfig:
    """Create a fast benchmark configuration for testing."""
    from .lightgbm_model import get_default_lightgbm_config, get_fast_lightgbm_config

    configs = [get_fast_lightgbm_config(), get_default_lightgbm_config()]

    # Disable hyperopt for fast benchmarking
    for config in configs:
        config.enable_hyperopt = False
        config.num_boost_round = 100

    return BenchmarkConfig(lightgbm_configs=configs, n_runs=2, cv_folds=3)


def create_comprehensive_benchmark_config() -> BenchmarkConfig:
    """Create a comprehensive benchmark configuration."""
    from .lightgbm_model import (
        get_default_lightgbm_config,
        get_fast_lightgbm_config,
        get_optimized_lightgbm_config,
    )

    configs = [
        get_fast_lightgbm_config(),
        get_default_lightgbm_config(),
        get_optimized_lightgbm_config(),
    ]

    return BenchmarkConfig(lightgbm_configs=configs, n_runs=5, cv_folds=5)


def run_lightgbm_benchmark(
    X: pd.DataFrame, y: pd.Series, config: Optional[BenchmarkConfig] = None
) -> List[BenchmarkResult]:
    """Convenience function to run LightGBM benchmark."""
    if config is None:
        config = create_fast_benchmark_config()

    benchmark = LightGBMBenchmark(config)
    return benchmark.run_benchmark(X, y)
