"""
Sustainable AI Benchmarking Framework with Industry Comparisons.

This module implements comprehensive benchmarking of AI models against sustainability
metrics, comparing performance with industry standards and providing actionable
insights for improving carbon efficiency and environmental impact.
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import logging
import requests
from abc import ABC, abstractmethod

try:
    from ..core.logging import get_logger, get_audit_logger
    from .carbon_calculator import CarbonCalculator, CarbonFootprint
    from .energy_tracker import EnergyTracker, EnergyReport
    from ..models.dnn_model import DNNModel
    from ..models.lstm_model import LSTMModel
    from ..models.gnn_model import GNNModel
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_logger, get_audit_logger
    from sustainability.carbon_calculator import CarbonCalculator, CarbonFootprint
    from sustainability.energy_tracker import EnergyTracker, EnergyReport
    from models.dnn_model import DNNModel
    from models.lstm_model import LSTMModel
    from models.gnn_model import GNNModel

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class BenchmarkCategory(Enum):
    """Categories of AI sustainability benchmarks."""

    CARBON_EFFICIENCY = "carbon_efficiency"
    ENERGY_EFFICIENCY = "energy_efficiency"
    MODEL_EFFICIENCY = "model_efficiency"
    TRAINING_EFFICIENCY = "training_efficiency"
    INFERENCE_EFFICIENCY = "inference_efficiency"
    OVERALL_SUSTAINABILITY = "overall_sustainability"


class IndustrySector(Enum):
    """Industry sectors for benchmarking."""

    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    TECHNOLOGY = "technology"
    AUTOMOTIVE = "automotive"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    ENERGY = "energy"
    TRANSPORTATION = "transportation"


@dataclass
class BenchmarkMetric:
    """Individual benchmark metric."""

    metric_name: str
    metric_value: float
    unit: str
    category: BenchmarkCategory
    is_higher_better: bool = True
    industry_average: Optional[float] = None
    industry_percentile: Optional[float] = None
    best_practice_threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "unit": self.unit,
            "category": self.category.value,
            "is_higher_better": self.is_higher_better,
            "industry_average": self.industry_average,
            "industry_percentile": self.industry_percentile,
            "best_practice_threshold": self.best_practice_threshold,
        }


@dataclass
class ModelBenchmarkResult:
    """Benchmark result for a specific model."""

    model_id: str
    model_name: str
    model_type: str
    benchmark_timestamp: datetime

    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float

    # Sustainability metrics
    carbon_footprint_kg: float
    energy_consumption_kwh: float
    training_time_seconds: float
    inference_latency_ms: float
    model_size_mb: float

    # Efficiency metrics
    carbon_efficiency: float  # Performance per CO2
    energy_efficiency: float  # Performance per kWh
    time_efficiency: float  # Performance per second
    size_efficiency: float  # Performance per MB

    # Industry comparison
    industry_sector: IndustrySector
    benchmark_metrics: List[BenchmarkMetric] = field(default_factory=list)

    # Overall scores
    sustainability_score: float = 0.0
    performance_score: float = 0.0
    efficiency_score: float = 0.0
    overall_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "benchmark_timestamp": self.benchmark_timestamp.isoformat(),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "carbon_footprint_kg": self.carbon_footprint_kg,
            "energy_consumption_kwh": self.energy_consumption_kwh,
            "training_time_seconds": self.training_time_seconds,
            "inference_latency_ms": self.inference_latency_ms,
            "model_size_mb": self.model_size_mb,
            "carbon_efficiency": self.carbon_efficiency,
            "energy_efficiency": self.energy_efficiency,
            "time_efficiency": self.time_efficiency,
            "size_efficiency": self.size_efficiency,
            "industry_sector": self.industry_sector.value,
            "benchmark_metrics": [m.to_dict() for m in self.benchmark_metrics],
            "sustainability_score": self.sustainability_score,
            "performance_score": self.performance_score,
            "efficiency_score": self.efficiency_score,
            "overall_score": self.overall_score,
        }


@dataclass
class IndustryBenchmarkData:
    """Industry benchmark data for comparison."""

    sector: IndustrySector
    benchmark_metrics: Dict[
        str, Dict[str, float]
    ]  # metric_name -> {avg, p25, p75, p90, p95}
    last_updated: datetime
    data_source: str
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sector": self.sector.value,
            "benchmark_metrics": self.benchmark_metrics,
            "last_updated": self.last_updated.isoformat(),
            "data_source": self.data_source,
            "sample_size": self.sample_size,
        }


class IndustryBenchmarkProvider(ABC):
    """Abstract base class for industry benchmark data providers."""

    @abstractmethod
    def get_benchmark_data(self, sector: IndustrySector) -> IndustryBenchmarkData:
        """Get benchmark data for a specific industry sector."""
        pass

    @abstractmethod
    def update_benchmark_data(self, sector: IndustrySector) -> bool:
        """Update benchmark data for a specific sector."""
        pass


class MockIndustryBenchmarkProvider(IndustryBenchmarkProvider):
    """Mock implementation of industry benchmark provider."""

    def __init__(self):
        self.benchmark_data = self._initialize_mock_data()

    def _initialize_mock_data(self) -> Dict[IndustrySector, IndustryBenchmarkData]:
        """Initialize mock industry benchmark data."""

        # Finance sector benchmarks
        finance_benchmarks = {
            "carbon_footprint_kg": {
                "avg": 0.15,
                "p25": 0.08,
                "p75": 0.22,
                "p90": 0.35,
                "p95": 0.45,
            },
            "energy_consumption_kwh": {
                "avg": 0.08,
                "p25": 0.04,
                "p75": 0.12,
                "p90": 0.18,
                "p95": 0.25,
            },
            "training_time_seconds": {
                "avg": 1200,
                "p25": 600,
                "p75": 1800,
                "p90": 3000,
                "p95": 4500,
            },
            "inference_latency_ms": {
                "avg": 45,
                "p25": 25,
                "p75": 65,
                "p90": 100,
                "p95": 150,
            },
            "model_size_mb": {"avg": 25, "p25": 10, "p75": 40, "p90": 80, "p95": 120},
            "carbon_efficiency": {
                "avg": 6.5,
                "p25": 4.0,
                "p75": 9.0,
                "p90": 12.0,
                "p95": 15.0,
            },
            "energy_efficiency": {
                "avg": 12.0,
                "p25": 8.0,
                "p75": 16.0,
                "p90": 22.0,
                "p95": 28.0,
            },
            "accuracy": {
                "avg": 0.89,
                "p25": 0.85,
                "p75": 0.93,
                "p90": 0.96,
                "p95": 0.98,
            },
        }

        # Technology sector benchmarks (typically more efficient)
        technology_benchmarks = {
            "carbon_footprint_kg": {
                "avg": 0.08,
                "p25": 0.04,
                "p75": 0.12,
                "p90": 0.18,
                "p95": 0.25,
            },
            "energy_consumption_kwh": {
                "avg": 0.04,
                "p25": 0.02,
                "p75": 0.06,
                "p90": 0.09,
                "p95": 0.12,
            },
            "training_time_seconds": {
                "avg": 800,
                "p25": 400,
                "p75": 1200,
                "p90": 2000,
                "p95": 3000,
            },
            "inference_latency_ms": {
                "avg": 25,
                "p25": 15,
                "p75": 35,
                "p90": 50,
                "p95": 75,
            },
            "model_size_mb": {"avg": 15, "p25": 5, "p75": 25, "p90": 50, "p95": 80},
            "carbon_efficiency": {
                "avg": 12.0,
                "p25": 8.0,
                "p75": 16.0,
                "p90": 22.0,
                "p95": 28.0,
            },
            "energy_efficiency": {
                "avg": 22.0,
                "p25": 15.0,
                "p75": 30.0,
                "p90": 40.0,
                "p95": 50.0,
            },
            "accuracy": {
                "avg": 0.92,
                "p25": 0.88,
                "p75": 0.95,
                "p90": 0.97,
                "p95": 0.99,
            },
        }

        # Healthcare sector benchmarks (accuracy-focused)
        healthcare_benchmarks = {
            "carbon_footprint_kg": {
                "avg": 0.25,
                "p25": 0.15,
                "p75": 0.35,
                "p90": 0.50,
                "p95": 0.70,
            },
            "energy_consumption_kwh": {
                "avg": 0.12,
                "p25": 0.08,
                "p75": 0.16,
                "p90": 0.25,
                "p95": 0.35,
            },
            "training_time_seconds": {
                "avg": 2000,
                "p25": 1000,
                "p75": 3000,
                "p90": 5000,
                "p95": 8000,
            },
            "inference_latency_ms": {
                "avg": 60,
                "p25": 35,
                "p75": 85,
                "p90": 120,
                "p95": 180,
            },
            "model_size_mb": {"avg": 50, "p25": 20, "p75": 80, "p90": 150, "p95": 250},
            "carbon_efficiency": {
                "avg": 3.5,
                "p25": 2.0,
                "p75": 5.0,
                "p90": 7.0,
                "p95": 9.0,
            },
            "energy_efficiency": {
                "avg": 7.5,
                "p25": 4.0,
                "p75": 11.0,
                "p90": 15.0,
                "p95": 20.0,
            },
            "accuracy": {
                "avg": 0.95,
                "p25": 0.92,
                "p75": 0.97,
                "p90": 0.98,
                "p95": 0.99,
            },
        }

        return {
            IndustrySector.FINANCE: IndustryBenchmarkData(
                sector=IndustrySector.FINANCE,
                benchmark_metrics=finance_benchmarks,
                last_updated=datetime.now(),
                data_source="Sustainable AI Consortium 2024",
                sample_size=150,
            ),
            IndustrySector.TECHNOLOGY: IndustryBenchmarkData(
                sector=IndustrySector.TECHNOLOGY,
                benchmark_metrics=technology_benchmarks,
                last_updated=datetime.now(),
                data_source="Sustainable AI Consortium 2024",
                sample_size=200,
            ),
            IndustrySector.HEALTHCARE: IndustryBenchmarkData(
                sector=IndustrySector.HEALTHCARE,
                benchmark_metrics=healthcare_benchmarks,
                last_updated=datetime.now(),
                data_source="Sustainable AI Consortium 2024",
                sample_size=100,
            ),
        }

    def get_benchmark_data(self, sector: IndustrySector) -> IndustryBenchmarkData:
        """Get benchmark data for a specific industry sector."""
        return self.benchmark_data.get(sector)

    def update_benchmark_data(self, sector: IndustrySector) -> bool:
        """Update benchmark data for a specific sector."""
        # In a real implementation, this would fetch updated data from external sources
        return True


class SustainableAIBenchmark:
    """Main sustainable AI benchmarking system."""

    def __init__(self, industry_provider: Optional[IndustryBenchmarkProvider] = None):
        self.industry_provider = industry_provider or MockIndustryBenchmarkProvider()
        self.carbon_calculator = CarbonCalculator()
        self.energy_tracker = EnergyTracker()

        # Benchmark results storage
        self.benchmark_results: List[ModelBenchmarkResult] = []
        self.results_dir = Path("sustainable_ai_benchmarks")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Sustainable AI benchmark system initialized")

    def benchmark_model(
        self,
        model,
        model_name: str,
        model_type: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        industry_sector: IndustrySector = IndustrySector.FINANCE,
        training_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelBenchmarkResult:
        """Benchmark a model against sustainability metrics."""

        logger.info(f"Benchmarking model: {model_name}")

        # Start energy tracking for inference
        energy_experiment_id = self.energy_tracker.start_tracking(
            f"benchmark_{model_name}"
        )

        try:
            # Measure inference performance
            start_time = time.time()

            # Get model predictions
            if hasattr(model, "predict_proba"):
                predictions = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model.predict_proba(X_test), "shape")
                    and len(model.predict_proba(X_test).shape) > 1
                    else model.predict_proba(X_test)
                )
            else:
                predictions = model.predict(X_test)

            inference_time = time.time() - start_time

            # Stop energy tracking
            energy_report = self.energy_tracker.stop_tracking()

            # Calculate performance metrics
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
            )

            binary_predictions = (
                (predictions > 0.5).astype(int)
                if len(predictions.shape) == 1
                else (predictions[:, 1] > 0.5).astype(int)
            )

            accuracy = accuracy_score(y_test, binary_predictions)
            precision = precision_score(y_test, binary_predictions, average="weighted")
            recall = recall_score(y_test, binary_predictions, average="weighted")
            f1 = f1_score(y_test, binary_predictions, average="weighted")

            try:
                roc_auc = roc_auc_score(
                    y_test,
                    predictions if len(predictions.shape) == 1 else predictions[:, 1],
                )
            except:
                roc_auc = 0.5

            # Calculate carbon footprint
            carbon_footprint = self.carbon_calculator.calculate_carbon_footprint(
                energy_report, region="US"
            )

            # Calculate model size (estimate)
            model_size_mb = self._estimate_model_size(model)

            # Calculate efficiency metrics
            carbon_efficiency = (
                roc_auc / carbon_footprint.total_emissions_kg
                if carbon_footprint.total_emissions_kg > 0
                else 0
            )
            energy_efficiency = (
                roc_auc / energy_report.total_energy_kwh
                if energy_report.total_energy_kwh > 0
                else 0
            )
            time_efficiency = roc_auc / inference_time if inference_time > 0 else 0
            size_efficiency = roc_auc / model_size_mb if model_size_mb > 0 else 0

            # Get industry benchmarks
            industry_data = self.industry_provider.get_benchmark_data(industry_sector)

            # Create benchmark metrics
            benchmark_metrics = self._create_benchmark_metrics(
                carbon_footprint.total_emissions_kg,
                energy_report.total_energy_kwh,
                (
                    training_metadata.get("training_time_seconds", 0)
                    if training_metadata
                    else 0
                ),
                inference_time * 1000,  # Convert to ms
                model_size_mb,
                carbon_efficiency,
                energy_efficiency,
                roc_auc,
                industry_data,
            )

            # Calculate overall scores
            sustainability_score = self._calculate_sustainability_score(
                benchmark_metrics
            )
            performance_score = self._calculate_performance_score(
                accuracy, precision, recall, f1, roc_auc
            )
            efficiency_score = self._calculate_efficiency_score(benchmark_metrics)
            overall_score = (
                sustainability_score * 0.4
                + performance_score * 0.4
                + efficiency_score * 0.2
            )

            # Create benchmark result
            result = ModelBenchmarkResult(
                model_id=f"benchmark_{int(time.time())}_{model_name}",
                model_name=model_name,
                model_type=model_type,
                benchmark_timestamp=datetime.now(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                carbon_footprint_kg=carbon_footprint.total_emissions_kg,
                energy_consumption_kwh=energy_report.total_energy_kwh,
                training_time_seconds=(
                    training_metadata.get("training_time_seconds", 0)
                    if training_metadata
                    else 0
                ),
                inference_latency_ms=inference_time * 1000,
                model_size_mb=model_size_mb,
                carbon_efficiency=carbon_efficiency,
                energy_efficiency=energy_efficiency,
                time_efficiency=time_efficiency,
                size_efficiency=size_efficiency,
                industry_sector=industry_sector,
                benchmark_metrics=benchmark_metrics,
                sustainability_score=sustainability_score,
                performance_score=performance_score,
                efficiency_score=efficiency_score,
                overall_score=overall_score,
            )

            # Store result
            self.benchmark_results.append(result)
            self._save_benchmark_result(result)

            logger.info(
                f"Benchmark completed for {model_name}: Overall Score = {overall_score:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")
            raise

    def _estimate_model_size(self, model) -> float:
        """Estimate model size in MB."""

        try:
            # For PyTorch models
            if hasattr(model, "state_dict"):
                total_params = sum(p.numel() for p in model.parameters())
                # Assume 4 bytes per parameter (float32)
                size_bytes = total_params * 4
                return size_bytes / (1024 * 1024)  # Convert to MB

            # For scikit-learn models
            elif hasattr(model, "coef_"):
                # Rough estimate based on coefficients
                if hasattr(model.coef_, "shape"):
                    size_bytes = model.coef_.nbytes
                else:
                    size_bytes = len(model.coef_) * 4
                return size_bytes / (1024 * 1024)

            else:
                # Default estimate
                return 10.0  # 10 MB default

        except Exception as e:
            logger.warning(f"Could not estimate model size: {e}")
            return 10.0  # Default fallback

    def _create_benchmark_metrics(
        self,
        carbon_kg: float,
        energy_kwh: float,
        training_time: float,
        inference_latency: float,
        model_size_mb: float,
        carbon_efficiency: float,
        energy_efficiency: float,
        roc_auc: float,
        industry_data: IndustryBenchmarkData,
    ) -> List[BenchmarkMetric]:
        """Create benchmark metrics with industry comparisons."""

        metrics = []

        # Carbon footprint metric
        carbon_metric = BenchmarkMetric(
            metric_name="carbon_footprint_kg",
            metric_value=carbon_kg,
            unit="kg CO2e",
            category=BenchmarkCategory.CARBON_EFFICIENCY,
            is_higher_better=False,
            industry_average=industry_data.benchmark_metrics.get(
                "carbon_footprint_kg", {}
            ).get("avg"),
            industry_percentile=self._calculate_percentile(
                carbon_kg,
                industry_data.benchmark_metrics.get("carbon_footprint_kg", {}),
            ),
            best_practice_threshold=industry_data.benchmark_metrics.get(
                "carbon_footprint_kg", {}
            ).get("p25"),
        )
        metrics.append(carbon_metric)

        # Energy consumption metric
        energy_metric = BenchmarkMetric(
            metric_name="energy_consumption_kwh",
            metric_value=energy_kwh,
            unit="kWh",
            category=BenchmarkCategory.ENERGY_EFFICIENCY,
            is_higher_better=False,
            industry_average=industry_data.benchmark_metrics.get(
                "energy_consumption_kwh", {}
            ).get("avg"),
            industry_percentile=self._calculate_percentile(
                energy_kwh,
                industry_data.benchmark_metrics.get("energy_consumption_kwh", {}),
            ),
            best_practice_threshold=industry_data.benchmark_metrics.get(
                "energy_consumption_kwh", {}
            ).get("p25"),
        )
        metrics.append(energy_metric)

        # Carbon efficiency metric
        carbon_eff_metric = BenchmarkMetric(
            metric_name="carbon_efficiency",
            metric_value=carbon_efficiency,
            unit="AUC/kg CO2e",
            category=BenchmarkCategory.CARBON_EFFICIENCY,
            is_higher_better=True,
            industry_average=industry_data.benchmark_metrics.get(
                "carbon_efficiency", {}
            ).get("avg"),
            industry_percentile=self._calculate_percentile(
                carbon_efficiency,
                industry_data.benchmark_metrics.get("carbon_efficiency", {}),
            ),
            best_practice_threshold=industry_data.benchmark_metrics.get(
                "carbon_efficiency", {}
            ).get("p75"),
        )
        metrics.append(carbon_eff_metric)

        # Energy efficiency metric
        energy_eff_metric = BenchmarkMetric(
            metric_name="energy_efficiency",
            metric_value=energy_efficiency,
            unit="AUC/kWh",
            category=BenchmarkCategory.ENERGY_EFFICIENCY,
            is_higher_better=True,
            industry_average=industry_data.benchmark_metrics.get(
                "energy_efficiency", {}
            ).get("avg"),
            industry_percentile=self._calculate_percentile(
                energy_efficiency,
                industry_data.benchmark_metrics.get("energy_efficiency", {}),
            ),
            best_practice_threshold=industry_data.benchmark_metrics.get(
                "energy_efficiency", {}
            ).get("p75"),
        )
        metrics.append(energy_eff_metric)

        # Model accuracy metric
        accuracy_metric = BenchmarkMetric(
            metric_name="accuracy",
            metric_value=roc_auc,
            unit="AUC",
            category=BenchmarkCategory.MODEL_EFFICIENCY,
            is_higher_better=True,
            industry_average=industry_data.benchmark_metrics.get("accuracy", {}).get(
                "avg"
            ),
            industry_percentile=self._calculate_percentile(
                roc_auc, industry_data.benchmark_metrics.get("accuracy", {})
            ),
            best_practice_threshold=industry_data.benchmark_metrics.get(
                "accuracy", {}
            ).get("p75"),
        )
        metrics.append(accuracy_metric)

        return metrics

    def _calculate_percentile(
        self, value: float, industry_stats: Dict[str, float]
    ) -> Optional[float]:
        """Calculate percentile rank compared to industry."""

        if not industry_stats or len(industry_stats) < 4:
            return None

        # Simple percentile calculation based on quartiles
        p25 = industry_stats.get("p25", 0)
        p75 = industry_stats.get("p75", 0)
        avg = industry_stats.get("avg", 0)

        if value <= p25:
            return 25.0
        elif value <= avg:
            return 50.0
        elif value <= p75:
            return 75.0
        else:
            return 90.0

    def _calculate_sustainability_score(self, metrics: List[BenchmarkMetric]) -> float:
        """Calculate overall sustainability score."""

        sustainability_metrics = [
            m
            for m in metrics
            if m.category
            in [
                BenchmarkCategory.CARBON_EFFICIENCY,
                BenchmarkCategory.ENERGY_EFFICIENCY,
            ]
        ]

        if not sustainability_metrics:
            return 0.0

        total_score = 0.0
        for metric in sustainability_metrics:
            if metric.industry_percentile is not None:
                # Convert percentile to score (0-100)
                score = metric.industry_percentile
                if not metric.is_higher_better:
                    score = 100 - score  # Invert for lower-is-better metrics
                total_score += score

        return total_score / len(sustainability_metrics)

    def _calculate_performance_score(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        roc_auc: float,
    ) -> float:
        """Calculate overall performance score."""

        # Weighted average with ROC-AUC as primary metric
        return (
            roc_auc * 0.4 + f1 * 0.3 + accuracy * 0.2 + (precision + recall) / 2 * 0.1
        ) * 100

    def _calculate_efficiency_score(self, metrics: List[BenchmarkMetric]) -> float:
        """Calculate overall efficiency score."""

        efficiency_metrics = [
            m
            for m in metrics
            if m.category
            in [
                BenchmarkCategory.MODEL_EFFICIENCY,
                BenchmarkCategory.TRAINING_EFFICIENCY,
                BenchmarkCategory.INFERENCE_EFFICIENCY,
            ]
        ]

        if not efficiency_metrics:
            return 0.0

        total_score = 0.0
        for metric in efficiency_metrics:
            if metric.industry_percentile is not None:
                score = metric.industry_percentile
                if not metric.is_higher_better:
                    score = 100 - score
                total_score += score

        return total_score / len(efficiency_metrics)

    def _save_benchmark_result(self, result: ModelBenchmarkResult):
        """Save benchmark result to file."""

        try:
            result_file = self.results_dir / f"{result.model_id}.json"
            with open(result_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            logger.debug(f"Benchmark result saved: {result_file}")

        except Exception as e:
            logger.error(f"Failed to save benchmark result: {e}")

    def compare_models(self, model_results: List[ModelBenchmarkResult]) -> pd.DataFrame:
        """Compare multiple models across sustainability metrics."""

        comparison_data = []

        for result in model_results:
            comparison_data.append(
                {
                    "Model": result.model_name,
                    "Type": result.model_type,
                    "Overall Score": result.overall_score,
                    "Sustainability Score": result.sustainability_score,
                    "Performance Score": result.performance_score,
                    "Efficiency Score": result.efficiency_score,
                    "ROC-AUC": result.roc_auc,
                    "Carbon (kg)": result.carbon_footprint_kg,
                    "Energy (kWh)": result.energy_consumption_kwh,
                    "Carbon Efficiency": result.carbon_efficiency,
                    "Energy Efficiency": result.energy_efficiency,
                    "Model Size (MB)": result.model_size_mb,
                    "Inference (ms)": result.inference_latency_ms,
                }
            )

        df = pd.DataFrame(comparison_data)

        # Sort by overall score
        df = df.sort_values("Overall Score", ascending=False)

        return df

    def generate_benchmark_report(
        self, model_results: List[ModelBenchmarkResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""

        if not model_results:
            return {"error": "No benchmark results provided"}

        # Calculate summary statistics
        overall_scores = [r.overall_score for r in model_results]
        sustainability_scores = [r.sustainability_score for r in model_results]
        performance_scores = [r.performance_score for r in model_results]

        # Find best and worst performers
        best_overall = max(model_results, key=lambda x: x.overall_score)
        worst_overall = min(model_results, key=lambda x: x.overall_score)
        best_sustainability = max(model_results, key=lambda x: x.sustainability_score)
        best_performance = max(model_results, key=lambda x: x.performance_score)

        # Industry comparison
        industry_sectors = list(set(r.industry_sector for r in model_results))
        industry_comparisons = {}

        for sector in industry_sectors:
            sector_results = [r for r in model_results if r.industry_sector == sector]
            if sector_results:
                industry_comparisons[sector.value] = {
                    "count": len(sector_results),
                    "avg_overall_score": np.mean(
                        [r.overall_score for r in sector_results]
                    ),
                    "avg_sustainability_score": np.mean(
                        [r.sustainability_score for r in sector_results]
                    ),
                    "avg_performance_score": np.mean(
                        [r.performance_score for r in sector_results]
                    ),
                }

        return {
            "report_timestamp": datetime.now().isoformat(),
            "total_models_benchmarked": len(model_results),
            "summary_statistics": {
                "overall_score": {
                    "mean": np.mean(overall_scores),
                    "std": np.std(overall_scores),
                    "min": np.min(overall_scores),
                    "max": np.max(overall_scores),
                },
                "sustainability_score": {
                    "mean": np.mean(sustainability_scores),
                    "std": np.std(sustainability_scores),
                    "min": np.min(sustainability_scores),
                    "max": np.max(sustainability_scores),
                },
                "performance_score": {
                    "mean": np.mean(performance_scores),
                    "std": np.std(performance_scores),
                    "min": np.min(performance_scores),
                    "max": np.max(performance_scores),
                },
            },
            "best_performers": {
                "overall": {
                    "model": best_overall.model_name,
                    "score": best_overall.overall_score,
                    "type": best_overall.model_type,
                },
                "sustainability": {
                    "model": best_sustainability.model_name,
                    "score": best_sustainability.sustainability_score,
                    "type": best_sustainability.model_type,
                },
                "performance": {
                    "model": best_performance.model_name,
                    "score": best_performance.performance_score,
                    "type": best_performance.model_type,
                },
            },
            "worst_performers": {
                "overall": {
                    "model": worst_overall.model_name,
                    "score": worst_overall.overall_score,
                    "type": worst_overall.model_type,
                }
            },
            "industry_comparisons": industry_comparisons,
            "recommendations": self._generate_recommendations(model_results),
        }

    def _generate_recommendations(
        self, model_results: List[ModelBenchmarkResult]
    ) -> List[str]:
        """Generate actionable recommendations based on benchmark results."""

        recommendations = []

        # Analyze sustainability scores
        sustainability_scores = [r.sustainability_score for r in model_results]
        avg_sustainability = np.mean(sustainability_scores)

        if avg_sustainability < 50:
            recommendations.append(
                "Consider implementing carbon-aware training strategies to improve sustainability scores"
            )

        # Analyze carbon efficiency
        carbon_efficiencies = [r.carbon_efficiency for r in model_results]
        avg_carbon_efficiency = np.mean(carbon_efficiencies)

        if avg_carbon_efficiency < 5.0:
            recommendations.append(
                "Optimize model architectures for better carbon efficiency (target: >5.0 AUC/kg CO2e)"
            )

        # Analyze energy efficiency
        energy_efficiencies = [r.energy_efficiency for r in model_results]
        avg_energy_efficiency = np.mean(energy_efficiencies)

        if avg_energy_efficiency < 10.0:
            recommendations.append(
                "Implement energy-efficient training techniques (target: >10.0 AUC/kWh)"
            )

        # Analyze model sizes
        model_sizes = [r.model_size_mb for r in model_results]
        avg_model_size = np.mean(model_sizes)

        if avg_model_size > 50:
            recommendations.append(
                "Consider model compression techniques (pruning, quantization) to reduce model size"
            )

        # Analyze inference latency
        inference_latencies = [r.inference_latency_ms for r in model_results]
        avg_latency = np.mean(inference_latencies)

        if avg_latency > 100:
            recommendations.append(
                "Optimize inference pipeline for lower latency (target: <100ms)"
            )

        return recommendations


# Utility functions


def create_sustainable_ai_benchmark(
    industry_provider: Optional[IndustryBenchmarkProvider] = None,
) -> SustainableAIBenchmark:
    """Create and configure sustainable AI benchmark system."""
    return SustainableAIBenchmark(industry_provider)


def benchmark_model_sustainability(
    model,
    model_name: str,
    model_type: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    industry_sector: IndustrySector = IndustrySector.FINANCE,
    training_metadata: Optional[Dict[str, Any]] = None,
) -> ModelBenchmarkResult:
    """Benchmark a single model for sustainability."""

    benchmark = create_sustainable_ai_benchmark()
    return benchmark.benchmark_model(
        model,
        model_name,
        model_type,
        X_test,
        y_test,
        industry_sector,
        training_metadata,
    )


def compare_model_sustainability(
    models: List[Tuple[Any, str, str]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    industry_sector: IndustrySector = IndustrySector.FINANCE,
) -> pd.DataFrame:
    """Compare multiple models for sustainability."""

    benchmark = create_sustainable_ai_benchmark()
    results = []

    for model, name, model_type in models:
        try:
            result = benchmark.benchmark_model(
                model, name, model_type, X_test, y_test, industry_sector
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to benchmark {name}: {e}")

    return benchmark.compare_models(results)
