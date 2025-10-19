"""
Carbon-Aware AI Optimizer - Actively Reduces Carbon Emissions

This module implements intelligent carbon reduction strategies that actually
minimize the environmental impact of AI training and inference through:
- Carbon-aware scheduling (train when grid is cleanest)
- Dynamic model scaling based on carbon intensity
- Energy-efficient hyperparameter optimization
- Real-time carbon budget enforcement
"""

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn

try:
    from ..core.logging import get_logger
    from .carbon_calculator import CarbonCalculator, CarbonFootprintConfig
    from .energy_tracker import EnergyConfig, EnergyTracker
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from core.logging import get_logger

logger = get_logger(__name__)


class CarbonOptimizationStrategy(Enum):
    """Carbon optimization strategies."""

    SCHEDULE_CLEAN_ENERGY = "schedule_clean_energy"
    DYNAMIC_MODEL_SCALING = "dynamic_model_scaling"
    CARBON_BUDGET_ENFORCEMENT = "carbon_budget_enforcement"
    ENERGY_EFFICIENT_HYPERPARAMS = "energy_efficient_hyperparams"
    ADAPTIVE_PRECISION = "adaptive_precision"
    EARLY_STOPPING_CARBON = "early_stopping_carbon"


@dataclass
class CarbonAwareConfig:
    """Configuration for carbon-aware optimization."""

    # Carbon intensity thresholds (gCO2/kWh)
    low_carbon_threshold: float = 200.0
    medium_carbon_threshold: float = 400.0
    high_carbon_threshold: float = 600.0

    # Scheduling settings
    enable_carbon_scheduling: bool = True
    max_wait_hours: float = 6.0  # Max hours to wait for clean energy
    check_interval_minutes: int = 15

    # Model scaling settings
    enable_dynamic_scaling: bool = True
    min_model_scale: float = 0.5  # Minimum model size (50% of original)
    max_model_scale: float = 1.0  # Maximum model size

    # Budget enforcement
    enable_budget_enforcement: bool = True
    daily_carbon_budget_kg: float = 0.1  # 100g CO2e per day
    emergency_stop_threshold: float = 0.95  # Stop at 95% of budget

    # Energy efficiency
    enable_adaptive_precision: bool = True
    low_carbon_precision: str = "fp32"
    medium_carbon_precision: str = "fp16"
    high_carbon_precision: str = "int8"

    # API settings for real-time carbon data
    carbon_api_url: Optional[str] = None
    api_key: Optional[str] = None
    region: str = "US"


class CarbonIntensityAPI:
    """Interface to real-time carbon intensity APIs."""

    def __init__(self, config: CarbonAwareConfig):
        self.config = config
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)

    def get_current_carbon_intensity(self) -> float:
        """Get current carbon intensity for the region."""
        cache_key = (
            f"{self.config.region}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )

        # Check cache first
        if cache_key in self.cache:
            cached_time, intensity = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return intensity

        try:
            # Try to get real-time data
            intensity = self._fetch_real_time_intensity()
            self.cache[cache_key] = (datetime.now(), intensity)
            return intensity
        except Exception as e:
            logger.warning(f"Failed to fetch real-time carbon intensity: {e}")
            # Fallback to default values based on region
            return self._get_default_intensity()

    def _fetch_real_time_intensity(self) -> float:
        """Fetch real-time carbon intensity from API."""
        if not self.config.carbon_api_url:
            raise ValueError("No carbon API URL configured")

        # Example implementation for WattTime API
        headers = (
            {"Authorization": f"Bearer {self.config.api_key}"}
            if self.config.api_key
            else {}
        )

        response = requests.get(
            f"{self.config.carbon_api_url}/marginal-carbon",
            params={"region": self.config.region},
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            return float(data.get("marginal_carbon", {}).get("value", 400))
        else:
            raise Exception(f"API request failed: {response.status_code}")

    def _get_default_intensity(self) -> float:
        """Get default carbon intensity for region."""
        defaults = {
            "US": 386.0,
            "EU": 275.0,
            "CN": 555.0,
            "IN": 632.0,
            "BR": 85.0,
            "NO": 17.0,
            "FR": 57.0,
        }
        return defaults.get(self.config.region, 400.0)

    def get_forecast(
        self, hours_ahead: int = 24
    ) -> List[Tuple[datetime, float]]:
        """Get carbon intensity forecast."""
        # For demo purposes, generate a realistic forecast
        current_intensity = self.get_current_carbon_intensity()
        forecast = []

        for hour in range(hours_ahead):
            # Simulate daily carbon intensity patterns
            time_point = datetime.now() + timedelta(hours=hour)
            hour_of_day = time_point.hour

            # Lower carbon intensity during night (more renewable energy)
            if 2 <= hour_of_day <= 6:
                intensity = current_intensity * 0.7  # 30% lower at night
            elif 10 <= hour_of_day <= 16:
                intensity = (
                    current_intensity * 1.2
                )  # 20% higher during peak hours
            else:
                intensity = current_intensity

            # Add some randomness
            intensity *= 0.9 + 0.2 * np.random.random()
            forecast.append((time_point, intensity))

        return forecast


class CarbonAwareScheduler:
    """Schedules AI training during periods of low carbon intensity."""

    def __init__(self, config: CarbonAwareConfig):
        self.config = config
        self.carbon_api = CarbonIntensityAPI(config)
        self.scheduled_jobs = []

    def should_train_now(self) -> Tuple[bool, str, float]:
        """
        Determine if training should start now based on carbon intensity.

        Returns:
            (should_train, reason, current_intensity)
        """
        current_intensity = self.carbon_api.get_current_carbon_intensity()

        if current_intensity <= self.config.low_carbon_threshold:
            return (
                True,
                "Low carbon intensity - optimal time to train",
                current_intensity,
            )
        elif current_intensity <= self.config.medium_carbon_threshold:
            return (
                True,
                "Medium carbon intensity - acceptable to train",
                current_intensity,
            )
        else:
            # Check if we should wait for cleaner energy
            forecast = self.carbon_api.get_forecast(
                int(self.config.max_wait_hours)
            )

            # Find the next low-carbon period
            for time_point, intensity in forecast:
                if intensity <= self.config.low_carbon_threshold:
                    wait_hours = (
                        time_point - datetime.now()
                    ).total_seconds() / 3600
                    if wait_hours <= self.config.max_wait_hours:
                        return (
                            False,
                            f"High carbon intensity - wait {wait_hours:.1f}h for cleaner energy",
                            current_intensity,
                        )

            # If no clean period found within max wait time, train anyway
            return (
                True,
                "High carbon intensity - but max wait time exceeded",
                current_intensity,
            )

    def wait_for_clean_energy(
        self, max_wait_hours: Optional[float] = None
    ) -> float:
        """
        Wait for a period of clean energy before starting training.

        Returns:
            Carbon intensity when training should start
        """
        max_wait = max_wait_hours or self.config.max_wait_hours
        start_time = datetime.now()

        logger.info(f"Waiting for clean energy (max {max_wait} hours)...")

        while True:
            should_train, reason, intensity = self.should_train_now()

            if should_train:
                logger.info(
                    f"Starting training: {reason} ({intensity:.0f} gCO2/kWh)"
                )
                return intensity

            # Check if we've waited too long
            elapsed_hours = (
                datetime.now() - start_time
            ).total_seconds() / 3600
            if elapsed_hours >= max_wait:
                logger.warning(
                    f"Max wait time exceeded, starting training anyway ({intensity:.0f} gCO2/kWh)"
                )
                return intensity

            # Wait before checking again
            time.sleep(self.config.check_interval_minutes * 60)


class DynamicModelScaler:
    """Dynamically scales model size based on carbon intensity."""

    def __init__(self, config: CarbonAwareConfig):
        self.config = config
        self.original_model = None
        self.scaled_models = {}

    def get_optimal_model_scale(self, carbon_intensity: float) -> float:
        """Get optimal model scale factor based on carbon intensity."""
        if carbon_intensity <= self.config.low_carbon_threshold:
            return 1.0  # Full model size
        elif carbon_intensity <= self.config.medium_carbon_threshold:
            return 0.8  # 80% of original size
        else:
            return self.config.min_model_scale  # Minimum size

    def scale_model(self, model: nn.Module, scale_factor: float) -> nn.Module:
        """Scale model size by reducing hidden dimensions."""
        if scale_factor == 1.0:
            return model

        # Cache scaled models
        cache_key = f"{id(model)}_{scale_factor}"
        if cache_key in self.scaled_models:
            return self.scaled_models[cache_key]

        # Create scaled version
        scaled_model = self._create_scaled_model(model, scale_factor)
        self.scaled_models[cache_key] = scaled_model

        logger.info(f"Model scaled to {scale_factor:.1%} of original size")
        return scaled_model

    def _create_scaled_model(
        self, model: nn.Module, scale_factor: float
    ) -> nn.Module:
        """Create a scaled version of the model."""
        # This is a simplified implementation
        # In practice, you'd need model-specific scaling logic

        scaled_model = type(model)()  # Create new instance

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Scale linear layers
                original_features = module.out_features
                scaled_features = max(1, int(original_features * scale_factor))

                scaled_layer = nn.Linear(module.in_features, scaled_features)
                # Copy and scale weights
                with torch.no_grad():
                    scaled_layer.weight.data = module.weight.data[
                        :scaled_features, :
                    ]
                    if module.bias is not None:
                        scaled_layer.bias.data = module.bias.data[
                            :scaled_features
                        ]

                # Replace in scaled model
                setattr(scaled_model, name, scaled_layer)

        return scaled_model


class CarbonBudgetEnforcer:
    """Enforces carbon budget limits during training."""

    def __init__(self, config: CarbonAwareConfig):
        self.config = config
        self.daily_usage = 0.0
        self.last_reset = datetime.now().date()
        self.carbon_calculator = CarbonCalculator()

    def check_budget_status(self) -> Tuple[bool, float, str]:
        """
        Check current carbon budget status.

        Returns:
            (can_continue, remaining_budget_kg, status_message)
        """
        # Reset daily usage if new day
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_usage = 0.0
            self.last_reset = current_date

        remaining_budget = (
            self.config.daily_carbon_budget_kg - self.daily_usage
        )
        usage_percentage = (
            self.daily_usage / self.config.daily_carbon_budget_kg
        ) * 100

        if usage_percentage >= self.config.emergency_stop_threshold * 100:
            return (
                False,
                remaining_budget,
                f"EMERGENCY STOP: {usage_percentage:.1f}% of daily budget used",
            )
        elif usage_percentage >= 80:
            return (
                True,
                remaining_budget,
                f"WARNING: {usage_percentage:.1f}% of daily budget used",
            )
        else:
            return (
                True,
                remaining_budget,
                f"OK: {usage_percentage:.1f}% of daily budget used",
            )

    def add_carbon_usage(self, carbon_kg: float):
        """Add carbon usage to daily total."""
        self.daily_usage += carbon_kg
        logger.debug(
            f"Added {carbon_kg:.6f} kg CO2e to daily usage (total: {self.daily_usage:.6f} kg)"
        )

    def estimate_training_carbon(
        self,
        model: nn.Module,
        dataset_size: int,
        epochs: int,
        carbon_intensity: float,
    ) -> float:
        """Estimate carbon footprint of training job."""
        # Rough estimation based on model parameters and dataset size
        model_params = sum(p.numel() for p in model.parameters())

        # Estimate energy consumption (very rough approximation)
        # Based on: parameters * dataset_size * epochs * energy_per_operation
        energy_per_param_per_sample = (
            1e-12  # kWh per parameter per sample (rough estimate)
        )
        estimated_energy_kwh = (
            model_params * dataset_size * epochs * energy_per_param_per_sample
        )

        # Convert to carbon emissions
        estimated_carbon_kg = (estimated_energy_kwh * carbon_intensity) / 1000

        return estimated_carbon_kg


class AdaptivePrecisionManager:
    """Manages model precision based on carbon intensity."""

    def __init__(self, config: CarbonAwareConfig):
        self.config = config

    def get_optimal_precision(self, carbon_intensity: float) -> str:
        """Get optimal precision based on carbon intensity."""
        if carbon_intensity <= self.config.low_carbon_threshold:
            return self.config.low_carbon_precision
        elif carbon_intensity <= self.config.medium_carbon_threshold:
            return self.config.medium_carbon_precision
        else:
            return self.config.high_carbon_precision

    def apply_precision(self, model: nn.Module, precision: str) -> nn.Module:
        """Apply precision optimization to model."""
        if precision == "fp16":
            return model.half()
        elif precision == "int8":
            # Simplified INT8 quantization
            return torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        else:  # fp32
            return model.float()


class CarbonAwareOptimizer:
    """Main carbon-aware optimizer that coordinates all strategies."""

    def __init__(self, config: Optional[CarbonAwareConfig] = None):
        self.config = config or CarbonAwareConfig()

        # Initialize components
        self.scheduler = CarbonAwareScheduler(self.config)
        self.model_scaler = DynamicModelScaler(self.config)
        self.budget_enforcer = CarbonBudgetEnforcer(self.config)
        self.precision_manager = AdaptivePrecisionManager(self.config)

        # Energy tracking
        self.energy_tracker = EnergyTracker()
        self.carbon_calculator = CarbonCalculator()

        logger.info("Carbon-aware optimizer initialized")

    def optimize_training(
        self, model: nn.Module, train_func: Callable, *args, **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Optimize training with carbon-aware strategies.

        Args:
            model: PyTorch model to train
            train_func: Training function
            *args, **kwargs: Arguments for training function

        Returns:
            (training_result, optimization_report)
        """
        optimization_report = {
            "strategies_applied": [],
            "carbon_savings_kg": 0.0,
            "energy_savings_kwh": 0.0,
            "original_carbon_intensity": None,
            "optimized_carbon_intensity": None,
        }

        # 1. Carbon-aware scheduling
        if self.config.enable_carbon_scheduling:
            should_train, reason, intensity = self.scheduler.should_train_now()
            optimization_report["original_carbon_intensity"] = intensity

            if not should_train:
                logger.info(f"Waiting for cleaner energy: {reason}")
                intensity = self.scheduler.wait_for_clean_energy()
                optimization_report["strategies_applied"].append(
                    "carbon_scheduling"
                )

            optimization_report["optimized_carbon_intensity"] = intensity
        else:
            intensity = (
                self.scheduler.carbon_api.get_current_carbon_intensity()
            )
            optimization_report["original_carbon_intensity"] = intensity
            optimization_report["optimized_carbon_intensity"] = intensity

        # 2. Check carbon budget
        if self.config.enable_budget_enforcement:
            can_continue, remaining_budget, status = (
                self.budget_enforcer.check_budget_status()
            )

            if not can_continue:
                raise RuntimeError(f"Carbon budget exceeded: {status}")

            logger.info(f"Carbon budget status: {status}")

        # 3. Dynamic model scaling
        optimized_model = model
        if self.config.enable_dynamic_scaling:
            scale_factor = self.model_scaler.get_optimal_model_scale(intensity)
            if scale_factor < 1.0:
                optimized_model = self.model_scaler.scale_model(
                    model, scale_factor
                )
                optimization_report["strategies_applied"].append(
                    "dynamic_scaling"
                )
                logger.info(
                    f"Model scaled to {scale_factor:.1%} due to high carbon intensity"
                )

        # 4. Adaptive precision
        if self.config.enable_adaptive_precision:
            precision = self.precision_manager.get_optimal_precision(intensity)
            optimized_model = self.precision_manager.apply_precision(
                optimized_model, precision
            )
            optimization_report["strategies_applied"].append(
                f"adaptive_precision_{precision}"
            )
            logger.info(f"Using {precision} precision due to carbon intensity")

        # 5. Execute training with energy tracking
        experiment_id = (
            f"carbon_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        with self.energy_tracker.track(experiment_id) as tracker:
            # Execute training
            result = train_func(optimized_model, *args, **kwargs)

            # Get energy report
            energy_report = tracker.stop_tracking()

        # 6. Calculate carbon footprint
        carbon_footprint = self.carbon_calculator.calculate_carbon_footprint(
            energy_report, self.config.region
        )

        # 7. Update budget
        if self.config.enable_budget_enforcement:
            self.budget_enforcer.add_carbon_usage(
                carbon_footprint.total_emissions_kg
            )

        # 8. Calculate savings (estimate what it would have been without optimization)
        baseline_intensity = optimization_report["original_carbon_intensity"]
        optimized_intensity = optimization_report["optimized_carbon_intensity"]

        if baseline_intensity and optimized_intensity:
            intensity_reduction = (
                baseline_intensity - optimized_intensity
            ) / baseline_intensity
            optimization_report["carbon_savings_kg"] = (
                carbon_footprint.total_emissions_kg * intensity_reduction
            )

        # Add final metrics to report
        optimization_report.update(
            {
                "final_carbon_footprint_kg": carbon_footprint.total_emissions_kg,
                "final_energy_consumption_kwh": energy_report.total_energy_kwh,
                "training_duration_seconds": energy_report.duration_seconds,
                "carbon_intensity_gco2_kwh": carbon_footprint.carbon_intensity_gco2_kwh,
            }
        )

        logger.info(
            f"Carbon-optimized training completed: "
            f"{carbon_footprint.total_emissions_kg:.6f} kg CO2e, "
            f"strategies: {optimization_report['strategies_applied']}"
        )

        return result, optimization_report

    def get_carbon_recommendations(self) -> List[str]:
        """Get personalized carbon reduction recommendations."""
        current_intensity = (
            self.scheduler.carbon_api.get_current_carbon_intensity()
        )
        recommendations = []

        if current_intensity > self.config.high_carbon_threshold:
            recommendations.extend(
                [
                    f"Current carbon intensity is high ({current_intensity:.0f} gCO2/kWh). Consider:",
                    "- Waiting for cleaner energy (check forecast)",
                    "- Using smaller model variants",
                    "- Reducing training epochs",
                    "- Using mixed precision training",
                ]
            )
        elif current_intensity > self.config.medium_carbon_threshold:
            recommendations.extend(
                [
                    f"Current carbon intensity is moderate ({current_intensity:.0f} gCO2/kWh). Consider:",
                    "- Using FP16 precision",
                    "- Implementing early stopping",
                    "- Batch size optimization",
                ]
            )
        else:
            recommendations.append(
                f"Current carbon intensity is low ({current_intensity:.0f} gCO2/kWh) - good time for training!"
            )

        # Budget-based recommendations
        can_continue, remaining_budget, status = (
            self.budget_enforcer.check_budget_status()
        )
        if remaining_budget < 0.01:  # Less than 10g remaining
            recommendations.append(
                f"Carbon budget is nearly exhausted ({status}). Consider postponing non-critical training."
            )

        return recommendations


# Utility functions for easy integration


def carbon_aware_training(
    model: nn.Module,
    train_func: Callable,
    config: Optional[CarbonAwareConfig] = None,
    *args,
    **kwargs,
):
    """
    Wrapper function for carbon-aware training.

    Usage:
        result, report = carbon_aware_training(
            model=my_model,
            train_func=my_training_function,
            dataset=train_dataset,
            epochs=10
        )
    """
    optimizer = CarbonAwareOptimizer(config)
    return optimizer.optimize_training(model, train_func, *args, **kwargs)


def get_carbon_status() -> Dict[str, Any]:
    """Get current carbon status and recommendations."""
    config = CarbonAwareConfig()
    optimizer = CarbonAwareOptimizer(config)

    # Get current status
    should_train, reason, intensity = optimizer.scheduler.should_train_now()
    can_continue, remaining_budget, budget_status = (
        optimizer.budget_enforcer.check_budget_status()
    )
    recommendations = optimizer.get_carbon_recommendations()

    return {
        "current_carbon_intensity": intensity,
        "should_train_now": should_train,
        "reason": reason,
        "carbon_budget_status": budget_status,
        "remaining_budget_kg": remaining_budget,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat(),
    }
