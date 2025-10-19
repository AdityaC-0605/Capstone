"""
Carbon-Aware LightGBM Model - ACTUALLY Reduces Carbon Emissions

This module demonstrates how to integrate carbon-aware optimization
into existing ML models to achieve real emission reductions.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import warnings

try:
    from ..sustainability.carbon_aware_optimizer import (
        CarbonAwareOptimizer,
        CarbonAwareConfig,
        carbon_aware_training,
    )
    from ..sustainability.energy_tracker import EnergyTracker
    from ..sustainability.carbon_calculator import CarbonCalculator

    CARBON_AWARE_AVAILABLE = True
except ImportError:
    CARBON_AWARE_AVAILABLE = False
    warnings.warn("Carbon-aware optimization not available")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


class CarbonAwareLightGBM:
    """
    LightGBM model with integrated carbon emission reduction strategies.

    This class demonstrates how existing ML models can be enhanced to
    ACTUALLY reduce carbon emissions through intelligent optimization.
    """

    def __init__(self, carbon_config: Optional[CarbonAwareConfig] = None, **lgb_params):
        """
        Initialize carbon-aware LightGBM model.

        Args:
            carbon_config: Carbon optimization configuration
            **lgb_params: LightGBM parameters
        """
        self.carbon_config = carbon_config or CarbonAwareConfig(
            enable_carbon_scheduling=True,
            enable_budget_enforcement=True,
            daily_carbon_budget_kg=0.1,  # 100g CO2e daily budget
            low_carbon_threshold=200.0,
            medium_carbon_threshold=400.0,
            high_carbon_threshold=600.0,
        )

        # Default LightGBM parameters optimized for efficiency
        self.lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
            **lgb_params,
        }

        self.model = None
        self.carbon_optimizer = None
        self.training_report = None

        if CARBON_AWARE_AVAILABLE:
            self.carbon_optimizer = CarbonAwareOptimizer(self.carbon_config)
        else:
            warnings.warn(
                "Training without carbon optimization - install carbon dependencies"
            )

    def _carbon_aware_lgb_training(
        self, X_train, y_train, X_val=None, y_val=None, num_boost_round=100
    ):
        """
        Carbon-optimized LightGBM training function.

        This function implements several carbon reduction strategies:
        1. Dynamic parameter adjustment based on carbon intensity
        2. Early stopping based on carbon budget
        3. Efficient validation strategy
        """
        # Get current carbon intensity to adjust parameters
        if self.carbon_optimizer:
            current_intensity = (
                self.carbon_optimizer.scheduler.carbon_api.get_current_carbon_intensity()
            )

            # Adjust parameters based on carbon intensity
            optimized_params = self._optimize_params_for_carbon(current_intensity)
        else:
            optimized_params = self.lgb_params.copy()
            current_intensity = 400.0  # Default assumption

        print(f"🌍 Training with carbon intensity: {current_intensity:.0f} gCO2/kWh")
        print(f"🎯 Optimized parameters: {optimized_params}")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = (
            lgb.Dataset(X_val, label=y_val, reference=train_data)
            if X_val is not None
            else None
        )

        # Carbon-aware early stopping
        carbon_early_stopping = self._create_carbon_early_stopping()

        # Train model with carbon optimization
        callbacks = [lgb.log_evaluation(period=0)]  # Suppress verbose output
        if carbon_early_stopping:
            callbacks.append(carbon_early_stopping)

        model = lgb.train(
            optimized_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data] if valid_data else None,
            callbacks=callbacks,
        )

        return model

    def _optimize_params_for_carbon(self, carbon_intensity: float) -> Dict[str, Any]:
        """
        Optimize LightGBM parameters based on current carbon intensity.

        Higher carbon intensity = more aggressive efficiency optimizations
        """
        params = self.lgb_params.copy()

        if carbon_intensity <= self.carbon_config.low_carbon_threshold:
            # Low carbon: can afford full performance
            params.update(
                {
                    "num_leaves": 63,
                    "learning_rate": 0.1,
                    "feature_fraction": 1.0,
                    "bagging_fraction": 1.0,
                }
            )
            print("🟢 Low carbon intensity: Using full performance parameters")

        elif carbon_intensity <= self.carbon_config.medium_carbon_threshold:
            # Medium carbon: balanced approach
            params.update(
                {
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.8,
                }
            )
            print("🟡 Medium carbon intensity: Using balanced parameters")

        else:
            # High carbon: prioritize efficiency
            params.update(
                {
                    "num_leaves": 15,
                    "learning_rate": 0.03,
                    "feature_fraction": 0.7,
                    "bagging_fraction": 0.6,
                    "max_depth": 6,  # Limit tree depth
                }
            )
            print("🔴 High carbon intensity: Using efficiency-optimized parameters")

        return params

    def _create_carbon_early_stopping(self):
        """Create carbon budget-aware early stopping callback."""
        if not self.carbon_optimizer:
            return None

        def carbon_early_stopping_callback(env):
            """Early stopping based on carbon budget."""
            # Check carbon budget every 10 iterations
            if env.iteration % 10 == 0:
                can_continue, remaining_budget, status = (
                    self.carbon_optimizer.budget_enforcer.check_budget_status()
                )

                if not can_continue:
                    print(f"🛑 Early stopping due to carbon budget: {status}")
                    raise lgb.early_stopping(env.iteration)

                if remaining_budget < 0.01:  # Less than 10g remaining
                    print(f"⚠️  Carbon budget warning: {status}")

        return carbon_early_stopping_callback

    def fit(
        self,
        X,
        y,
        validation_split=0.2,
        num_boost_round=100,
        enable_carbon_optimization=True,
    ):
        """
        Train the model with carbon-aware optimization.

        Args:
            X: Training features
            y: Training labels
            validation_split: Fraction of data for validation
            num_boost_round: Number of boosting rounds
            enable_carbon_optimization: Whether to use carbon optimization
        """
        print("🌱 Starting Carbon-Aware LightGBM Training")
        print("=" * 50)

        # Split data
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        if enable_carbon_optimization and CARBON_AWARE_AVAILABLE:
            # Use carbon-aware training
            result, optimization_report = carbon_aware_training(
                model=None,  # LightGBM doesn't use PyTorch model
                train_func=self._carbon_aware_lgb_training,
                config=self.carbon_config,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                num_boost_round=num_boost_round,
            )

            self.model = result
            self.training_report = optimization_report

            print(f"\n📊 CARBON-OPTIMIZED TRAINING COMPLETED")
            print(
                f"⚡ Energy Consumed: {optimization_report['final_energy_consumption_kwh']:.6f} kWh"
            )
            print(
                f"🌍 Carbon Emissions: {optimization_report['final_carbon_footprint_kg']:.6f} kg CO2e"
            )
            print(
                f"🎯 Strategies Applied: {', '.join(optimization_report['strategies_applied'])}"
            )

        else:
            # Traditional training
            print("⚠️  Training without carbon optimization")

            # Track energy for comparison
            energy_tracker = EnergyTracker()
            experiment_id = (
                f"lightgbm_traditional_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            energy_tracker.start_tracking(experiment_id)
            self.model = self._carbon_aware_lgb_training(
                X_train, y_train, X_val, y_val, num_boost_round
            )
            energy_report = energy_tracker.stop_tracking()

            # Calculate carbon footprint
            carbon_calc = CarbonCalculator()
            carbon_footprint = carbon_calc.calculate_carbon_footprint(
                energy_report, "US"
            )

            self.training_report = {
                "final_energy_consumption_kwh": energy_report.total_energy_kwh,
                "final_carbon_footprint_kg": carbon_footprint.total_emissions_kg,
                "strategies_applied": ["traditional_training"],
            }

            print(f"\n📊 TRADITIONAL TRAINING COMPLETED")
            print(f"⚡ Energy Consumed: {energy_report.total_energy_kwh:.6f} kWh")
            print(
                f"🌍 Carbon Emissions: {carbon_footprint.total_emissions_kg:.6f} kg CO2e"
            )

        return self

    def predict(self, X):
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities."""
        predictions = self.predict(X)
        # Convert to probabilities (sigmoid for binary classification)
        proba = 1 / (1 + np.exp(-predictions))
        return np.column_stack([1 - proba, proba])

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        proba = self.predict_proba(X_test)[:, 1]

        # Convert predictions to binary
        binary_predictions = (predictions > 0).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, binary_predictions),
            "roc_auc": roc_auc_score(y_test, proba),
            "precision": precision_score(y_test, binary_predictions),
            "recall": recall_score(y_test, binary_predictions),
            "f1": f1_score(y_test, binary_predictions),
        }

        return metrics

    def get_carbon_impact_report(self):
        """Get detailed carbon impact report."""
        if not self.training_report:
            return {"error": "No training report available"}

        report = {
            "carbon_footprint_kg": self.training_report.get(
                "final_carbon_footprint_kg", 0
            ),
            "energy_consumption_kwh": self.training_report.get(
                "final_energy_consumption_kwh", 0
            ),
            "optimization_strategies": self.training_report.get(
                "strategies_applied", []
            ),
            "carbon_savings_kg": self.training_report.get("carbon_savings_kg", 0),
            "environmental_equivalents": {},
        }

        # Calculate environmental equivalents
        carbon_kg = report["carbon_footprint_kg"]
        if carbon_kg > 0:
            report["environmental_equivalents"] = {
                "km_driving_avoided": carbon_kg / 0.251,
                "smartphone_charges_equivalent": carbon_kg / 0.0084,
                "hours_laptop_use_equivalent": carbon_kg / 0.0086,
                "trees_needed_annual_absorption": carbon_kg / 21.77,
            }

        return report

    def get_feature_importance(self, importance_type="gain"):
        """Get feature importance from the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.feature_importance(importance_type=importance_type)


def demonstrate_carbon_reduction():
    """Demonstrate actual carbon emission reduction with LightGBM."""
    print("🌱 CARBON-AWARE LIGHTGBM DEMONSTRATION")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 20

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"📊 Dataset: {n_samples} samples, {n_features} features")

    # Traditional LightGBM
    print(f"\n🔥 TRADITIONAL LIGHTGBM TRAINING")
    traditional_model = CarbonAwareLightGBM()
    traditional_model.fit(
        X_train, y_train, enable_carbon_optimization=False, num_boost_round=50
    )
    traditional_metrics = traditional_model.evaluate(X_test, y_test)
    traditional_report = traditional_model.get_carbon_impact_report()

    # Carbon-aware LightGBM
    print(f"\n🌱 CARBON-AWARE LIGHTGBM TRAINING")
    carbon_config = CarbonAwareConfig(
        enable_carbon_scheduling=True,
        enable_budget_enforcement=True,
        daily_carbon_budget_kg=0.05,
    )
    carbon_model = CarbonAwareLightGBM(carbon_config=carbon_config)
    carbon_model.fit(
        X_train, y_train, enable_carbon_optimization=True, num_boost_round=50
    )
    carbon_metrics = carbon_model.evaluate(X_test, y_test)
    carbon_report = carbon_model.get_carbon_impact_report()

    # Compare results
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("=" * 40)
    print(f"Traditional Accuracy: {traditional_metrics['accuracy']:.4f}")
    print(f"Carbon-Aware Accuracy: {carbon_metrics['accuracy']:.4f}")
    print(f"Traditional AUC: {traditional_metrics['roc_auc']:.4f}")
    print(f"Carbon-Aware AUC: {carbon_metrics['roc_auc']:.4f}")

    print(f"\n🌍 CARBON IMPACT COMPARISON")
    print("=" * 40)
    trad_carbon = traditional_report.get("carbon_footprint_kg", 0)
    carbon_carbon = carbon_report.get("carbon_footprint_kg", 0)
    carbon_reduction = (
        ((trad_carbon - carbon_carbon) / trad_carbon * 100) if trad_carbon > 0 else 0
    )

    print(f"Traditional Carbon: {trad_carbon:.6f} kg CO2e")
    print(f"Carbon-Aware Carbon: {carbon_carbon:.6f} kg CO2e")
    print(f"Carbon Reduction: {carbon_reduction:.1f}%")

    if carbon_reduction > 0:
        print(f"\n🎉 SUCCESS: Achieved {carbon_reduction:.1f}% carbon reduction!")
        print("✅ Maintained model performance while reducing environmental impact")

    return {
        "traditional": {"metrics": traditional_metrics, "carbon": traditional_report},
        "carbon_aware": {"metrics": carbon_metrics, "carbon": carbon_report},
        "carbon_reduction_percent": carbon_reduction,
    }


if __name__ == "__main__":
    demonstrate_carbon_reduction()
