"""
Hyperparameter optimization system using Optuna for automated tuning.
Supports multi-objective optimization including accuracy and energy efficiency.
"""

import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ML imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

try:
    from ..models.dnn_model import DNNModel, DNNTrainer, DNNConfig
    from ..models.lstm_model import LSTMModel, LSTMTrainer, LSTMConfig
    from ..models.gnn_model import GNNModel, GNNTrainer, GNNConfig
    from ..models.tcn_model import TCNModel, TCNTrainer, TCNConfig
    from ..models.lightgbm_model import LightGBMModel, LightGBMTrainer, LightGBMConfig
    from ..core.logging import get_logger, get_audit_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from models.dnn_model import DNNModel, DNNTrainer, DNNConfig
    from models.lstm_model import LSTMModel, LSTMTrainer, LSTMConfig
    from models.gnn_model import GNNModel, GNNTrainer, GNNConfig
    from models.tcn_model import TCNModel, TCNTrainer, TCNConfig
    from models.lightgbm_model import LightGBMModel, LightGBMTrainer, LightGBMConfig
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
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    # Study configuration
    study_name: str = "credit_risk_optimization"
    storage_url: Optional[str] = None  # For distributed optimization
    direction: str = "maximize"  # 'maximize' or 'minimize'
    
    # Multi-objective optimization
    use_multi_objective: bool = True
    objectives: List[str] = field(default_factory=lambda: ["accuracy", "energy_efficiency"])
    
    # Optimization parameters
    n_trials: int = 100
    timeout: Optional[int] = None  # Timeout in seconds
    n_jobs: int = 1  # Number of parallel jobs
    
    # Sampler configuration
    sampler_type: str = "tpe"  # 'tpe', 'cmaes', 'random'
    sampler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Pruner configuration
    pruner_type: str = "median"  # 'median', 'hyperband', 'none'
    pruner_params: Dict[str, Any] = field(default_factory=dict)
    
    # Cross-validation
    cv_folds: int = 3
    cv_scoring: str = "roc_auc"
    
    # Early stopping for individual trials
    early_stopping_rounds: int = 10
    min_improvement: float = 0.001
    
    # Energy monitoring
    track_energy: bool = True
    energy_weight: float = 0.3  # Weight for energy efficiency in multi-objective
    
    # Results storage
    save_results: bool = True
    results_path: str = "optimization_results"
    
    # Model-specific settings
    max_epochs_per_trial: int = 50
    validation_split: float = 0.2


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    success: bool
    study: Optional[optuna.Study]
    best_params: Dict[str, Any]
    best_value: float
    best_values: Optional[List[float]]  # For multi-objective
    n_trials: int
    optimization_time_seconds: float
    model_type: str
    config_class: type
    best_config: Any
    pareto_front: Optional[List[Dict[str, Any]]]  # For multi-objective
    parameter_importance: Dict[str, float]
    message: str


class EnergyTracker:
    """Simple energy tracking for optimization trials."""
    
    def __init__(self):
        self.start_time = None
        self.energy_consumed = 0.0
    
    def start(self):
        """Start energy tracking."""
        self.start_time = time.time()
        self.energy_consumed = 0.0
    
    def stop(self) -> float:
        """Stop tracking and return estimated energy consumption."""
        if self.start_time is None:
            return 0.0
        
        duration = time.time() - self.start_time
        # Simple estimation: assume 100W average power consumption
        # In practice, this would integrate with actual power monitoring
        estimated_power_watts = 100.0
        energy_kwh = (estimated_power_watts * duration) / (1000 * 3600)
        
        self.energy_consumed = energy_kwh
        return energy_kwh


class HyperparameterOptimizer:
    """Main hyperparameter optimization class."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.energy_tracker = EnergyTracker()
        
        # Initialize Optuna study
        self.study = None
        self._setup_study()
    
    def _setup_study(self):
        """Set up Optuna study with sampler and pruner."""
        # Configure sampler
        if self.config.sampler_type == "tpe":
            sampler = TPESampler(**self.config.sampler_params)
        elif self.config.sampler_type == "cmaes":
            sampler = CmaEsSampler(**self.config.sampler_params)
        else:
            sampler = optuna.samplers.RandomSampler(**self.config.sampler_params)
        
        # Configure pruner
        if self.config.pruner_type == "median":
            pruner = MedianPruner(**self.config.pruner_params)
        elif self.config.pruner_type == "hyperband":
            pruner = HyperbandPruner(**self.config.pruner_params)
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create study
        if self.config.use_multi_objective:
            directions = ["maximize" if obj == "accuracy" else "minimize" 
                         for obj in self.config.objectives]
            self.study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage_url,
                directions=directions,
                sampler=sampler,
                pruner=pruner
            )
        else:
            self.study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage_url,
                direction=self.config.direction,
                sampler=sampler,
                pruner=pruner
            )
    
    def optimize_dnn(self, X: pd.DataFrame, y: pd.Series) -> OptimizationResult:
        """Optimize DNN hyperparameters."""
        logger.info("Starting DNN hyperparameter optimization")
        
        def objective(trial):
            return self._dnn_objective(trial, X, y)
        
        return self._run_optimization(objective, "dnn", DNNConfig)
    
    def optimize_lstm(self, X: pd.DataFrame, y: pd.Series) -> OptimizationResult:
        """Optimize LSTM hyperparameters."""
        logger.info("Starting LSTM hyperparameter optimization")
        
        def objective(trial):
            return self._lstm_objective(trial, X, y)
        
        return self._run_optimization(objective, "lstm", LSTMConfig)
    
    def optimize_gnn(self, X: pd.DataFrame, y: pd.Series) -> OptimizationResult:
        """Optimize GNN hyperparameters."""
        logger.info("Starting GNN hyperparameter optimization")
        
        def objective(trial):
            return self._gnn_objective(trial, X, y)
        
        return self._run_optimization(objective, "gnn", GNNConfig)
    
    def optimize_tcn(self, X: pd.DataFrame, y: pd.Series) -> OptimizationResult:
        """Optimize TCN hyperparameters."""
        logger.info("Starting TCN hyperparameter optimization")
        
        def objective(trial):
            return self._tcn_objective(trial, X, y)
        
        return self._run_optimization(objective, "tcn", TCNConfig)
    
    def optimize_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> OptimizationResult:
        """Optimize LightGBM hyperparameters."""
        logger.info("Starting LightGBM hyperparameter optimization")
        
        def objective(trial):
            return self._lightgbm_objective(trial, X, y)
        
        return self._run_optimization(objective, "lightgbm", LightGBMConfig)
    
    def _dnn_objective(self, trial, X: pd.DataFrame, y: pd.Series) -> Union[float, List[float]]:
        """Objective function for DNN optimization."""
        # Suggest hyperparameters
        n_layers = trial.suggest_int("n_layers", 2, 5)
        hidden_layers = [
            trial.suggest_int(f"hidden_layer_{i}", 32, 512, log=True)
            for i in range(n_layers)
        ]
        
        config = DNNConfig(
            hidden_layers=hidden_layers,
            dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.5),
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            optimizer=trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"]),
            activation=trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu", "gelu"]),
            use_batch_norm=trial.suggest_categorical("use_batch_norm", [True, False]),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            focal_alpha=trial.suggest_float("focal_alpha", 0.1, 0.5),
            focal_gamma=trial.suggest_float("focal_gamma", 1.0, 3.0),
            epochs=self.config.max_epochs_per_trial,
            early_stopping_patience=self.config.early_stopping_rounds,
            save_model=False  # Don't save during optimization
        )
        
        return self._evaluate_config(trial, DNNTrainer, config, X, y)
    
    def _lstm_objective(self, trial, X: pd.DataFrame, y: pd.Series) -> Union[float, List[float]]:
        """Objective function for LSTM optimization."""
        config = LSTMConfig(
            hidden_size=trial.suggest_int("hidden_size", 64, 256, log=True),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.5),
            bidirectional=trial.suggest_categorical("bidirectional", [True, False]),
            use_attention=trial.suggest_categorical("use_attention", [True, False]),
            attention_dim=trial.suggest_int("attention_dim", 32, 128),
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            optimizer=trial.suggest_categorical("optimizer", ["adam", "adamw"]),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            max_sequence_length=trial.suggest_int("max_sequence_length", 20, 100),
            epochs=self.config.max_epochs_per_trial,
            early_stopping_patience=self.config.early_stopping_rounds,
            save_model=False
        )
        
        return self._evaluate_config(trial, LSTMTrainer, config, X, y)
    
    def _gnn_objective(self, trial, X: pd.DataFrame, y: pd.Series) -> Union[float, List[float]]:
        """Objective function for GNN optimization."""
        num_layers = trial.suggest_int("num_layers", 2, 4)
        config = GNNConfig(
            hidden_dims=[
                trial.suggest_int(f"hidden_dim_{i}", 32, 256, log=True)
                for i in range(num_layers)
            ],
            conv_type=trial.suggest_categorical("conv_type", ["gcn", "gat", "graph_conv"]),
            num_heads=trial.suggest_int("num_heads", 2, 8) if trial.params.get("conv_type") == "gat" else 4,
            dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.5),
            k_neighbors=trial.suggest_int("k_neighbors", 3, 10),
            graph_construction_method=trial.suggest_categorical("graph_method", ["knn", "threshold"]),
            similarity_threshold=trial.suggest_float("similarity_threshold", 0.5, 0.9),
            pooling_method=trial.suggest_categorical("pooling_method", ["mean", "max", "attention"]),
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            optimizer=trial.suggest_categorical("optimizer", ["adam", "adamw"]),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            epochs=self.config.max_epochs_per_trial,
            early_stopping_patience=self.config.early_stopping_rounds,
            save_model=False
        )
        
        return self._evaluate_config(trial, GNNTrainer, config, X, y)
    
    def _tcn_objective(self, trial, X: pd.DataFrame, y: pd.Series) -> Union[float, List[float]]:
        """Objective function for TCN optimization."""
        num_layers = trial.suggest_int("num_layers", 2, 6)
        config = TCNConfig(
            num_channels=[
                trial.suggest_int(f"channel_{i}", 32, 128, log=True)
                for i in range(num_layers)
            ],
            kernel_size=trial.suggest_int("kernel_size", 2, 7),
            dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.4),
            dilation_base=trial.suggest_int("dilation_base", 2, 4),
            activation=trial.suggest_categorical("activation", ["relu", "gelu", "swish"]),
            use_layer_norm=trial.suggest_categorical("use_layer_norm", [True, False]),
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            optimizer=trial.suggest_categorical("optimizer", ["adam", "adamw"]),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            max_sequence_length=trial.suggest_int("max_sequence_length", 20, 80),
            epochs=self.config.max_epochs_per_trial,
            early_stopping_patience=self.config.early_stopping_rounds,
            save_model=False
        )
        
        return self._evaluate_config(trial, TCNTrainer, config, X, y)
    
    def _lightgbm_objective(self, trial, X: pd.DataFrame, y: pd.Series) -> Union[float, List[float]]:
        """Objective function for LightGBM optimization."""
        config = LightGBMConfig(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 10, 300),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
            save_model=False
        )
        
        return self._evaluate_config(trial, LightGBMTrainer, config, X, y)
    
    def _evaluate_config(self, trial, trainer_class, config, X: pd.DataFrame, y: pd.Series) -> Union[float, List[float]]:
        """Evaluate a configuration using cross-validation."""
        try:
            # Start energy tracking
            if self.config.track_energy:
                self.energy_tracker.start()
            
            # Create trainer and train model
            trainer = trainer_class(config)
            result = trainer.train_and_evaluate(X, y, test_size=self.config.validation_split)
            
            if not result.success:
                # Return poor score for failed trials
                if self.config.use_multi_objective:
                    return [0.0, 1.0]  # Low accuracy, high energy
                else:
                    return 0.0
            
            # Get accuracy metric
            accuracy = result.test_metrics.get('roc_auc', 0.0)
            
            # Get energy efficiency metric
            energy_consumed = 0.0
            if self.config.track_energy:
                energy_consumed = self.energy_tracker.stop()
            
            # Calculate energy efficiency (inverse of energy consumption)
            energy_efficiency = 1.0 / (1.0 + energy_consumed * 1000)  # Scale for optimization
            
            # Report intermediate values for pruning
            if hasattr(result, 'training_metrics') and result.training_metrics:
                for i, metrics in enumerate(result.training_metrics):
                    trial.report(metrics.auc_roc, i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            # Return single or multiple objectives
            if self.config.use_multi_objective:
                return [accuracy, energy_efficiency]
            else:
                # Combine accuracy and energy efficiency for single objective
                combined_score = (accuracy * (1 - self.config.energy_weight) + 
                                energy_efficiency * self.config.energy_weight)
                return combined_score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            if self.config.use_multi_objective:
                return [0.0, 1.0]
            else:
                return 0.0
    
    def _run_optimization(self, objective: Callable, model_type: str, config_class: type) -> OptimizationResult:
        """Run the optimization process."""
        start_time = datetime.now()
        
        try:
            # Run optimization
            self.study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=self.config.n_jobs
            )
            
            # Extract results
            if self.config.use_multi_objective:
                # Multi-objective results
                best_trials = self.study.best_trials
                if best_trials:
                    best_trial = best_trials[0]  # First Pareto optimal solution
                    best_params = best_trial.params
                    best_values = best_trial.values
                    best_value = best_values[0] if best_values else 0.0
                    
                    # Extract Pareto front
                    pareto_front = [
                        {"params": trial.params, "values": trial.values}
                        for trial in best_trials
                    ]
                else:
                    best_params = {}
                    best_values = [0.0, 1.0]
                    best_value = 0.0
                    pareto_front = []
            else:
                # Single objective results
                best_trial = self.study.best_trial
                best_params = best_trial.params
                best_value = best_trial.value
                best_values = None
                pareto_front = None
            
            # Create best configuration (filter out non-config parameters)
            try:
                best_config = self._create_config_from_params(config_class, best_params)
            except Exception as e:
                logger.warning(f"Could not create config from best params: {e}")
                best_config = None
            
            # Calculate parameter importance
            try:
                importance = optuna.importance.get_param_importances(self.study)
            except Exception:
                importance = {}
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Save results if requested
            if self.config.save_results:
                self._save_results(model_type, best_params, best_value, best_values, importance)
            
            # Log completion
            audit_logger.log_model_operation(
                user_id="system",
                model_id=f"{model_type}_optimization",
                operation="hyperparameter_optimization_completed",
                success=True,
                details={
                    "optimization_time_seconds": optimization_time,
                    "n_trials": len(self.study.trials),
                    "best_value": best_value,
                    "best_params": best_params
                }
            )
            
            logger.info(f"{model_type.upper()} optimization completed in {optimization_time:.2f} seconds")
            logger.info(f"Best value: {best_value:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            return OptimizationResult(
                success=True,
                study=self.study,
                best_params=best_params,
                best_value=best_value,
                best_values=best_values,
                n_trials=len(self.study.trials),
                optimization_time_seconds=optimization_time,
                model_type=model_type,
                config_class=config_class,
                best_config=best_config,
                pareto_front=pareto_front,
                parameter_importance=importance,
                message=f"{model_type.upper()} optimization completed successfully"
            )
            
        except Exception as e:
            optimization_time = (datetime.now() - start_time).total_seconds()
            error_message = f"{model_type.upper()} optimization failed: {str(e)}"
            logger.error(error_message)
            
            return OptimizationResult(
                success=False,
                study=self.study,
                best_params={},
                best_value=0.0,
                best_values=None,
                n_trials=len(self.study.trials) if self.study else 0,
                optimization_time_seconds=optimization_time,
                model_type=model_type,
                config_class=config_class,
                best_config=None,
                pareto_front=None,
                parameter_importance={},
                message=error_message
            )
    
    def _save_results(self, model_type: str, best_params: Dict[str, Any], 
                     best_value: float, best_values: Optional[List[float]],
                     importance: Dict[str, float]):
        """Save optimization results to disk."""
        results_dir = Path(self.config.results_path)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save study
        study_file = results_dir / f"{model_type}_study.pkl"
        with open(study_file, 'wb') as f:
            import pickle
            pickle.dump(self.study, f)
        
        # Save results summary
        results = {
            "model_type": model_type,
            "best_params": best_params,
            "best_value": best_value,
            "best_values": best_values,
            "parameter_importance": importance,
            "n_trials": len(self.study.trials),
            "optimization_config": self.config.__dict__,
            "timestamp": datetime.now().isoformat()
        }
        
        results_file = results_dir / f"{model_type}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
    
    def _create_config_from_params(self, config_class: type, params: Dict[str, Any]) -> Any:
        """Create a config object from trial parameters, filtering out non-config params."""
        # Filter out parameters that are not valid for the config class
        filtered_params = {}
        
        if config_class == DNNConfig:
            # Handle DNN-specific parameter conversion
            n_layers = params.get('n_layers', 3)
            hidden_layers = [params.get(f'hidden_layer_{i}', 64) for i in range(n_layers)]
            filtered_params['hidden_layers'] = hidden_layers
            
            # Copy other valid parameters
            valid_params = ['dropout_rate', 'learning_rate', 'batch_size', 'optimizer', 
                          'activation', 'use_batch_norm', 'weight_decay', 'focal_alpha', 'focal_gamma']
            for param in valid_params:
                if param in params:
                    filtered_params[param] = params[param]
        
        elif config_class == LSTMConfig:
            # Copy valid LSTM parameters
            valid_params = ['hidden_size', 'num_layers', 'dropout_rate', 'bidirectional',
                          'use_attention', 'attention_dim', 'learning_rate', 'batch_size',
                          'optimizer', 'weight_decay', 'max_sequence_length']
            for param in valid_params:
                if param in params:
                    filtered_params[param] = params[param]
        
        elif config_class == GNNConfig:
            # Handle GNN-specific parameter conversion
            n_layers = params.get('num_layers', 3)
            hidden_dims = [params.get(f'hidden_dim_{i}', 64) for i in range(n_layers)]
            filtered_params['hidden_dims'] = hidden_dims
            
            # Copy other valid parameters
            valid_params = ['conv_type', 'num_heads', 'dropout_rate', 'k_neighbors',
                          'graph_construction_method', 'similarity_threshold', 'pooling_method',
                          'learning_rate', 'batch_size', 'optimizer', 'weight_decay']
            for param in valid_params:
                if param in params:
                    filtered_params[param] = params[param]
        
        elif config_class == TCNConfig:
            # Handle TCN-specific parameter conversion
            n_layers = params.get('num_layers', 3)
            num_channels = [params.get(f'channel_{i}', 64) for i in range(n_layers)]
            filtered_params['num_channels'] = num_channels
            
            # Copy other valid parameters
            valid_params = ['kernel_size', 'dropout_rate', 'dilation_base', 'activation',
                          'use_layer_norm', 'learning_rate', 'batch_size', 'optimizer',
                          'weight_decay', 'max_sequence_length']
            for param in valid_params:
                if param in params:
                    filtered_params[param] = params[param]
        
        elif config_class == LightGBMConfig:
            # Copy valid LightGBM parameters
            valid_params = ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves',
                          'min_child_samples', 'subsample', 'colsample_bytree', 'reg_alpha',
                          'reg_lambda', 'min_split_gain']
            for param in valid_params:
                if param in params:
                    filtered_params[param] = params[param]
        
        return config_class(**filtered_params)


# Utility functions
def optimize_all_models(X: pd.DataFrame, y: pd.Series, 
                       config: Optional[OptimizationConfig] = None) -> Dict[str, OptimizationResult]:
    """Optimize hyperparameters for all model types."""
    config = config or OptimizationConfig()
    results = {}
    
    model_optimizers = {
        "dnn": lambda opt: opt.optimize_dnn(X, y),
        "lstm": lambda opt: opt.optimize_lstm(X, y),
        "gnn": lambda opt: opt.optimize_gnn(X, y),
        "tcn": lambda opt: opt.optimize_tcn(X, y),
        "lightgbm": lambda opt: opt.optimize_lightgbm(X, y)
    }
    
    for model_type, optimize_func in model_optimizers.items():
        logger.info(f"Starting optimization for {model_type}")
        
        # Create separate optimizer for each model
        optimizer = HyperparameterOptimizer(config)
        result = optimize_func(optimizer)
        results[model_type] = result
        
        if result.success:
            logger.info(f"✓ {model_type} optimization completed successfully")
        else:
            logger.error(f"✗ {model_type} optimization failed: {result.message}")
    
    return results


def get_fast_optimization_config() -> OptimizationConfig:
    """Get fast optimization configuration for testing."""
    return OptimizationConfig(
        n_trials=10,
        cv_folds=2,
        max_epochs_per_trial=10,
        early_stopping_rounds=3,
        use_multi_objective=False,
        track_energy=False
    )


def get_production_optimization_config() -> OptimizationConfig:
    """Get production optimization configuration."""
    return OptimizationConfig(
        n_trials=200,
        cv_folds=5,
        max_epochs_per_trial=100,
        early_stopping_rounds=15,
        use_multi_objective=True,
        track_energy=True,
        n_jobs=4,
        timeout=3600 * 6  # 6 hours
    )