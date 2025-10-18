"""
LightGBM baseline model for credit risk prediction.
Includes hyperparameter optimization, feature importance analysis, and benchmarking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import json
from pathlib import Path

# LightGBM imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

# ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
import optuna

try:
    from ..core.interfaces import BaseModel
    from ..core.config import get_config
    from ..core.logging import get_logger, get_audit_logger
    from ..data.cross_validation import validate_model_cv, get_imbalanced_cv_config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from core.interfaces import BaseModel
    from core.logging import get_logger, get_audit_logger
    
    # Create minimal implementations for testing
    class MockAuditLogger:
        def log_model_operation(self, **kwargs):
            pass
    
    def get_audit_logger():
        return MockAuditLogger()
    
    # Mock cross-validation functions
    def validate_model_cv(model, X, y, config=None):
        return type('MockResult', (), {
            'mean_scores': {'roc_auc': 0.8},
            'std_scores': {'roc_auc': 0.1},
            'cv_time_seconds': 1.0
        })()
    
    def get_imbalanced_cv_config():
        return type('MockConfig', (), {'n_splits': 5})()


logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class LightGBMConfig:
    """Configuration for LightGBM model."""
    # Model parameters
    objective: str = 'binary'
    metric: str = 'binary_logloss'
    boosting_type: str = 'gbdt'
    num_leaves: int = 31
    learning_rate: float = 0.05
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_child_samples: int = 20
    min_child_weight: float = 0.001
    min_split_gain: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    
    # Training parameters
    num_boost_round: int = 1000
    early_stopping_rounds: int = 100
    verbose: int = -1
    
    # Hyperparameter optimization
    enable_hyperopt: bool = True
    n_trials: int = 100
    timeout_seconds: Optional[int] = 3600  # 1 hour
    
    # Cross-validation
    cv_folds: int = 5
    stratified: bool = True
    
    # Feature importance
    importance_type: str = 'gain'  # 'split', 'gain'
    
    # Model saving
    save_model: bool = True
    model_path: str = "models/lightgbm"


@dataclass
class LightGBMResult:
    """Result of LightGBM training and evaluation."""
    success: bool
    model: Optional[Any]  # LightGBM model
    best_params: Dict[str, Any]
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    cv_results: Optional[Dict[str, Any]]
    training_time_seconds: float
    model_path: Optional[str]
    message: str


class LightGBMOptimizer:
    """Hyperparameter optimizer for LightGBM using Optuna."""
    
    def __init__(self, config: LightGBMConfig):
        self.config = config
        self.study = None
        self.best_params = None
    
    def optimize(self, X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        if not self.config.enable_hyperopt:
            return self._get_default_params()
        
        try:
            # Create study
            study_name = f"lightgbm_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            # Define objective function
            def objective(trial):
                params = self._suggest_params(trial)
                
                # Create datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Train model
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=self.config.num_boost_round,
                    callbacks=[
                        lgb.early_stopping(self.config.early_stopping_rounds),
                        lgb.log_evaluation(0)  # Suppress output
                    ]
                )
                
                # Predict and evaluate
                y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
                auc_score = roc_auc_score(y_val, y_pred_proba)
                
                return auc_score
            
            # Optimize
            logger.info(f"Starting hyperparameter optimization with {self.config.n_trials} trials")
            self.study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_seconds,
                show_progress_bar=False
            )
            
            self.best_params = self.study.best_params
            logger.info(f"Optimization completed. Best AUC: {self.study.best_value:.4f}")
            
            return self.best_params
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            return self._get_default_params()
    
    def _suggest_params(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for optimization."""
        params = {
            'objective': self.config.objective,
            'metric': self.config.metric,
            'boosting_type': self.config.boosting_type,
            'verbose': -1,
            
            # Hyperparameters to optimize
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        }
        
        return params
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters when optimization is disabled."""
        return {
            'objective': self.config.objective,
            'metric': self.config.metric,
            'boosting_type': self.config.boosting_type,
            'num_leaves': self.config.num_leaves,
            'learning_rate': self.config.learning_rate,
            'feature_fraction': self.config.feature_fraction,
            'bagging_fraction': self.config.bagging_fraction,
            'bagging_freq': self.config.bagging_freq,
            'min_child_samples': self.config.min_child_samples,
            'min_child_weight': self.config.min_child_weight,
            'min_split_gain': self.config.min_split_gain,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'verbose': self.config.verbose
        }
    
    def get_optimization_history(self) -> Optional[pd.DataFrame]:
        """Get optimization history as DataFrame."""
        if self.study is None:
            return None
        
        trials_df = self.study.trials_dataframe()
        return trials_df


class LightGBMModel(BaseEstimator):
    """LightGBM model implementation with scikit-learn compatibility."""
    
    def __init__(self, config: Optional[LightGBMConfig] = None):
        self.config = config or LightGBMConfig()
        self.model = None
        self.optimizer = LightGBMOptimizer(self.config)
        self.feature_names = None
        self.label_encoder = None
        self.is_trained = False
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Please install it with: pip install lightgbm")
    
    def get_params(self, deep=True):
        """Get parameters for scikit-learn compatibility."""
        return {'config': self.config}
    
    def set_params(self, **params):
        """Set parameters for scikit-learn compatibility."""
        if 'config' in params:
            self.config = params['config']
        return self
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> 'LightGBMModel':
        """Train the LightGBM model."""
        try:
            logger.info("Starting LightGBM training")
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Encode labels if necessary
            if y.dtype == 'object':
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                if y_val is not None:
                    y_val_encoded = self.label_encoder.transform(y_val)
                else:
                    y_val_encoded = None
            else:
                y_encoded = y
                y_val_encoded = y_val
            
            # Create validation split if not provided
            if X_val is None or y_val is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
            else:
                X_train, y_train = X, y_encoded
                y_val = y_val_encoded
            
            # Optimize hyperparameters
            best_params = self.optimizer.optimize(X_train, y_train, X_val, y_val)
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train final model
            logger.info("Training final model with optimized parameters")
            self.model = lgb.train(
                best_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=self.config.num_boost_round,
                callbacks=[
                    lgb.early_stopping(self.config.early_stopping_rounds),
                    lgb.log_evaluation(100) if self.config.verbose > 0 else lgb.log_evaluation(0)
                ]
            )
            
            self.is_trained = True
            logger.info(f"LightGBM training completed. Best iteration: {self.model.best_iteration}")
            
            return self
            
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure feature order matches training
        if self.feature_names:
            X = X[self.feature_names]
        
        # Get probabilities
        y_pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Convert to binary predictions
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Decode labels if necessary
        if self.label_encoder:
            y_pred = self.label_encoder.inverse_transform(y_pred)
        
        return y_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure feature order matches training
        if self.feature_names:
            X = X[self.feature_names]
        
        # Get probabilities
        y_pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Return as 2D array for binary classification
        return np.column_stack([1 - y_pred_proba, y_pred_proba])
    
    def get_feature_importance(self, importance_type: Optional[str] = None) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_type = importance_type or self.config.importance_type
        
        # Get importance scores
        importance_scores = self.model.feature_importance(importance_type=importance_type)
        
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
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Create save path
        save_path = path or self.config.model_path
        model_dir = Path(save_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_dir / "lightgbm_model.txt"
        self.model.save_model(str(model_file))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'config': self.config.__dict__,
            'best_params': self.optimizer.best_params,
            'best_iteration': self.model.best_iteration,
            'num_features': self.model.num_feature(),
            'objective': self.model.params.get('objective'),
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {model_file}")
        return str(model_file)
    
    def load_model(self, path: str) -> 'LightGBMModel':
        """Load a trained model."""
        model_path = Path(path)
        
        if model_path.is_file():
            # Path is the model file
            model_file = model_path
            metadata_file = model_path.parent / "metadata.json"
        else:
            # Path is the directory
            model_file = model_path / "lightgbm_model.txt"
            metadata_file = model_path / "metadata.json"
        
        # Load model
        self.model = lgb.Booster(model_file=str(model_file))
        
        # Load metadata if available
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata.get('feature_names')
            if 'best_params' in metadata:
                self.optimizer.best_params = metadata['best_params']
        
        self.is_trained = True
        logger.info(f"Model loaded from {model_file}")
        
        return self


class LightGBMTrainer:
    """High-level trainer for LightGBM models."""
    
    def __init__(self, config: Optional[LightGBMConfig] = None):
        self.config = config or LightGBMConfig()
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.2) -> LightGBMResult:
        """Train and evaluate LightGBM model."""
        start_time = datetime.now()
        
        try:
            logger.info("Starting LightGBM training and evaluation")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Further split training data for validation
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            logger.info(f"Data split - Train: {len(X_train_split)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Create and train model
            model = LightGBMModel(self.config)
            model.fit(X_train_split, y_train_split, X_val, y_val)
            
            # Get best parameters
            best_params = model.optimizer.best_params or model.optimizer._get_default_params()
            
            # Evaluate on different sets
            training_metrics = self._evaluate_model(model, X_train_split, y_train_split, "Training")
            validation_metrics = self._evaluate_model(model, X_val, y_val, "Validation")
            test_metrics = self._evaluate_model(model, X_test, y_test, "Test")
            
            # Get feature importance
            feature_importance = model.get_feature_importance()
            
            # Cross-validation
            cv_results = None
            if self.config.cv_folds > 1:
                cv_results = self._perform_cross_validation(X_train, y_train, best_params)
            
            # Save model if requested
            model_path = None
            if self.config.save_model:
                model_path = model.save_model()
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Log training completion
            audit_logger.log_model_operation(
                user_id="system",
                model_id="lightgbm_baseline",
                operation="training_completed",
                success=True,
                details={
                    "training_time_seconds": training_time,
                    "test_auc": test_metrics.get('roc_auc', 0.0),
                    "best_iteration": model.model.best_iteration
                }
            )
            
            logger.info(f"LightGBM training completed in {training_time:.2f} seconds")
            
            return LightGBMResult(
                success=True,
                model=model,
                best_params=best_params,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                feature_importance=feature_importance,
                cv_results=cv_results,
                training_time_seconds=training_time,
                model_path=model_path,
                message="LightGBM training completed successfully"
            )
            
        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            error_message = f"LightGBM training failed: {str(e)}"
            logger.error(error_message)
            
            return LightGBMResult(
                success=False,
                model=None,
                best_params={},
                training_metrics={},
                validation_metrics={},
                test_metrics={},
                feature_importance={},
                cv_results=None,
                training_time_seconds=training_time,
                model_path=None,
                message=error_message
            )
    
    def _evaluate_model(self, model: LightGBMModel, X: pd.DataFrame, y: pd.Series, 
                       dataset_name: str) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        try:
            # Make predictions
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0),
                'roc_auc': roc_auc_score(y, y_pred_proba)
            }
            
            logger.info(f"{dataset_name} Metrics - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed for {dataset_name}: {e}")
            return {}
    
    def _perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, 
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-validation."""
        try:
            logger.info(f"Performing {self.config.cv_folds}-fold cross-validation")
            
            # Create temporary model for CV
            temp_model = LightGBMModel(self.config)
            temp_model.optimizer.best_params = params
            
            # Use our cross-validation system
            cv_config = get_imbalanced_cv_config()
            cv_config.n_splits = self.config.cv_folds
            
            cv_result = validate_model_cv(temp_model, X, y, config=cv_config)
            
            cv_results = {
                'mean_scores': cv_result.mean_scores,
                'std_scores': cv_result.std_scores,
                'cv_time_seconds': cv_result.cv_time_seconds
            }
            
            logger.info(f"CV Results - AUC: {cv_results['mean_scores'].get('roc_auc', 0):.4f} ± {cv_results['std_scores'].get('roc_auc', 0):.4f}")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {}


# Factory functions and utilities
def create_lightgbm_model(config: Optional[LightGBMConfig] = None) -> LightGBMModel:
    """Create a LightGBM model instance."""
    return LightGBMModel(config)


def train_lightgbm_baseline(X: pd.DataFrame, y: pd.Series, 
                           config: Optional[LightGBMConfig] = None) -> LightGBMResult:
    """Convenience function to train LightGBM baseline."""
    trainer = LightGBMTrainer(config)
    return trainer.train_and_evaluate(X, y)


def get_default_lightgbm_config() -> LightGBMConfig:
    """Get default LightGBM configuration."""
    return LightGBMConfig()


def get_fast_lightgbm_config() -> LightGBMConfig:
    """Get fast LightGBM configuration for testing."""
    return LightGBMConfig(
        num_boost_round=100,
        early_stopping_rounds=20,
        enable_hyperopt=False,
        cv_folds=3
    )


def get_optimized_lightgbm_config() -> LightGBMConfig:
    """Get optimized LightGBM configuration for production."""
    return LightGBMConfig(
        num_boost_round=2000,
        early_stopping_rounds=200,
        enable_hyperopt=True,
        n_trials=200,
        timeout_seconds=7200,  # 2 hours
        cv_folds=5
    )