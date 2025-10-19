"""
Cross-validation strategies for credit risk modeling.
Implements stratified k-fold, time-series, and nested cross-validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
from abc import ABC, abstractmethod

# ML imports
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    TimeSeriesSplit,
    ParameterGrid,
    ParameterSampler,
    cross_val_score,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.base import BaseEstimator, clone
from scipy import stats
import joblib

from ..core.interfaces import DataProcessor
from ..core.config import get_config
from ..core.logging import get_logger, get_audit_logger


logger = get_logger(__name__)
audit_logger = get_audit_logger()


class CVStrategy(Enum):
    """Cross-validation strategies."""

    STRATIFIED_KFOLD = "stratified_kfold"
    KFOLD = "kfold"
    TIME_SERIES = "time_series"
    NESTED_CV = "nested_cv"
    CUSTOM = "custom"


@dataclass
class CVConfig:
    """Configuration for cross-validation."""

    # Basic CV parameters
    strategy: CVStrategy = CVStrategy.STRATIFIED_KFOLD
    n_splits: int = 5
    random_state: int = 42
    shuffle: bool = True

    # Stratified parameters
    stratify_column: Optional[str] = None

    # Time series parameters
    max_train_size: Optional[int] = None
    test_size: Optional[int] = None
    gap: int = 0  # Gap between train and test

    # Nested CV parameters
    inner_cv_splits: int = 3
    outer_cv_splits: int = 5

    # Scoring and evaluation
    scoring_metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1", "roc_auc"]
    )
    primary_metric: str = "roc_auc"

    # Statistical testing
    enable_statistical_tests: bool = True
    significance_level: float = 0.05

    # Parallel processing
    n_jobs: int = -1
    verbose: int = 1


@dataclass
class CVFold:
    """Single cross-validation fold."""

    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_size: int
    test_size: int
    stratification_info: Optional[Dict[str, Any]] = None


@dataclass
class CVResult:
    """Cross-validation result."""

    strategy: CVStrategy
    n_splits: int
    scores: Dict[str, List[float]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    fold_results: List[Dict[str, Any]]
    best_fold: int
    worst_fold: int
    statistical_tests: Dict[str, Any]
    cv_time_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NestedCVResult:
    """Nested cross-validation result."""

    outer_cv_scores: Dict[str, List[float]]
    inner_cv_scores: Dict[str, List[List[float]]]
    best_params_per_fold: List[Dict[str, Any]]
    best_estimators: List[BaseEstimator]
    generalization_score: Dict[str, float]
    hyperparameter_stability: Dict[str, float]
    cv_time_seconds: float


class BaseCrossValidator(ABC):
    """Abstract base class for cross-validators."""

    def __init__(self, config: CVConfig):
        self.config = config

    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series) -> Iterator[CVFold]:
        """Generate cross-validation folds."""
        pass

    @abstractmethod
    def validate_data(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Validate data for this CV strategy."""
        pass


class StratifiedKFoldValidator(BaseCrossValidator):
    """Stratified K-Fold cross-validation for imbalanced data."""

    def __init__(self, config: CVConfig):
        super().__init__(config)
        self.cv = StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=config.shuffle,
            random_state=config.random_state,
        )

    def split(self, X: pd.DataFrame, y: pd.Series) -> Iterator[CVFold]:
        """Generate stratified k-fold splits."""
        for fold_id, (train_idx, test_idx) in enumerate(self.cv.split(X, y)):
            # Calculate stratification info
            train_y = y.iloc[train_idx]
            test_y = y.iloc[test_idx]

            stratification_info = {
                "train_class_distribution": train_y.value_counts(
                    normalize=True
                ).to_dict(),
                "test_class_distribution": test_y.value_counts(
                    normalize=True
                ).to_dict(),
                "train_class_balance": self._calculate_balance_ratio(train_y),
                "test_class_balance": self._calculate_balance_ratio(test_y),
            }

            yield CVFold(
                fold_id=fold_id,
                train_indices=train_idx,
                test_indices=test_idx,
                train_size=len(train_idx),
                test_size=len(test_idx),
                stratification_info=stratification_info,
            )

    def validate_data(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Validate data for stratified k-fold."""
        # Check if we have enough samples per class
        class_counts = y.value_counts()
        min_class_count = class_counts.min()

        if min_class_count < self.config.n_splits:
            logger.error(
                f"Insufficient samples for stratification: min class has {min_class_count} samples, need at least {self.config.n_splits}"
            )
            return False

        return True

    def _calculate_balance_ratio(self, y: pd.Series) -> float:
        """Calculate class balance ratio (minority/majority)."""
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            return 1.0
        return class_counts.min() / class_counts.max()


class TimeSeriesValidator(BaseCrossValidator):
    """Time series cross-validation for temporal data."""

    def __init__(self, config: CVConfig):
        super().__init__(config)
        self.cv = TimeSeriesSplit(
            n_splits=config.n_splits,
            max_train_size=config.max_train_size,
            test_size=config.test_size,
            gap=config.gap,
        )

    def split(self, X: pd.DataFrame, y: pd.Series) -> Iterator[CVFold]:
        """Generate time series splits."""
        for fold_id, (train_idx, test_idx) in enumerate(self.cv.split(X)):
            # Calculate temporal info
            temporal_info = {
                "train_start_idx": train_idx[0],
                "train_end_idx": train_idx[-1],
                "test_start_idx": test_idx[0],
                "test_end_idx": test_idx[-1],
                "gap_size": (
                    test_idx[0] - train_idx[-1] - 1 if len(train_idx) > 0 else 0
                ),
            }

            yield CVFold(
                fold_id=fold_id,
                train_indices=train_idx,
                test_indices=test_idx,
                train_size=len(train_idx),
                test_size=len(test_idx),
                stratification_info=temporal_info,
            )

    def validate_data(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Validate data for time series CV."""
        # Check if we have enough samples for time series splits
        min_samples_needed = (self.config.n_splits + 1) * (self.config.test_size or 1)

        if len(X) < min_samples_needed:
            logger.error(
                f"Insufficient samples for time series CV: have {len(X)}, need at least {min_samples_needed}"
            )
            return False

        return True


class NestedCrossValidator:
    """Nested cross-validation for hyperparameter tuning."""

    def __init__(self, config: CVConfig):
        self.config = config
        self.outer_cv = StratifiedKFold(
            n_splits=config.outer_cv_splits,
            shuffle=config.shuffle,
            random_state=config.random_state,
        )
        self.inner_cv = StratifiedKFold(
            n_splits=config.inner_cv_splits,
            shuffle=config.shuffle,
            random_state=config.random_state + 1,
        )

    def validate_estimator(
        self,
        estimator: BaseEstimator,
        param_grid: Dict[str, List],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> NestedCVResult:
        """Perform nested cross-validation."""
        start_time = datetime.now()

        outer_scores = {metric: [] for metric in self.config.scoring_metrics}
        inner_scores = {metric: [] for metric in self.config.scoring_metrics}
        best_params_per_fold = []
        best_estimators = []

        logger.info(
            f"Starting nested CV with {self.config.outer_cv_splits} outer and {self.config.inner_cv_splits} inner folds"
        )

        for outer_fold, (train_outer_idx, test_outer_idx) in enumerate(
            self.outer_cv.split(X, y)
        ):
            logger.info(
                f"Processing outer fold {outer_fold + 1}/{self.config.outer_cv_splits}"
            )

            X_train_outer = X.iloc[train_outer_idx]
            y_train_outer = y.iloc[train_outer_idx]
            X_test_outer = X.iloc[test_outer_idx]
            y_test_outer = y.iloc[test_outer_idx]

            # Inner CV for hyperparameter tuning
            best_score = -np.inf
            best_params = None
            best_estimator = None
            fold_inner_scores = {metric: [] for metric in self.config.scoring_metrics}

            # Grid search over parameters
            for params in ParameterGrid(param_grid):
                # Inner CV with current parameters
                estimator_clone = clone(estimator)
                estimator_clone.set_params(**params)

                inner_fold_scores = {
                    metric: [] for metric in self.config.scoring_metrics
                }

                for train_inner_idx, test_inner_idx in self.inner_cv.split(
                    X_train_outer, y_train_outer
                ):
                    X_train_inner = X_train_outer.iloc[train_inner_idx]
                    y_train_inner = y_train_outer.iloc[train_inner_idx]
                    X_test_inner = X_train_outer.iloc[test_inner_idx]
                    y_test_inner = y_train_outer.iloc[test_inner_idx]

                    # Train and evaluate
                    estimator_clone.fit(X_train_inner, y_train_inner)
                    y_pred_inner = estimator_clone.predict(X_test_inner)

                    # Calculate scores
                    scores = self._calculate_scores(
                        y_test_inner, y_pred_inner, estimator_clone, X_test_inner
                    )
                    for metric, score in scores.items():
                        inner_fold_scores[metric].append(score)

                # Average inner CV scores
                avg_inner_scores = {
                    metric: np.mean(scores)
                    for metric, scores in inner_fold_scores.items()
                }

                # Check if this is the best parameter set
                primary_score = avg_inner_scores.get(self.config.primary_metric, 0)
                if primary_score > best_score:
                    best_score = primary_score
                    best_params = params
                    best_estimator = clone(estimator)
                    best_estimator.set_params(**params)
                    fold_inner_scores = inner_fold_scores

            # Train best estimator on full outer training set
            best_estimator.fit(X_train_outer, y_train_outer)
            y_pred_outer = best_estimator.predict(X_test_outer)

            # Calculate outer CV scores
            outer_fold_scores = self._calculate_scores(
                y_test_outer, y_pred_outer, best_estimator, X_test_outer
            )

            for metric, score in outer_fold_scores.items():
                outer_scores[metric].append(score)
                inner_scores[metric].append(fold_inner_scores[metric])

            best_params_per_fold.append(best_params)
            best_estimators.append(best_estimator)

        # Calculate generalization scores and hyperparameter stability
        generalization_score = {
            metric: np.mean(scores) for metric, scores in outer_scores.items()
        }
        hyperparameter_stability = self._calculate_hyperparameter_stability(
            best_params_per_fold
        )

        cv_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"Nested CV completed in {cv_time:.2f} seconds")

        return NestedCVResult(
            outer_cv_scores=outer_scores,
            inner_cv_scores=inner_scores,
            best_params_per_fold=best_params_per_fold,
            best_estimators=best_estimators,
            generalization_score=generalization_score,
            hyperparameter_stability=hyperparameter_stability,
            cv_time_seconds=cv_time,
        )

    def _calculate_scores(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        estimator: BaseEstimator,
        X_test: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate evaluation scores."""
        scores = {}

        try:
            if "accuracy" in self.config.scoring_metrics:
                scores["accuracy"] = accuracy_score(y_true, y_pred)

            if "precision" in self.config.scoring_metrics:
                scores["precision"] = precision_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )

            if "recall" in self.config.scoring_metrics:
                scores["recall"] = recall_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )

            if "f1" in self.config.scoring_metrics:
                scores["f1"] = f1_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )

            if "roc_auc" in self.config.scoring_metrics and hasattr(
                estimator, "predict_proba"
            ):
                try:
                    y_proba = estimator.predict_proba(X_test)
                    if y_proba.shape[1] == 2:  # Binary classification
                        scores["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                    else:  # Multiclass
                        scores["roc_auc"] = roc_auc_score(
                            y_true, y_proba, multi_class="ovr", average="weighted"
                        )
                except:
                    scores["roc_auc"] = 0.0

        except Exception as e:
            logger.warning(f"Score calculation failed: {e}")
            for metric in self.config.scoring_metrics:
                scores[metric] = 0.0

        return scores

    def _calculate_hyperparameter_stability(
        self, best_params_per_fold: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate hyperparameter stability across folds."""
        if not best_params_per_fold:
            return {}

        stability = {}

        # Get all parameter names
        all_param_names = set()
        for params in best_params_per_fold:
            all_param_names.update(params.keys())

        for param_name in all_param_names:
            param_values = [params.get(param_name) for params in best_params_per_fold]

            # Calculate stability (consistency across folds)
            if all(isinstance(v, (int, float)) for v in param_values if v is not None):
                # Numerical parameter - use coefficient of variation
                param_values = [v for v in param_values if v is not None]
                if param_values:
                    stability[param_name] = 1.0 - (
                        np.std(param_values) / (np.mean(param_values) + 1e-8)
                    )
            else:
                # Categorical parameter - use mode frequency
                param_counts = pd.Series(param_values).value_counts()
                stability[param_name] = (
                    param_counts.iloc[0] / len(param_values)
                    if len(param_counts) > 0
                    else 0.0
                )

        return stability


class CrossValidationManager(DataProcessor):
    """Main cross-validation manager."""

    def __init__(self, config: Optional[CVConfig] = None):
        self.config = config or CVConfig()
        self.validators = {
            CVStrategy.STRATIFIED_KFOLD: StratifiedKFoldValidator(self.config),
            CVStrategy.KFOLD: self._create_kfold_validator(),
            CVStrategy.TIME_SERIES: TimeSeriesValidator(self.config),
        }
        self.nested_validator = NestedCrossValidator(self.config)

    def _create_kfold_validator(self) -> BaseCrossValidator:
        """Create K-Fold validator."""

        class KFoldValidator(BaseCrossValidator):
            def __init__(self, config):
                super().__init__(config)
                self.cv = KFold(
                    n_splits=config.n_splits,
                    shuffle=config.shuffle,
                    random_state=config.random_state,
                )

            def split(self, X, y):
                for fold_id, (train_idx, test_idx) in enumerate(self.cv.split(X)):
                    yield CVFold(
                        fold_id=fold_id,
                        train_indices=train_idx,
                        test_indices=test_idx,
                        train_size=len(train_idx),
                        test_size=len(test_idx),
                    )

            def validate_data(self, X, y):
                return len(X) >= self.config.n_splits

        return KFoldValidator(self.config)

    def process(
        self,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, List]] = None,
    ) -> Union[CVResult, NestedCVResult]:
        """Process cross-validation."""
        start_time = datetime.now()

        try:
            logger.info(
                f"Starting cross-validation with strategy: {self.config.strategy.value}"
            )

            # Nested CV if parameter grid is provided
            if param_grid and self.config.strategy == CVStrategy.NESTED_CV:
                return self.nested_validator.validate_estimator(
                    estimator, param_grid, X, y
                )

            # Regular CV
            validator = self.validators.get(self.config.strategy)
            if not validator:
                raise ValueError(f"Unsupported CV strategy: {self.config.strategy}")

            # Validate data
            if not validator.validate_data(X, y):
                raise ValueError("Data validation failed for CV strategy")

            # Perform cross-validation
            scores = {metric: [] for metric in self.config.scoring_metrics}
            fold_results = []

            for fold in validator.split(X, y):
                logger.debug(
                    f"Processing fold {fold.fold_id + 1}/{self.config.n_splits}"
                )

                # Split data
                X_train = X.iloc[fold.train_indices]
                y_train = y.iloc[fold.train_indices]
                X_test = X.iloc[fold.test_indices]
                y_test = y.iloc[fold.test_indices]

                # Train and predict
                estimator_clone = clone(estimator)
                estimator_clone.fit(X_train, y_train)
                y_pred = estimator_clone.predict(X_test)

                # Calculate scores
                fold_scores = self.nested_validator._calculate_scores(
                    y_test, y_pred, estimator_clone, X_test
                )

                for metric, score in fold_scores.items():
                    scores[metric].append(score)

                fold_results.append(
                    {
                        "fold_id": fold.fold_id,
                        "train_size": fold.train_size,
                        "test_size": fold.test_size,
                        "scores": fold_scores,
                        "stratification_info": fold.stratification_info,
                    }
                )

            # Calculate summary statistics
            mean_scores = {
                metric: np.mean(score_list) for metric, score_list in scores.items()
            }
            std_scores = {
                metric: np.std(score_list) for metric, score_list in scores.items()
            }

            # Find best and worst folds
            primary_scores = scores[self.config.primary_metric]
            best_fold = int(np.argmax(primary_scores))
            worst_fold = int(np.argmin(primary_scores))

            # Statistical tests
            statistical_tests = {}
            if self.config.enable_statistical_tests:
                statistical_tests = self._perform_statistical_tests(scores)

            cv_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"Cross-validation completed in {cv_time:.2f} seconds")

            # Log CV results
            audit_logger.log_data_access(
                user_id="system",
                resource="cross_validation",
                action="model_validation",
                success=True,
                details={
                    "strategy": self.config.strategy.value,
                    "n_splits": self.config.n_splits,
                    "mean_score": mean_scores.get(self.config.primary_metric, 0.0),
                    "cv_time_seconds": cv_time,
                },
            )

            return CVResult(
                strategy=self.config.strategy,
                n_splits=self.config.n_splits,
                scores=scores,
                mean_scores=mean_scores,
                std_scores=std_scores,
                fold_results=fold_results,
                best_fold=best_fold,
                worst_fold=worst_fold,
                statistical_tests=statistical_tests,
                cv_time_seconds=cv_time,
            )

        except Exception as e:
            cv_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Cross-validation failed: {str(e)}"
            logger.error(error_message)

            return CVResult(
                strategy=self.config.strategy,
                n_splits=0,
                scores={},
                mean_scores={},
                std_scores={},
                fold_results=[],
                best_fold=-1,
                worst_fold=-1,
                statistical_tests={},
                cv_time_seconds=cv_time,
                metadata={"error": error_message},
            )

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate data for cross-validation."""
        try:
            if data.empty:
                logger.error("Data is empty")
                return False

            if len(data) < self.config.n_splits:
                logger.error(
                    f"Insufficient data: {len(data)} samples for {self.config.n_splits} folds"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"CV validation failed: {e}")
            return False

    def _perform_statistical_tests(
        self, scores: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Perform statistical tests on CV scores."""
        tests = {}

        try:
            for metric, score_list in scores.items():
                if len(score_list) > 1:
                    # Normality test
                    _, normality_p = stats.shapiro(score_list)

                    # Confidence interval
                    mean_score = np.mean(score_list)
                    std_score = np.std(score_list)
                    n = len(score_list)

                    # 95% confidence interval
                    ci_lower = mean_score - 1.96 * (std_score / np.sqrt(n))
                    ci_upper = mean_score + 1.96 * (std_score / np.sqrt(n))

                    tests[metric] = {
                        "normality_p_value": normality_p,
                        "is_normal": normality_p > self.config.significance_level,
                        "confidence_interval_95": (ci_lower, ci_upper),
                        "coefficient_of_variation": (
                            std_score / mean_score if mean_score != 0 else float("inf")
                        ),
                    }

        except Exception as e:
            logger.warning(f"Statistical tests failed: {e}")

        return tests


# Factory functions and utilities
def create_cv_manager(config: Optional[CVConfig] = None) -> CrossValidationManager:
    """Create a cross-validation manager instance."""
    return CrossValidationManager(config)


def validate_model_cv(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    strategy: CVStrategy = CVStrategy.STRATIFIED_KFOLD,
    n_splits: int = 5,
    config: Optional[CVConfig] = None,
) -> CVResult:
    """Convenience function to validate a model with cross-validation."""
    if config is None:
        config = CVConfig(strategy=strategy, n_splits=n_splits)

    cv_manager = create_cv_manager(config)
    return cv_manager.process(estimator, X, y)


def nested_cv_hyperparameter_tuning(
    estimator: BaseEstimator,
    param_grid: Dict[str, List],
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: int = 5,
    inner_cv: int = 3,
    config: Optional[CVConfig] = None,
) -> NestedCVResult:
    """Convenience function for nested CV hyperparameter tuning."""
    if config is None:
        config = CVConfig(
            strategy=CVStrategy.NESTED_CV,
            outer_cv_splits=outer_cv,
            inner_cv_splits=inner_cv,
        )

    cv_manager = create_cv_manager(config)
    return cv_manager.process(estimator, X, y, param_grid)


def get_default_cv_config() -> CVConfig:
    """Get default cross-validation configuration."""
    return CVConfig()


def get_imbalanced_cv_config() -> CVConfig:
    """Get CV configuration optimized for imbalanced datasets."""
    return CVConfig(
        strategy=CVStrategy.STRATIFIED_KFOLD,
        n_splits=5,
        scoring_metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
        primary_metric="f1",
        enable_statistical_tests=True,
    )


def get_time_series_cv_config(
    n_splits: int = 5, test_size: Optional[int] = None
) -> CVConfig:
    """Get CV configuration for time series data."""
    return CVConfig(
        strategy=CVStrategy.TIME_SERIES,
        n_splits=n_splits,
        test_size=test_size,
        gap=1,  # One time step gap
        shuffle=False,  # Never shuffle time series
        scoring_metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
        primary_metric="roc_auc",
    )


def get_nested_cv_config(outer_cv: int = 5, inner_cv: int = 3) -> CVConfig:
    """Get nested CV configuration for hyperparameter tuning."""
    return CVConfig(
        strategy=CVStrategy.NESTED_CV,
        outer_cv_splits=outer_cv,
        inner_cv_splits=inner_cv,
        scoring_metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
        primary_metric="roc_auc",
        enable_statistical_tests=True,
    )


class CVResultAnalyzer:
    """Analyzer for cross-validation results."""

    @staticmethod
    def analyze_cv_results(cv_result: CVResult) -> Dict[str, Any]:
        """Analyze cross-validation results."""
        analysis = {
            "summary": {
                "strategy": cv_result.strategy.value,
                "n_splits": cv_result.n_splits,
                "cv_time_seconds": cv_result.cv_time_seconds,
            },
            "performance": {},
            "stability": {},
            "recommendations": [],
        }

        # Performance analysis
        for metric, mean_score in cv_result.mean_scores.items():
            std_score = cv_result.std_scores.get(metric, 0)
            scores = cv_result.scores.get(metric, [])

            analysis["performance"][metric] = {
                "mean": mean_score,
                "std": std_score,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "cv": std_score / mean_score if mean_score != 0 else float("inf"),
            }

        # Stability analysis
        primary_scores = cv_result.scores.get(
            cv_result.metadata.get("primary_metric", "accuracy"), []
        )
        if primary_scores:
            analysis["stability"] = {
                "score_range": max(primary_scores) - min(primary_scores),
                "coefficient_of_variation": np.std(primary_scores)
                / np.mean(primary_scores),
                "is_stable": np.std(primary_scores) / np.mean(primary_scores) < 0.1,
            }

        # Generate recommendations
        recommendations = []

        # Check for high variance
        for metric, perf in analysis["performance"].items():
            if perf["cv"] > 0.2:  # High coefficient of variation
                recommendations.append(
                    f"High variance in {metric} scores - consider more data or regularization"
                )

        # Check for low performance
        primary_metric = cv_result.metadata.get("primary_metric", "accuracy")
        if primary_metric in analysis["performance"]:
            primary_perf = analysis["performance"][primary_metric]["mean"]
            if primary_perf < 0.7:
                recommendations.append(
                    "Low performance - consider feature engineering or different algorithms"
                )

        # Check stability
        if not analysis["stability"].get("is_stable", True):
            recommendations.append(
                "Unstable performance across folds - check for data leakage or overfitting"
            )

        analysis["recommendations"] = recommendations

        return analysis

    @staticmethod
    def analyze_nested_cv_results(nested_result: NestedCVResult) -> Dict[str, Any]:
        """Analyze nested cross-validation results."""
        analysis = {
            "generalization": nested_result.generalization_score,
            "hyperparameter_stability": nested_result.hyperparameter_stability,
            "cv_time_seconds": nested_result.cv_time_seconds,
            "parameter_analysis": {},
            "recommendations": [],
        }

        # Analyze parameter consistency
        if nested_result.best_params_per_fold:
            param_consistency = {}
            all_params = set()
            for params in nested_result.best_params_per_fold:
                all_params.update(params.keys())

            for param in all_params:
                values = [
                    params.get(param) for params in nested_result.best_params_per_fold
                ]
                unique_values = len(set(v for v in values if v is not None))
                param_consistency[param] = {
                    "unique_values": unique_values,
                    "consistency_ratio": 1.0
                    - (unique_values - 1) / len(nested_result.best_params_per_fold),
                    "most_common": (
                        max(set(values), key=values.count) if values else None
                    ),
                }

            analysis["parameter_analysis"] = param_consistency

        # Generate recommendations
        recommendations = []

        # Check generalization
        primary_score = (
            list(nested_result.generalization_score.values())[0]
            if nested_result.generalization_score
            else 0
        )
        if primary_score < 0.7:
            recommendations.append(
                "Low generalization performance - consider simpler models or more data"
            )

        # Check parameter stability
        unstable_params = [
            param
            for param, stability in nested_result.hyperparameter_stability.items()
            if stability < 0.5
        ]
        if unstable_params:
            recommendations.append(
                f"Unstable hyperparameters: {unstable_params} - consider fixing these parameters"
            )

        analysis["recommendations"] = recommendations

        return analysis

    @staticmethod
    def compare_cv_results(
        results: List[CVResult], labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple cross-validation results."""
        if not results:
            return {}

        labels = labels or [f"Model_{i+1}" for i in range(len(results))]

        comparison = {
            "models": labels,
            "performance_comparison": {},
            "statistical_tests": {},
            "recommendations": [],
        }

        # Performance comparison
        metrics = set()
        for result in results:
            metrics.update(result.mean_scores.keys())

        for metric in metrics:
            metric_data = {"means": [], "stds": [], "scores": []}

            for result in results:
                metric_data["means"].append(result.mean_scores.get(metric, 0))
                metric_data["stds"].append(result.std_scores.get(metric, 0))
                metric_data["scores"].append(result.scores.get(metric, []))

            comparison["performance_comparison"][metric] = metric_data

        # Statistical significance tests
        for metric in metrics:
            scores_lists = [result.scores.get(metric, []) for result in results]
            valid_scores = [scores for scores in scores_lists if len(scores) > 1]

            if len(valid_scores) >= 2:
                try:
                    # Perform ANOVA or Kruskal-Wallis test
                    if all(len(scores) >= 3 for scores in valid_scores):
                        # Check normality
                        normal_tests = [
                            stats.shapiro(scores)[1] > 0.05 for scores in valid_scores
                        ]

                        if all(normal_tests):
                            # ANOVA for normal distributions
                            f_stat, p_value = stats.f_oneway(*valid_scores)
                            test_name = "ANOVA"
                        else:
                            # Kruskal-Wallis for non-normal distributions
                            h_stat, p_value = stats.kruskal(*valid_scores)
                            test_name = "Kruskal-Wallis"

                        comparison["statistical_tests"][metric] = {
                            "test": test_name,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                        }
                except Exception as e:
                    logger.warning(f"Statistical test failed for {metric}: {e}")

        # Generate recommendations
        recommendations = []

        # Find best performing model
        primary_metric = "accuracy"  # Default
        if results and results[0].scores:
            primary_metric = list(results[0].scores.keys())[0]

        if primary_metric in comparison["performance_comparison"]:
            means = comparison["performance_comparison"][primary_metric]["means"]
            best_idx = np.argmax(means)
            recommendations.append(
                f"Best performing model: {labels[best_idx]} ({primary_metric}: {means[best_idx]:.3f})"
            )

        # Check for significant differences
        significant_metrics = [
            metric
            for metric, test in comparison["statistical_tests"].items()
            if test.get("significant", False)
        ]
        if significant_metrics:
            recommendations.append(
                f"Significant performance differences found in: {significant_metrics}"
            )

        comparison["recommendations"] = recommendations

        return comparison


def export_cv_results(
    cv_result: Union[CVResult, NestedCVResult],
    file_path: str,
    include_analysis: bool = True,
) -> bool:
    """Export cross-validation results to file."""
    try:
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "cv_result": (
                cv_result.__dict__ if hasattr(cv_result, "__dict__") else str(cv_result)
            ),
        }

        # Add analysis if requested
        if include_analysis:
            if isinstance(cv_result, CVResult):
                export_data["analysis"] = CVResultAnalyzer.analyze_cv_results(cv_result)
            elif isinstance(cv_result, NestedCVResult):
                export_data["analysis"] = CVResultAnalyzer.analyze_nested_cv_results(
                    cv_result
                )

        # Convert numpy arrays and other non-serializable objects
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif hasattr(obj, "__dict__"):
                return obj.__dict__
            else:
                return str(obj)

        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2, default=convert_for_json)

        logger.info(f"CV results exported to {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export CV results: {e}")
        return False
