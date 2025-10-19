"""
Feature selection and importance analysis for credit risk modeling.
Implements statistical methods, correlation analysis, and recursive feature elimination.
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from ..core.config import get_config
from ..core.interfaces import DataProcessor
from ..core.logging import get_audit_logger, get_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection."""

    # Statistical methods
    enable_statistical_selection: bool = True
    statistical_method: str = "mutual_info"  # "chi2", "f_classif", "mutual_info"
    statistical_k_features: int = 20
    statistical_percentile: float = 50.0

    # Correlation analysis
    enable_correlation_analysis: bool = True
    correlation_threshold: float = 0.95
    correlation_method: str = "pearson"  # "pearson", "spearman"

    # Recursive feature elimination
    enable_rfe: bool = True
    rfe_estimator: str = "random_forest"  # "random_forest", "logistic_regression"
    rfe_n_features: Optional[int] = None  # If None, uses cross-validation
    rfe_cv_folds: int = 5

    # Model-based selection
    enable_model_based_selection: bool = True
    model_based_estimator: str = "random_forest"
    model_based_threshold: str = "median"  # "mean", "median", or float value

    # Variance threshold
    enable_variance_threshold: bool = True
    variance_threshold: float = 0.01

    # Feature importance ranking
    importance_methods: List[str] = field(
        default_factory=lambda: ["mutual_info", "correlation", "model_based"]
    )
    top_k_features: int = 30


@dataclass
class FeatureImportanceScore:
    """Feature importance score from different methods."""

    feature_name: str
    mutual_info_score: float = 0.0
    correlation_score: float = 0.0
    model_importance_score: float = 0.0
    rfe_ranking: int = 0
    combined_score: float = 0.0
    selected: bool = False


@dataclass
class FeatureSelectionResult:
    """Result of feature selection process."""

    success: bool
    selected_features: List[str]
    feature_scores: List[FeatureImportanceScore]
    correlation_matrix: Optional[pd.DataFrame]
    multicollinearity_pairs: List[Tuple[str, str, float]]
    removed_features: Dict[str, str]  # feature -> reason
    selection_summary: Dict[str, Any]
    processing_time_seconds: float
    message: str


class StatisticalFeatureSelector:
    """Statistical feature selection methods."""

    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        self.selectors = {
            "chi2": SelectKBest(chi2, k=config.statistical_k_features),
            "f_classif": SelectKBest(f_classif, k=config.statistical_k_features),
            "mutual_info": SelectKBest(
                mutual_info_classif, k=config.statistical_k_features
            ),
        }

    def select_features(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select features using statistical methods."""
        if not self.config.enable_statistical_selection:
            return list(X.columns), {}

        method = self.config.statistical_method
        selector = self.selectors.get(method)

        if selector is None:
            logger.warning(f"Unknown statistical method: {method}")
            return list(X.columns), {}

        try:
            # Ensure non-negative values for chi2
            if method == "chi2":
                X_processed = X.copy()
                # Make all values non-negative
                for col in X_processed.select_dtypes(include=[np.number]).columns:
                    X_processed[col] = X_processed[col] - X_processed[col].min() + 1e-6
            else:
                X_processed = X

            # Fit selector
            selector.fit(X_processed, y)

            # Get selected features
            selected_mask = selector.get_support()
            selected_features = X.columns[selected_mask].tolist()

            # Get feature scores
            scores = selector.scores_
            feature_scores = dict(zip(X.columns, scores))

            logger.info(
                f"Statistical selection ({method}): {len(selected_features)}/{len(X.columns)} features selected"
            )

            return selected_features, feature_scores

        except Exception as e:
            logger.error(f"Statistical feature selection failed: {e}")
            return list(X.columns), {}


class CorrelationAnalyzer:
    """Correlation analysis and multicollinearity detection."""

    def __init__(self, config: FeatureSelectionConfig):
        self.config = config

    def analyze_correlations(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]], Dict[str, float]]:
        """Analyze feature correlations and detect multicollinearity."""
        if not self.config.enable_correlation_analysis:
            return pd.DataFrame(), [], {}

        # Calculate correlation matrix
        numeric_features = X.select_dtypes(include=[np.number])

        if self.config.correlation_method == "pearson":
            corr_matrix = numeric_features.corr()
        else:  # spearman
            corr_matrix = numeric_features.corr(method="spearman")

        # Find highly correlated pairs
        multicollinear_pairs = []
        threshold = self.config.correlation_threshold

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > threshold:
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    multicollinear_pairs.append((feature1, feature2, corr_value))

        # Calculate target correlations
        target_correlations = {}
        for feature in numeric_features.columns:
            try:
                if self.config.correlation_method == "pearson":
                    corr, _ = pearsonr(numeric_features[feature], y)
                else:
                    corr, _ = spearmanr(numeric_features[feature], y)
                target_correlations[feature] = abs(corr)
            except:
                target_correlations[feature] = 0.0

        logger.info(
            f"Found {len(multicollinear_pairs)} highly correlated feature pairs"
        )

        return corr_matrix, multicollinear_pairs, target_correlations

    def remove_multicollinear_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        multicollinear_pairs: List[Tuple[str, str, float]],
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Remove multicollinear features, keeping the one with higher target correlation."""
        removed_features = {}
        features_to_remove = set()

        # Calculate target correlations for decision making
        target_correlations = {}
        numeric_features = X.select_dtypes(include=[np.number])

        for feature in numeric_features.columns:
            try:
                corr, _ = pearsonr(numeric_features[feature], y)
                target_correlations[feature] = abs(corr)
            except:
                target_correlations[feature] = 0.0

        # Decide which features to remove
        for feature1, feature2, corr_value in multicollinear_pairs:
            if feature1 in features_to_remove or feature2 in features_to_remove:
                continue

            # Keep the feature with higher target correlation
            corr1 = target_correlations.get(feature1, 0)
            corr2 = target_correlations.get(feature2, 0)

            if corr1 >= corr2:
                features_to_remove.add(feature2)
                removed_features[feature2] = (
                    f"Multicollinear with {feature1} (corr={corr_value:.3f})"
                )
            else:
                features_to_remove.add(feature1)
                removed_features[feature1] = (
                    f"Multicollinear with {feature2} (corr={corr_value:.3f})"
                )

        # Remove features
        X_filtered = X.drop(columns=list(features_to_remove))

        logger.info(f"Removed {len(features_to_remove)} multicollinear features")

        return X_filtered, removed_features


class RecursiveFeatureEliminator:
    """Recursive feature elimination with cross-validation."""

    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        self.estimators = {
            "random_forest": RandomForestClassifier(n_estimators=50, random_state=42),
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        }

    def select_features(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[List[str], Dict[str, int]]:
        """Select features using recursive feature elimination."""
        if not self.config.enable_rfe:
            return list(X.columns), {}

        estimator = self.estimators.get(self.config.rfe_estimator)
        if estimator is None:
            logger.warning(f"Unknown RFE estimator: {self.config.rfe_estimator}")
            return list(X.columns), {}

        try:
            # Use RFECV if n_features is not specified
            if self.config.rfe_n_features is None:
                selector = RFECV(
                    estimator=estimator,
                    cv=self.config.rfe_cv_folds,
                    scoring="roc_auc",
                    n_jobs=-1,
                )
            else:
                selector = RFE(
                    estimator=estimator, n_features_to_select=self.config.rfe_n_features
                )

            # Fit selector
            selector.fit(X, y)

            # Get selected features
            selected_mask = selector.support_
            selected_features = X.columns[selected_mask].tolist()

            # Get feature rankings
            feature_rankings = dict(zip(X.columns, selector.ranking_))

            logger.info(
                f"RFE selection: {len(selected_features)}/{len(X.columns)} features selected"
            )

            return selected_features, feature_rankings

        except Exception as e:
            logger.error(f"RFE feature selection failed: {e}")
            return list(X.columns), {}


class ModelBasedFeatureSelector:
    """Model-based feature selection using feature importance."""

    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        self.estimators = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "logistic_regression": LogisticRegression(
                random_state=42, max_iter=1000, penalty="l1", solver="liblinear"
            ),
        }

    def select_features(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select features using model-based importance."""
        if not self.config.enable_model_based_selection:
            return list(X.columns), {}

        estimator = self.estimators.get(self.config.model_based_estimator)
        if estimator is None:
            logger.warning(
                f"Unknown model-based estimator: {self.config.model_based_estimator}"
            )
            return list(X.columns), {}

        try:
            # Create selector
            selector = SelectFromModel(
                estimator=estimator, threshold=self.config.model_based_threshold
            )

            # Fit selector
            selector.fit(X, y)

            # Get selected features
            selected_mask = selector.get_support()
            selected_features = X.columns[selected_mask].tolist()

            # Get feature importance scores
            if hasattr(selector.estimator_, "feature_importances_"):
                importance_scores = selector.estimator_.feature_importances_
            elif hasattr(selector.estimator_, "coef_"):
                importance_scores = np.abs(selector.estimator_.coef_[0])
            else:
                importance_scores = np.ones(len(X.columns))

            feature_importance = dict(zip(X.columns, importance_scores))

            logger.info(
                f"Model-based selection: {len(selected_features)}/{len(X.columns)} features selected"
            )

            return selected_features, feature_importance

        except Exception as e:
            logger.error(f"Model-based feature selection failed: {e}")
            return list(X.columns), {}


class FeatureSelectionPipeline(DataProcessor):
    """Main feature selection pipeline."""

    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        self.config = config or FeatureSelectionConfig()
        self.statistical_selector = StatisticalFeatureSelector(self.config)
        self.correlation_analyzer = CorrelationAnalyzer(self.config)
        self.rfe_selector = RecursiveFeatureEliminator(self.config)
        self.model_selector = ModelBasedFeatureSelector(self.config)

    def process(self, X: pd.DataFrame, y: pd.Series) -> FeatureSelectionResult:
        """Process feature selection pipeline."""
        start_time = datetime.now()

        try:
            logger.info("Starting feature selection pipeline")
            logger.info(f"Input features: {X.shape[1]}")

            # Initialize tracking
            all_feature_scores = {
                feature: FeatureImportanceScore(feature_name=feature)
                for feature in X.columns
            }
            removed_features = {}
            X_current = X.copy()

            # 1. Remove low variance features
            if self.config.enable_variance_threshold:
                X_current, low_var_removed = self._remove_low_variance_features(
                    X_current
                )
                removed_features.update(low_var_removed)

            # 2. Correlation analysis and multicollinearity removal
            (
                corr_matrix,
                multicollinear_pairs,
                target_correlations,
            ) = self.correlation_analyzer.analyze_correlations(X_current, y)

            if multicollinear_pairs:
                (
                    X_current,
                    multicoll_removed,
                ) = self.correlation_analyzer.remove_multicollinear_features(
                    X_current, y, multicollinear_pairs
                )
                removed_features.update(multicoll_removed)

            # Update correlation scores
            for feature, score in target_correlations.items():
                if feature in all_feature_scores:
                    all_feature_scores[feature].correlation_score = score

            # 3. Statistical feature selection
            (
                statistical_features,
                statistical_scores,
            ) = self.statistical_selector.select_features(X_current, y)

            # Update statistical scores
            for feature, score in statistical_scores.items():
                if feature in all_feature_scores:
                    all_feature_scores[feature].mutual_info_score = score

            # 4. Recursive feature elimination
            rfe_features, rfe_rankings = self.rfe_selector.select_features(X_current, y)

            # Update RFE rankings
            for feature, ranking in rfe_rankings.items():
                if feature in all_feature_scores:
                    all_feature_scores[feature].rfe_ranking = ranking

            # 5. Model-based feature selection
            model_features, model_importance = self.model_selector.select_features(
                X_current, y
            )

            # Update model importance scores
            for feature, importance in model_importance.items():
                if feature in all_feature_scores:
                    all_feature_scores[feature].model_importance_score = importance

            # 6. Combine selection results
            selected_features = self._combine_selection_results(
                X_current.columns.tolist(),
                statistical_features,
                rfe_features,
                model_features,
                all_feature_scores,
            )

            # 7. Calculate combined scores and final ranking
            feature_scores_list = self._calculate_combined_scores(
                all_feature_scores, selected_features
            )

            # Create selection summary
            selection_summary = {
                "original_features": X.shape[1],
                "after_variance_threshold": (
                    X_current.shape[1]
                    if self.config.enable_variance_threshold
                    else X.shape[1]
                ),
                "after_correlation_filter": len(X_current.columns),
                "statistical_selected": len(statistical_features),
                "rfe_selected": len(rfe_features),
                "model_based_selected": len(model_features),
                "final_selected": len(selected_features),
                "removed_count": len(removed_features),
            }

            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Feature selection completed: {len(selected_features)}/{X.shape[1]} features selected"
            )

            # Log feature selection
            audit_logger.log_data_access(
                user_id="system",
                resource="feature_selection_pipeline",
                action="feature_selection",
                success=True,
                details={
                    "original_features": X.shape[1],
                    "selected_features": len(selected_features),
                    "processing_time_seconds": processing_time,
                },
            )

            return FeatureSelectionResult(
                success=True,
                selected_features=selected_features,
                feature_scores=feature_scores_list,
                correlation_matrix=corr_matrix,
                multicollinearity_pairs=multicollinear_pairs,
                removed_features=removed_features,
                selection_summary=selection_summary,
                processing_time_seconds=processing_time,
                message="Feature selection completed successfully",
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Feature selection failed: {str(e)}"

            logger.error(error_message)

            return FeatureSelectionResult(
                success=False,
                selected_features=[],
                feature_scores=[],
                correlation_matrix=None,
                multicollinearity_pairs=[],
                removed_features={},
                selection_summary={},
                processing_time_seconds=processing_time,
                message=error_message,
            )

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate selected features."""
        try:
            # Check if we have any features left
            if data.shape[1] == 0:
                logger.error("No features remaining after selection")
                return False

            # Check for constant features
            numeric_features = data.select_dtypes(include=[np.number])
            constant_features = numeric_features.columns[numeric_features.var() == 0]

            if len(constant_features) > 0:
                logger.warning(f"Constant features detected: {list(constant_features)}")

            return True

        except Exception as e:
            logger.error(f"Feature selection validation failed: {e}")
            return False

    def _remove_low_variance_features(
        self, X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Remove features with low variance."""
        selector = VarianceThreshold(threshold=self.config.variance_threshold)

        # Only apply to numeric features
        numeric_features = X.select_dtypes(include=[np.number])
        categorical_features = X.select_dtypes(exclude=[np.number])

        if len(numeric_features.columns) > 0:
            selected_mask = selector.fit_transform(numeric_features)
            selected_numeric = numeric_features.loc[:, selector.get_support()]

            # Combine with categorical features
            X_filtered = pd.concat([selected_numeric, categorical_features], axis=1)

            # Track removed features
            removed_features = {}
            removed_cols = numeric_features.columns[~selector.get_support()]
            for col in removed_cols:
                removed_features[col] = (
                    f"Low variance (< {self.config.variance_threshold})"
                )

            logger.info(f"Removed {len(removed_cols)} low variance features")

            return X_filtered, removed_features
        else:
            return X, {}

    def _combine_selection_results(
        self,
        all_features: List[str],
        statistical_features: List[str],
        rfe_features: List[str],
        model_features: List[str],
        feature_scores: Dict[str, FeatureImportanceScore],
    ) -> List[str]:
        """Combine results from different selection methods."""
        # Create voting system
        feature_votes = {feature: 0 for feature in all_features}

        # Vote from statistical selection
        for feature in statistical_features:
            if feature in feature_votes:
                feature_votes[feature] += 1

        # Vote from RFE
        for feature in rfe_features:
            if feature in feature_votes:
                feature_votes[feature] += 1

        # Vote from model-based selection
        for feature in model_features:
            if feature in feature_votes:
                feature_votes[feature] += 1

        # Select features with at least 2 votes or top features by combined score
        selected_by_voting = [
            feature for feature, votes in feature_votes.items() if votes >= 2
        ]

        # If we don't have enough features, add top features by importance
        if len(selected_by_voting) < self.config.top_k_features:
            # Calculate temporary combined scores for ranking
            temp_scores = []
            for feature in all_features:
                if feature in feature_scores:
                    score_obj = feature_scores[feature]
                    combined = (
                        score_obj.mutual_info_score
                        + score_obj.correlation_score
                        + score_obj.model_importance_score
                    ) / 3
                    temp_scores.append((feature, combined))

            # Sort by combined score
            temp_scores.sort(key=lambda x: x[1], reverse=True)

            # Add top features until we reach target
            for feature, _ in temp_scores:
                if (
                    feature not in selected_by_voting
                    and len(selected_by_voting) < self.config.top_k_features
                ):
                    selected_by_voting.append(feature)

        return selected_by_voting[: self.config.top_k_features]

    def _calculate_combined_scores(
        self,
        feature_scores: Dict[str, FeatureImportanceScore],
        selected_features: List[str],
    ) -> List[FeatureImportanceScore]:
        """Calculate combined scores and create final ranking."""
        # Normalize scores to 0-1 range
        all_mutual_info = [score.mutual_info_score for score in feature_scores.values()]
        all_correlation = [score.correlation_score for score in feature_scores.values()]
        all_model_importance = [
            score.model_importance_score for score in feature_scores.values()
        ]

        max_mutual_info = max(all_mutual_info) if max(all_mutual_info) > 0 else 1
        max_correlation = max(all_correlation) if max(all_correlation) > 0 else 1
        max_model_importance = (
            max(all_model_importance) if max(all_model_importance) > 0 else 1
        )

        # Calculate combined scores
        for feature_name, score_obj in feature_scores.items():
            normalized_mutual_info = score_obj.mutual_info_score / max_mutual_info
            normalized_correlation = score_obj.correlation_score / max_correlation
            normalized_model_importance = (
                score_obj.model_importance_score / max_model_importance
            )

            # Weighted combination (can be adjusted)
            combined_score = (
                0.4 * normalized_mutual_info
                + 0.3 * normalized_correlation
                + 0.3 * normalized_model_importance
            )

            score_obj.combined_score = combined_score
            score_obj.selected = feature_name in selected_features

        # Sort by combined score
        feature_scores_list = list(feature_scores.values())
        feature_scores_list.sort(key=lambda x: x.combined_score, reverse=True)

        return feature_scores_list


# Factory functions and utilities
def create_feature_selection_pipeline(
    config: Optional[FeatureSelectionConfig] = None,
) -> FeatureSelectionPipeline:
    """Create a feature selection pipeline instance."""
    return FeatureSelectionPipeline(config)


def select_banking_features(
    X: pd.DataFrame, y: pd.Series, config: Optional[FeatureSelectionConfig] = None
) -> FeatureSelectionResult:
    """Convenience function to select banking features."""
    pipeline = create_feature_selection_pipeline(config)
    return pipeline.process(X, y)


def get_default_selection_config() -> FeatureSelectionConfig:
    """Get default feature selection configuration."""
    return FeatureSelectionConfig()


def get_fast_selection_config() -> FeatureSelectionConfig:
    """Get fast feature selection configuration for testing."""
    return FeatureSelectionConfig(
        statistical_k_features=15, enable_rfe=False, top_k_features=20  # RFE is slow
    )
