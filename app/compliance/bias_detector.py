"""
Bias Detection System for Credit Risk AI.

This module implements comprehensive bias detection and fairness monitoring
for credit risk models, including demographic parity, equal opportunity,
and other fairness metrics across protected attributes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import warnings
import json

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Install with: pip install scipy")

try:
    from sklearn.metrics import (
        confusion_matrix,
        roc_auc_score,
        precision_score,
        recall_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Install with: pip install scikit-learn")

try:
    from ..core.logging import get_logger, get_audit_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from core.logging import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class FairnessMetric(Enum):
    """Types of fairness metrics."""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"
    TREATMENT_EQUALITY = "treatment_equality"
    CONDITIONAL_USE_ACCURACY_EQUALITY = "conditional_use_accuracy_equality"


class ProtectedAttribute(Enum):
    """Protected attributes for bias detection."""

    RACE = "race"
    GENDER = "gender"
    AGE = "age"
    ETHNICITY = "ethnicity"
    RELIGION = "religion"
    MARITAL_STATUS = "marital_status"
    DISABILITY = "disability"
    SEXUAL_ORIENTATION = "sexual_orientation"


class BiasLevel(Enum):
    """Bias severity levels."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


@dataclass
class FairnessThreshold:
    """Fairness threshold configuration."""

    metric: FairnessMetric
    threshold: float
    tolerance: float = 0.05  # 5% tolerance
    description: str = ""


@dataclass
class BiasDetectionResult:
    """Result of bias detection analysis."""

    metric: FairnessMetric
    protected_attribute: ProtectedAttribute
    groups: List[str]
    metric_values: Dict[str, float]
    overall_metric: float
    bias_detected: bool
    bias_level: BiasLevel
    threshold: float
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtectedGroupStats:
    """Statistics for a protected group."""

    group_name: str
    group_size: int
    positive_rate: float
    negative_rate: float
    true_positive_rate: float
    false_positive_rate: float
    true_negative_rate: float
    false_negative_rate: float
    precision: float
    recall: float
    f1_score: float
    auc_score: Optional[float] = None


class FairnessMetricsCalculator:
    """Calculates various fairness metrics."""

    def __init__(self):
        self.metric_functions = {
            FairnessMetric.DEMOGRAPHIC_PARITY: self._demographic_parity,
            FairnessMetric.EQUAL_OPPORTUNITY: self._equal_opportunity,
            FairnessMetric.EQUALIZED_ODDS: self._equalized_odds,
            FairnessMetric.CALIBRATION: self._calibration,
            FairnessMetric.TREATMENT_EQUALITY: self._treatment_equality,
            FairnessMetric.CONDITIONAL_USE_ACCURACY_EQUALITY: self._conditional_use_accuracy_equality,
        }

    def calculate_metric(
        self,
        metric: FairnessMetric,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attr: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate a specific fairness metric."""

        if metric not in self.metric_functions:
            raise ValueError(f"Unsupported metric: {metric}")

        return self.metric_functions[metric](y_true, y_pred, y_prob, protected_attr)

    def _demographic_parity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attr: np.ndarray,
    ) -> Dict[str, float]:
        """
        Demographic Parity: P(Y_hat = 1 | A = a) should be equal across groups.
        Measures if positive prediction rates are similar across groups.
        """
        results = {}
        unique_groups = np.unique(protected_attr)

        for group in unique_groups:
            group_mask = protected_attr == group
            group_predictions = y_pred[group_mask]
            positive_rate = np.mean(group_predictions)
            results[str(group)] = positive_rate

        # Calculate overall disparity
        rates = list(results.values())
        results["disparity"] = max(rates) - min(rates)
        results["ratio"] = min(rates) / max(rates) if max(rates) > 0 else 0

        return results

    def _equal_opportunity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attr: np.ndarray,
    ) -> Dict[str, float]:
        """
        Equal Opportunity: P(Y_hat = 1 | Y = 1, A = a) should be equal across groups.
        Measures if true positive rates are similar across groups.
        """
        results = {}
        unique_groups = np.unique(protected_attr)

        for group in unique_groups:
            group_mask = protected_attr == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]

            # True positive rate for positive cases
            positive_cases = group_true == 1
            if np.sum(positive_cases) > 0:
                tpr = np.mean(group_pred[positive_cases])
                results[str(group)] = tpr
            else:
                results[str(group)] = 0.0

        # Calculate disparity
        rates = [v for v in results.values() if v is not None]
        if rates:
            results["disparity"] = max(rates) - min(rates)
            results["ratio"] = min(rates) / max(rates) if max(rates) > 0 else 0

        return results

    def _equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attr: np.ndarray,
    ) -> Dict[str, float]:
        """
        Equalized Odds: Both TPR and FPR should be equal across groups.
        Combines equal opportunity with equal false positive rates.
        """
        results = {}
        unique_groups = np.unique(protected_attr)

        tpr_results = {}
        fpr_results = {}

        for group in unique_groups:
            group_mask = protected_attr == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]

            # True positive rate
            positive_cases = group_true == 1
            if np.sum(positive_cases) > 0:
                tpr = np.mean(group_pred[positive_cases])
                tpr_results[str(group)] = tpr

            # False positive rate
            negative_cases = group_true == 0
            if np.sum(negative_cases) > 0:
                fpr = np.mean(group_pred[negative_cases])
                fpr_results[str(group)] = fpr

        # Calculate disparities
        if tpr_results:
            tpr_rates = list(tpr_results.values())
            results["tpr_disparity"] = max(tpr_rates) - min(tpr_rates)
            results["tpr_ratio"] = (
                min(tpr_rates) / max(tpr_rates) if max(tpr_rates) > 0 else 0
            )

        if fpr_results:
            fpr_rates = list(fpr_results.values())
            results["fpr_disparity"] = max(fpr_rates) - min(fpr_rates)
            results["fpr_ratio"] = (
                min(fpr_rates) / max(fpr_rates) if max(fpr_rates) > 0 else 0
            )

        # Overall equalized odds violation
        results["overall_violation"] = results.get("tpr_disparity", 0) + results.get(
            "fpr_disparity", 0
        )

        return results

    def _calibration(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attr: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calibration: P(Y = 1 | Y_hat = p, A = a) should equal p across groups.
        Measures if predicted probabilities match actual outcomes across groups.
        """
        if y_prob is None:
            return {"error": "Probabilities required for calibration metric"}

        results = {}
        unique_groups = np.unique(protected_attr)

        # Create probability bins
        bins = np.linspace(0, 1, 11)  # 10 bins

        for group in unique_groups:
            group_mask = protected_attr == group
            group_true = y_true[group_mask]
            group_prob = y_prob[group_mask]

            calibration_error = 0
            bin_count = 0

            for i in range(len(bins) - 1):
                bin_mask = (group_prob >= bins[i]) & (group_prob < bins[i + 1])
                if np.sum(bin_mask) > 0:
                    bin_accuracy = np.mean(group_true[bin_mask])
                    bin_confidence = np.mean(group_prob[bin_mask])
                    calibration_error += abs(bin_accuracy - bin_confidence) * np.sum(
                        bin_mask
                    )
                    bin_count += np.sum(bin_mask)

            if bin_count > 0:
                results[str(group)] = calibration_error / bin_count
            else:
                results[str(group)] = 0.0

        # Calculate disparity
        errors = list(results.values())
        if errors:
            results["max_calibration_error"] = max(errors)
            results["calibration_disparity"] = max(errors) - min(errors)

        return results

    def _treatment_equality(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attr: np.ndarray,
    ) -> Dict[str, float]:
        """
        Treatment Equality: FN/FP ratio should be equal across groups.
        Measures if the ratio of false negatives to false positives is similar.
        """
        results = {}
        unique_groups = np.unique(protected_attr)

        for group in unique_groups:
            group_mask = protected_attr == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]

            # Calculate confusion matrix components
            tn = np.sum((group_true == 0) & (group_pred == 0))
            fp = np.sum((group_true == 0) & (group_pred == 1))
            fn = np.sum((group_true == 1) & (group_pred == 0))
            tp = np.sum((group_true == 1) & (group_pred == 1))

            # Treatment equality ratio
            if fp > 0:
                ratio = fn / fp
                results[str(group)] = ratio
            else:
                results[str(group)] = float("inf") if fn > 0 else 0.0

        # Calculate disparity (excluding infinite values)
        finite_ratios = [v for v in results.values() if np.isfinite(v)]
        if finite_ratios:
            results["disparity"] = max(finite_ratios) - min(finite_ratios)

        return results

    def _conditional_use_accuracy_equality(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attr: np.ndarray,
    ) -> Dict[str, float]:
        """
        Conditional Use Accuracy Equality: PPV and NPV should be equal across groups.
        Measures if positive and negative predictive values are similar.
        """
        results = {}
        unique_groups = np.unique(protected_attr)

        ppv_results = {}
        npv_results = {}

        for group in unique_groups:
            group_mask = protected_attr == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]

            # Calculate confusion matrix components
            tn = np.sum((group_true == 0) & (group_pred == 0))
            fp = np.sum((group_true == 0) & (group_pred == 1))
            fn = np.sum((group_true == 1) & (group_pred == 0))
            tp = np.sum((group_true == 1) & (group_pred == 1))

            # Positive Predictive Value (Precision)
            if tp + fp > 0:
                ppv = tp / (tp + fp)
                ppv_results[str(group)] = ppv

            # Negative Predictive Value
            if tn + fn > 0:
                npv = tn / (tn + fn)
                npv_results[str(group)] = npv

        # Calculate disparities
        if ppv_results:
            ppv_values = list(ppv_results.values())
            results["ppv_disparity"] = max(ppv_values) - min(ppv_values)
            results["ppv_ratio"] = (
                min(ppv_values) / max(ppv_values) if max(ppv_values) > 0 else 0
            )

        if npv_results:
            npv_values = list(npv_results.values())
            results["npv_disparity"] = max(npv_values) - min(npv_values)
            results["npv_ratio"] = (
                min(npv_values) / max(npv_values) if max(npv_values) > 0 else 0
            )

        return results


class ProtectedAttributeAnalyzer:
    """Analyzes protected attributes and group statistics."""

    def __init__(self):
        self.group_stats_cache = {}

    def analyze_protected_groups(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attr: np.ndarray,
        attribute_name: str,
    ) -> Dict[str, ProtectedGroupStats]:
        """Analyze statistics for each protected group."""

        results = {}
        unique_groups = np.unique(protected_attr)

        for group in unique_groups:
            group_mask = protected_attr == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]
            group_prob = y_prob[group_mask] if y_prob is not None else None

            stats = self._calculate_group_stats(
                str(group), group_true, group_pred, group_prob
            )
            results[str(group)] = stats

        return results

    def _calculate_group_stats(
        self,
        group_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
    ) -> ProtectedGroupStats:
        """Calculate comprehensive statistics for a group."""

        # Basic rates
        group_size = len(y_true)
        positive_rate = np.mean(y_pred)
        negative_rate = 1 - positive_rate

        # Confusion matrix components
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))

        # Performance metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # AUC score if probabilities available
        auc_score = None
        if y_prob is not None and len(np.unique(y_true)) > 1 and SKLEARN_AVAILABLE:
            try:
                auc_score = roc_auc_score(y_true, y_prob)
            except Exception:
                auc_score = None

        return ProtectedGroupStats(
            group_name=group_name,
            group_size=group_size,
            positive_rate=positive_rate,
            negative_rate=negative_rate,
            true_positive_rate=tpr,
            false_positive_rate=fpr,
            true_negative_rate=tnr,
            false_negative_rate=fnr,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc_score,
        )

    def detect_representation_bias(
        self, protected_attr: np.ndarray, attribute_name: str
    ) -> Dict[str, Any]:
        """Detect representation bias in the dataset."""

        unique_groups, counts = np.unique(protected_attr, return_counts=True)
        total_samples = len(protected_attr)

        representation = {}
        for group, count in zip(unique_groups, counts):
            representation[str(group)] = {
                "count": int(count),
                "percentage": float(count / total_samples * 100),
            }

        # Calculate representation disparity
        percentages = [info["percentage"] for info in representation.values()]
        max_percentage = max(percentages)
        min_percentage = min(percentages)

        disparity_ratio = min_percentage / max_percentage if max_percentage > 0 else 0

        # Determine if there's significant underrepresentation
        underrepresented_threshold = (
            10.0  # Less than 10% is considered underrepresented
        )
        underrepresented_groups = [
            group
            for group, info in representation.items()
            if info["percentage"] < underrepresented_threshold
        ]

        return {
            "representation": representation,
            "disparity_ratio": disparity_ratio,
            "max_percentage": max_percentage,
            "min_percentage": min_percentage,
            "underrepresented_groups": underrepresented_groups,
            "representation_bias_detected": disparity_ratio
            < 0.5,  # Significant if ratio < 0.5
        }


class BiasDetector:
    """Main bias detection system."""

    def __init__(self, fairness_thresholds: Optional[List[FairnessThreshold]] = None):
        self.metrics_calculator = FairnessMetricsCalculator()
        self.attribute_analyzer = ProtectedAttributeAnalyzer()

        # Default fairness thresholds
        self.fairness_thresholds = fairness_thresholds or self._get_default_thresholds()

        # Detection history
        self.detection_history: List[BiasDetectionResult] = []

        logger.info("Bias detector initialized")

    def _get_default_thresholds(self) -> List[FairnessThreshold]:
        """Get default fairness thresholds based on industry standards."""
        return [
            FairnessThreshold(
                FairnessMetric.DEMOGRAPHIC_PARITY,
                0.1,
                0.02,
                "Demographic parity violation: >10% difference in positive rates",
            ),
            FairnessThreshold(
                FairnessMetric.EQUAL_OPPORTUNITY,
                0.1,
                0.02,
                "Equal opportunity violation: >10% difference in TPR",
            ),
            FairnessThreshold(
                FairnessMetric.EQUALIZED_ODDS,
                0.15,
                0.03,
                "Equalized odds violation: >15% combined TPR/FPR difference",
            ),
            FairnessThreshold(
                FairnessMetric.CALIBRATION,
                0.05,
                0.01,
                "Calibration violation: >5% calibration error difference",
            ),
            FairnessThreshold(
                FairnessMetric.TREATMENT_EQUALITY,
                0.2,
                0.05,
                "Treatment equality violation: >20% difference in FN/FP ratio",
            ),
        ]

    def detect_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attributes: Dict[str, np.ndarray],
        y_prob: Optional[np.ndarray] = None,
        metrics: Optional[List[FairnessMetric]] = None,
    ) -> List[BiasDetectionResult]:
        """
        Comprehensive bias detection across multiple protected attributes and metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attributes: Dict mapping attribute names to attribute values
            y_prob: Predicted probabilities (optional)
            metrics: List of metrics to calculate (optional, defaults to all)

        Returns:
            List of bias detection results
        """

        if metrics is None:
            metrics = [threshold.metric for threshold in self.fairness_thresholds]

        results = []

        for attr_name, attr_values in protected_attributes.items():
            try:
                protected_attr_enum = ProtectedAttribute(attr_name.lower())
            except ValueError:
                logger.warning(f"Unknown protected attribute: {attr_name}")
                continue

            for metric in metrics:
                try:
                    result = self._detect_bias_for_attribute_metric(
                        y_true, y_pred, y_prob, attr_values, protected_attr_enum, metric
                    )
                    results.append(result)

                except Exception as e:
                    logger.error(
                        f"Error detecting bias for {attr_name} - {metric.value}: {e}"
                    )

        # Store results in history
        self.detection_history.extend(results)

        # Log significant bias detections
        significant_bias = [
            r
            for r in results
            if r.bias_detected and r.bias_level in [BiasLevel.HIGH, BiasLevel.SEVERE]
        ]
        if significant_bias:
            self._log_bias_violations(significant_bias)

        return results

    def _detect_bias_for_attribute_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attr: np.ndarray,
        attr_enum: ProtectedAttribute,
        metric: FairnessMetric,
    ) -> BiasDetectionResult:
        """Detect bias for a specific attribute and metric combination."""

        # Calculate metric values
        metric_values = self.metrics_calculator.calculate_metric(
            metric, y_true, y_pred, y_prob, protected_attr
        )

        # Get threshold for this metric
        threshold_config = next(
            (t for t in self.fairness_thresholds if t.metric == metric),
            FairnessThreshold(metric, 0.1, 0.02),  # Default threshold
        )

        # Determine bias level and detection
        bias_detected, bias_level, overall_metric = self._evaluate_bias_level(
            metric_values, threshold_config
        )

        # Statistical significance test
        p_value = self._calculate_statistical_significance(
            y_true, y_pred, protected_attr, metric
        )

        # Get unique groups
        unique_groups = [str(g) for g in np.unique(protected_attr)]

        return BiasDetectionResult(
            metric=metric,
            protected_attribute=attr_enum,
            groups=unique_groups,
            metric_values=metric_values,
            overall_metric=overall_metric,
            bias_detected=bias_detected,
            bias_level=bias_level,
            threshold=threshold_config.threshold,
            p_value=p_value,
            timestamp=datetime.now(),
            details={
                "threshold_config": {
                    "threshold": threshold_config.threshold,
                    "tolerance": threshold_config.tolerance,
                    "description": threshold_config.description,
                },
                "group_sizes": {
                    str(g): int(np.sum(protected_attr == g))
                    for g in np.unique(protected_attr)
                },
            },
        )

    def _evaluate_bias_level(
        self, metric_values: Dict[str, float], threshold_config: FairnessThreshold
    ) -> Tuple[bool, BiasLevel, float]:
        """Evaluate bias level based on metric values and thresholds."""

        # Extract the main disparity metric
        disparity_keys = [
            "disparity",
            "overall_violation",
            "max_calibration_error",
            "calibration_disparity",
        ]
        overall_metric = 0.0

        for key in disparity_keys:
            if key in metric_values:
                overall_metric = metric_values[key]
                break

        # If no disparity metric found, calculate from ratio
        if overall_metric == 0.0 and "ratio" in metric_values:
            ratio = metric_values["ratio"]
            overall_metric = 1.0 - ratio if ratio <= 1.0 else ratio - 1.0

        # Determine bias level
        threshold = threshold_config.threshold
        tolerance = threshold_config.tolerance

        if overall_metric <= tolerance:
            bias_level = BiasLevel.NONE
            bias_detected = False
        elif overall_metric <= threshold:
            bias_level = BiasLevel.LOW
            bias_detected = True
        elif overall_metric <= threshold * 2:
            bias_level = BiasLevel.MODERATE
            bias_detected = True
        elif overall_metric <= threshold * 3:
            bias_level = BiasLevel.HIGH
            bias_detected = True
        else:
            bias_level = BiasLevel.SEVERE
            bias_detected = True

        return bias_detected, bias_level, overall_metric

    def _calculate_statistical_significance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
        metric: FairnessMetric,
    ) -> Optional[float]:
        """Calculate statistical significance of bias detection."""

        if not SCIPY_AVAILABLE:
            return None

        try:
            unique_groups = np.unique(protected_attr)
            if len(unique_groups) != 2:
                return None  # Chi-square test works best for binary comparisons

            # Create contingency table
            group1_mask = protected_attr == unique_groups[0]
            group2_mask = protected_attr == unique_groups[1]

            # For demographic parity, test difference in positive rates
            if metric == FairnessMetric.DEMOGRAPHIC_PARITY:
                group1_pos = np.sum(y_pred[group1_mask])
                group1_total = np.sum(group1_mask)
                group2_pos = np.sum(y_pred[group2_mask])
                group2_total = np.sum(group2_mask)

                contingency_table = np.array(
                    [
                        [group1_pos, group1_total - group1_pos],
                        [group2_pos, group2_total - group2_pos],
                    ]
                )

                chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                return p_value

            # For other metrics, use permutation test
            return self._permutation_test(y_true, y_pred, protected_attr, metric)

        except Exception as e:
            logger.warning(f"Could not calculate statistical significance: {e}")
            return None

    def _permutation_test(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
        metric: FairnessMetric,
        n_permutations: int = 1000,
    ) -> float:
        """Perform permutation test for statistical significance."""

        # Calculate observed metric
        observed_values = self.metrics_calculator.calculate_metric(
            metric, y_true, y_pred, None, protected_attr
        )
        observed_disparity = observed_values.get("disparity", 0)

        # Permutation test
        permuted_disparities = []
        for _ in range(n_permutations):
            # Randomly permute protected attributes
            permuted_attr = np.random.permutation(protected_attr)

            # Calculate metric for permuted data
            permuted_values = self.metrics_calculator.calculate_metric(
                metric, y_true, y_pred, None, permuted_attr
            )
            permuted_disparity = permuted_values.get("disparity", 0)
            permuted_disparities.append(permuted_disparity)

        # Calculate p-value
        p_value = np.mean(np.array(permuted_disparities) >= observed_disparity)
        return p_value

    def _log_bias_violations(self, violations: List[BiasDetectionResult]):
        """Log significant bias violations."""

        for violation in violations:
            message = (
                f"Bias violation detected: {violation.metric.value} "
                f"for {violation.protected_attribute.value} "
                f"(level: {violation.bias_level.value}, "
                f"metric: {violation.overall_metric:.3f}, "
                f"threshold: {violation.threshold:.3f})"
            )

            logger.warning(message)

            # Audit log for compliance
            audit_logger.log_model_operation(
                user_id="bias_detector",
                model_id="fairness_monitor",
                operation="bias_detection",
                success=False,
                details={
                    "violation_type": violation.metric.value,
                    "protected_attribute": violation.protected_attribute.value,
                    "bias_level": violation.bias_level.value,
                    "metric_value": violation.overall_metric,
                    "threshold": violation.threshold,
                    "groups": violation.groups,
                    "p_value": violation.p_value,
                },
            )

    def generate_bias_report(
        self, results: Optional[List[BiasDetectionResult]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive bias detection report."""

        if results is None:
            results = self.detection_history

        if not results:
            return {"error": "No bias detection results available"}

        # Aggregate results by attribute and metric
        by_attribute = defaultdict(list)
        by_metric = defaultdict(list)

        for result in results:
            by_attribute[result.protected_attribute.value].append(result)
            by_metric[result.metric.value].append(result)

        # Summary statistics
        total_tests = len(results)
        violations = [r for r in results if r.bias_detected]
        violation_rate = len(violations) / total_tests if total_tests > 0 else 0

        # Bias level distribution
        bias_levels = defaultdict(int)
        for result in results:
            bias_levels[result.bias_level.value] += 1

        # Most problematic attributes and metrics
        attribute_violation_rates = {}
        for attr, attr_results in by_attribute.items():
            attr_violations = [r for r in attr_results if r.bias_detected]
            attribute_violation_rates[attr] = len(attr_violations) / len(attr_results)

        metric_violation_rates = {}
        for metric, metric_results in by_metric.items():
            metric_violations = [r for r in metric_results if r.bias_detected]
            metric_violation_rates[metric] = len(metric_violations) / len(
                metric_results
            )

        return {
            "summary": {
                "total_tests": total_tests,
                "violations_detected": len(violations),
                "violation_rate": violation_rate,
                "bias_level_distribution": dict(bias_levels),
            },
            "by_protected_attribute": {
                attr: {
                    "tests_conducted": len(attr_results),
                    "violations": len([r for r in attr_results if r.bias_detected]),
                    "violation_rate": attribute_violation_rates[attr],
                    "worst_violation": max(
                        attr_results, key=lambda x: x.overall_metric
                    ).bias_level.value,
                }
                for attr, attr_results in by_attribute.items()
            },
            "by_fairness_metric": {
                metric: {
                    "tests_conducted": len(metric_results),
                    "violations": len([r for r in metric_results if r.bias_detected]),
                    "violation_rate": metric_violation_rates[metric],
                    "average_disparity": np.mean(
                        [r.overall_metric for r in metric_results]
                    ),
                }
                for metric, metric_results in by_metric.items()
            },
            "recommendations": self._generate_recommendations(results),
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_recommendations(
        self, results: List[BiasDetectionResult]
    ) -> List[str]:
        """Generate recommendations based on bias detection results."""

        recommendations = []

        # Check for severe violations
        severe_violations = [r for r in results if r.bias_level == BiasLevel.SEVERE]
        if severe_violations:
            recommendations.append(
                "URGENT: Severe bias detected. Consider immediate model retraining with bias mitigation techniques."
            )

        # Check for high violations
        high_violations = [r for r in results if r.bias_level == BiasLevel.HIGH]
        if high_violations:
            recommendations.append(
                "High bias levels detected. Implement bias mitigation strategies before deployment."
            )

        # Check for demographic parity issues
        dp_violations = [
            r
            for r in results
            if r.metric == FairnessMetric.DEMOGRAPHIC_PARITY and r.bias_detected
        ]
        if dp_violations:
            recommendations.append(
                "Demographic parity violations found. Consider reweighting training data or post-processing adjustments."
            )

        # Check for equal opportunity issues
        eo_violations = [
            r
            for r in results
            if r.metric == FairnessMetric.EQUAL_OPPORTUNITY and r.bias_detected
        ]
        if eo_violations:
            recommendations.append(
                "Equal opportunity violations detected. Focus on improving recall for underrepresented groups."
            )

        # Check for representation issues
        underrepresented = []
        for result in results:
            small_groups = [
                g
                for g, size in result.details.get("group_sizes", {}).items()
                if size < 100
            ]
            underrepresented.extend(small_groups)

        if underrepresented:
            recommendations.append(
                f"Small group sizes detected: {set(underrepresented)}. Consider data augmentation or stratified sampling."
            )

        if not recommendations:
            recommendations.append(
                "No significant bias detected. Continue monitoring fairness metrics."
            )

        return recommendations

    def get_detection_history(self, days: int = 30) -> List[BiasDetectionResult]:
        """Get bias detection history for the specified number of days."""

        cutoff_date = datetime.now() - timedelta(days=days)
        return [r for r in self.detection_history if r.timestamp >= cutoff_date]


# Utility functions


def create_bias_detector(
    custom_thresholds: Optional[List[FairnessThreshold]] = None,
) -> BiasDetector:
    """Create bias detector with optional custom thresholds."""
    return BiasDetector(custom_thresholds)


def analyze_dataset_bias(
    data: pd.DataFrame, target_column: str, protected_columns: List[str]
) -> Dict[str, Any]:
    """Analyze bias in a dataset before model training."""

    analyzer = ProtectedAttributeAnalyzer()
    results = {}

    for col in protected_columns:
        if col in data.columns:
            representation_analysis = analyzer.detect_representation_bias(
                data[col].values, col
            )
            results[col] = representation_analysis

    return results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate synthetic data
    n_samples = 1000

    # Protected attributes
    gender = np.random.choice(["male", "female"], n_samples, p=[0.6, 0.4])
    race = np.random.choice(
        ["white", "black", "hispanic", "asian"], n_samples, p=[0.5, 0.2, 0.2, 0.1]
    )

    # Simulate biased predictions (higher approval rate for certain groups)
    y_true = np.random.binomial(1, 0.3, n_samples)

    # Biased predictions
    bias_factor = np.where(gender == "male", 1.2, 0.8)
    y_prob = np.clip(np.random.beta(2, 5, n_samples) * bias_factor, 0, 1)
    y_pred = (y_prob > 0.5).astype(int)

    # Create bias detector
    detector = create_bias_detector()

    # Detect bias
    protected_attributes = {"gender": gender, "race": race}

    results = detector.detect_bias(y_true, y_pred, protected_attributes, y_prob)

    # Generate report
    report = detector.generate_bias_report(results)

    print("Bias Detection Report:")
    print(json.dumps(report, indent=2, default=str))
