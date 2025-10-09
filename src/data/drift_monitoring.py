"""
Data drift and quality monitoring system for credit risk models.
Implements statistical tests, concept drift detection, and automated alerts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
from pathlib import Path

# Statistical tests
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, mannwhitneyu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json

from ..core.interfaces import DataProcessor
from ..core.config import get_config
from ..core.logging import get_logger, get_audit_logger


logger = get_logger(__name__)
audit_logger = get_audit_logger()


class DriftType(Enum):
    """Types of data drift."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_DRIFT = "feature_drift"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection."""
    # Statistical test parameters
    significance_level: float = 0.05
    ks_test_threshold: float = 0.05
    psi_threshold: float = 0.1  # Population Stability Index
    chi2_threshold: float = 0.05
    
    # Concept drift parameters
    concept_drift_window_size: int = 1000
    concept_drift_threshold: float = 0.05
    performance_degradation_threshold: float = 0.1
    
    # Feature drift parameters
    feature_drift_threshold: float = 0.1
    correlation_drift_threshold: float = 0.1
    
    # Quality monitoring parameters
    missing_data_threshold: float = 0.1
    duplicate_threshold: float = 0.05
    outlier_threshold: float = 0.1
    
    # Alert configuration
    enable_alerts: bool = True
    alert_cooldown_hours: int = 24
    
    # Monitoring frequency
    monitoring_enabled: bool = True
    check_interval_hours: int = 6


@dataclass
class DriftAlert:
    """Data drift alert."""
    alert_id: str
    drift_type: DriftType
    severity: AlertSeverity
    feature_name: Optional[str]
    drift_score: float
    threshold: float
    message: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    timestamp: datetime
    data_drift_detected: bool
    concept_drift_detected: bool
    feature_drift_detected: bool
    alerts: List[DriftAlert]
    drift_scores: Dict[str, float]
    quality_metrics: Dict[str, float]
    recommendations: List[str]
    processing_time_seconds: float


@dataclass
class QualityMetrics:
    """Data quality metrics."""
    missing_data_percentage: float
    duplicate_percentage: float
    outlier_percentage: float
    schema_violations: int
    data_freshness_hours: float
    completeness_score: float
    consistency_score: float
    validity_score: float
    overall_quality_score: float


class StatisticalDriftDetector:
    """Detects statistical drift using various tests."""
    
    def __init__(self, config: DriftDetectionConfig):
        self.config = config
    
    def detect_numerical_drift(self, reference_data: pd.Series, 
                             current_data: pd.Series) -> Tuple[bool, float, str]:
        """Detect drift in numerical features using KS test."""
        try:
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = ks_2samp(reference_data.dropna(), current_data.dropna())
            
            drift_detected = p_value < self.config.ks_test_threshold
            
            test_info = f"KS test: statistic={ks_statistic:.4f}, p-value={p_value:.4f}"
            
            return drift_detected, ks_statistic, test_info
            
        except Exception as e:
            logger.error(f"Numerical drift detection failed: {e}")
            return False, 0.0, f"Error: {str(e)}"
    
    def detect_categorical_drift(self, reference_data: pd.Series, 
                               current_data: pd.Series) -> Tuple[bool, float, str]:
        """Detect drift in categorical features using Chi-square test."""
        try:
            # Get value counts
            ref_counts = reference_data.value_counts()
            curr_counts = current_data.value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
            
            # Chi-square test
            if sum(ref_aligned) > 0 and sum(curr_aligned) > 0:
                chi2_stat, p_value, _, _ = chi2_contingency([ref_aligned, curr_aligned])
                
                drift_detected = p_value < self.config.chi2_threshold
                
                test_info = f"Chi-square test: statistic={chi2_stat:.4f}, p-value={p_value:.4f}"
                
                return drift_detected, chi2_stat, test_info
            else:
                return False, 0.0, "Insufficient data for Chi-square test"
                
        except Exception as e:
            logger.error(f"Categorical drift detection failed: {e}")
            return False, 0.0, f"Error: {str(e)}"
    
    def calculate_psi(self, reference_data: pd.Series, 
                     current_data: pd.Series, bins: int = 10) -> Tuple[bool, float]:
        """Calculate Population Stability Index (PSI)."""
        try:
            # Create bins based on reference data
            if pd.api.types.is_numeric_dtype(reference_data):
                # Numerical data
                bin_edges = np.histogram_bin_edges(reference_data.dropna(), bins=bins)
                
                ref_hist, _ = np.histogram(reference_data.dropna(), bins=bin_edges)
                curr_hist, _ = np.histogram(current_data.dropna(), bins=bin_edges)
            else:
                # Categorical data
                ref_counts = reference_data.value_counts()
                curr_counts = current_data.value_counts()
                
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                
                ref_hist = np.array([ref_counts.get(cat, 0) for cat in all_categories])
                curr_hist = np.array([curr_counts.get(cat, 0) for cat in all_categories])
            
            # Convert to proportions
            ref_prop = ref_hist / ref_hist.sum()
            curr_prop = curr_hist / curr_hist.sum()
            
            # Avoid division by zero
            ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
            curr_prop = np.where(curr_prop == 0, 0.0001, curr_prop)
            
            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
            
            drift_detected = psi > self.config.psi_threshold
            
            return drift_detected, psi
            
        except Exception as e:
            logger.error(f"PSI calculation failed: {e}")
            return False, 0.0


class ConceptDriftDetector:
    """Detects concept drift in model predictions."""
    
    def __init__(self, config: DriftDetectionConfig):
        self.config = config
        self.reference_performance = {}
        self.performance_history = []
    
    def detect_concept_drift(self, y_true: pd.Series, y_pred: pd.Series, 
                           y_pred_proba: Optional[pd.Series] = None) -> Tuple[bool, Dict[str, float], str]:
        """Detect concept drift using performance degradation."""
        try:
            # Calculate current performance metrics
            current_metrics = self._calculate_performance_metrics(y_true, y_pred, y_pred_proba)
            
            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'metrics': current_metrics
            })
            
            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(days=30)
            self.performance_history = [
                entry for entry in self.performance_history 
                if entry['timestamp'] > cutoff_time
            ]
            
            # Check for concept drift
            drift_detected = False
            drift_info = "No concept drift detected"
            
            if len(self.performance_history) > 1:
                # Compare with recent average
                recent_metrics = [entry['metrics'] for entry in self.performance_history[-10:]]
                avg_recent_accuracy = np.mean([m['accuracy'] for m in recent_metrics])
                
                # Compare with reference performance
                if self.reference_performance:
                    ref_accuracy = self.reference_performance.get('accuracy', avg_recent_accuracy)
                    
                    accuracy_drop = ref_accuracy - current_metrics['accuracy']
                    
                    if accuracy_drop > self.config.performance_degradation_threshold:
                        drift_detected = True
                        drift_info = f"Performance degradation detected: accuracy dropped by {accuracy_drop:.3f}"
                else:
                    # Set current as reference if no reference exists
                    self.reference_performance = current_metrics.copy()
            
            return drift_detected, current_metrics, drift_info
            
        except Exception as e:
            logger.error(f"Concept drift detection failed: {e}")
            return False, {}, f"Error: {str(e)}"
    
    def _calculate_performance_metrics(self, y_true: pd.Series, y_pred: pd.Series, 
                                     y_pred_proba: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            if y_pred_proba is not None:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            
        except Exception as e:
            logger.warning(f"Performance metrics calculation failed: {e}")
            metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        return metrics
    
    def set_reference_performance(self, metrics: Dict[str, float]):
        """Set reference performance metrics."""
        self.reference_performance = metrics.copy()
        logger.info(f"Reference performance set: {metrics}")


class DataQualityMonitor:
    """Monitors data quality metrics."""
    
    def __init__(self, config: DriftDetectionConfig):
        self.config = config
    
    def assess_data_quality(self, data: pd.DataFrame, 
                          schema: Optional[Dict[str, Any]] = None) -> QualityMetrics:
        """Assess overall data quality."""
        try:
            # Calculate quality metrics
            missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            duplicate_percentage = (data.duplicated().sum() / len(data)) * 100
            
            # Outlier detection (simple IQR method for numeric columns)
            outlier_percentage = self._calculate_outlier_percentage(data)
            
            # Schema violations
            schema_violations = self._check_schema_violations(data, schema) if schema else 0
            
            # Data freshness (assume data has a timestamp column)
            data_freshness_hours = self._calculate_data_freshness(data)
            
            # Calculate component scores
            completeness_score = max(0, 1 - missing_percentage / 100)
            consistency_score = max(0, 1 - duplicate_percentage / 100)
            validity_score = max(0, 1 - schema_violations / len(data.columns)) if len(data.columns) > 0 else 1.0
            
            # Overall quality score
            overall_quality_score = (completeness_score + consistency_score + validity_score) / 3
            
            return QualityMetrics(
                missing_data_percentage=missing_percentage,
                duplicate_percentage=duplicate_percentage,
                outlier_percentage=outlier_percentage,
                schema_violations=schema_violations,
                data_freshness_hours=data_freshness_hours,
                completeness_score=completeness_score,
                consistency_score=consistency_score,
                validity_score=validity_score,
                overall_quality_score=overall_quality_score
            )
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            return QualityMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_outlier_percentage(self, data: pd.DataFrame) -> float:
        """Calculate percentage of outliers using IQR method."""
        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return 0.0
            
            total_outliers = 0
            total_values = 0
            
            for column in numeric_columns:
                series = data[column].dropna()
                if len(series) > 0:
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
                    total_outliers += outliers
                    total_values += len(series)
            
            return (total_outliers / total_values) * 100 if total_values > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Outlier calculation failed: {e}")
            return 0.0
    
    def _check_schema_violations(self, data: pd.DataFrame, 
                               schema: Dict[str, Any]) -> int:
        """Check for schema violations."""
        violations = 0
        
        try:
            # Check required columns
            required_columns = schema.get('required_columns', [])
            missing_columns = set(required_columns) - set(data.columns)
            violations += len(missing_columns)
            
            # Check data types
            expected_types = schema.get('column_types', {})
            for column, expected_type in expected_types.items():
                if column in data.columns:
                    actual_type = str(data[column].dtype)
                    if expected_type not in actual_type:
                        violations += 1
            
            # Check value ranges
            value_ranges = schema.get('value_ranges', {})
            for column, (min_val, max_val) in value_ranges.items():
                if column in data.columns and pd.api.types.is_numeric_dtype(data[column]):
                    out_of_range = ((data[column] < min_val) | (data[column] > max_val)).sum()
                    if out_of_range > 0:
                        violations += 1
            
        except Exception as e:
            logger.warning(f"Schema validation failed: {e}")
        
        return violations
    
    def _calculate_data_freshness(self, data: pd.DataFrame) -> float:
        """Calculate data freshness in hours."""
        try:
            # Look for timestamp columns
            timestamp_columns = []
            for column in data.columns:
                if 'timestamp' in column.lower() or 'date' in column.lower() or 'time' in column.lower():
                    timestamp_columns.append(column)
            
            if timestamp_columns:
                # Use the first timestamp column
                timestamp_col = timestamp_columns[0]
                if pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
                    latest_timestamp = data[timestamp_col].max()
                    current_time = pd.Timestamp.now()
                    freshness_hours = (current_time - latest_timestamp).total_seconds() / 3600
                    return max(0, freshness_hours)
            
            return 0.0  # No timestamp information available
            
        except Exception as e:
            logger.warning(f"Data freshness calculation failed: {e}")
            return 0.0


class AlertManager:
    """Manages drift detection alerts."""
    
    def __init__(self, config: DriftDetectionConfig):
        self.config = config
        self.active_alerts: List[DriftAlert] = []
        self.alert_history: List[DriftAlert] = []
        self.last_alert_times: Dict[str, datetime] = {}
    
    def create_alert(self, drift_type: DriftType, severity: AlertSeverity,
                    feature_name: Optional[str], drift_score: float,
                    threshold: float, message: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[DriftAlert]:
        """Create a new drift alert."""
        if not self.config.enable_alerts:
            return None
        
        # Check cooldown period
        alert_key = f"{drift_type.value}_{feature_name or 'global'}"
        if alert_key in self.last_alert_times:
            time_since_last = datetime.now() - self.last_alert_times[alert_key]
            if time_since_last.total_seconds() < self.config.alert_cooldown_hours * 3600:
                logger.debug(f"Alert suppressed due to cooldown: {alert_key}")
                return None
        
        # Create alert
        alert = DriftAlert(
            alert_id=f"{drift_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            drift_type=drift_type,
            severity=severity,
            feature_name=feature_name,
            drift_score=drift_score,
            threshold=threshold,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        self.last_alert_times[alert_key] = datetime.now()
        
        logger.warning(f"Drift alert created: {alert.message}")
        
        # Log alert
        audit_logger.log_security_event(
            event_type="drift_alert",
            user_id="system",
            severity=severity.value.upper(),
            details={
                "drift_type": drift_type.value,
                "feature_name": feature_name,
                "drift_score": drift_score,
                "threshold": threshold,
                "message": message
            }
        )
        
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                self.active_alerts.remove(alert)
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[DriftAlert]:
        """Get active alerts, optionally filtered by severity."""
        if severity:
            return [alert for alert in self.active_alerts if alert.severity == severity]
        return self.active_alerts.copy()
    
    def cleanup_old_alerts(self, days: int = 30):
        """Clean up old alerts from history."""
        cutoff_date = datetime.now() - timedelta(days=days)
        self.alert_history = [
            alert for alert in self.alert_history 
            if alert.timestamp > cutoff_date
        ]


class DriftMonitor(DataProcessor):
    """Main drift monitoring system."""
    
    def __init__(self, config: Optional[DriftDetectionConfig] = None):
        self.config = config or DriftDetectionConfig()
        self.statistical_detector = StatisticalDriftDetector(self.config)
        self.concept_detector = ConceptDriftDetector(self.config)
        self.quality_monitor = DataQualityMonitor(self.config)
        self.alert_manager = AlertManager(self.config)
        
        # Reference data storage
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_statistics: Dict[str, Any] = {}
        self.monitoring_active = False
    
    def set_reference_data(self, reference_data: pd.DataFrame, 
                          target_column: Optional[str] = None):
        """Set reference data for drift detection."""
        self.reference_data = reference_data.copy()
        
        # Calculate reference statistics
        self.reference_statistics = self._calculate_reference_statistics(reference_data)
        
        # Set reference performance if target is available
        if target_column and target_column in reference_data.columns:
            # Train a simple model to establish baseline performance
            X = reference_data.drop(columns=[target_column])
            y = reference_data[target_column]
            
            # Encode categorical variables for model training
            X_encoded = self._encode_for_monitoring(X)
            
            if len(X_encoded.columns) > 0:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_encoded, y, test_size=0.2, random_state=42
                    )
                    
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None
                    
                    reference_metrics = self.concept_detector._calculate_performance_metrics(
                        y_test, y_pred, y_pred_proba
                    )
                    
                    self.concept_detector.set_reference_performance(reference_metrics)
                    
                except Exception as e:
                    logger.warning(f"Failed to set reference performance: {e}")
        
        self.monitoring_active = True
        logger.info(f"Reference data set with {len(reference_data)} samples")
    
    def process(self, current_data: pd.DataFrame, 
               target_column: Optional[str] = None,
               predictions: Optional[pd.Series] = None,
               prediction_probabilities: Optional[pd.Series] = None) -> DriftDetectionResult:
        """Process drift detection on current data."""
        start_time = datetime.now()
        
        if not self.monitoring_active or self.reference_data is None:
            return DriftDetectionResult(
                timestamp=datetime.now(),
                data_drift_detected=False,
                concept_drift_detected=False,
                feature_drift_detected=False,
                alerts=[],
                drift_scores={},
                quality_metrics={},
                recommendations=["Set reference data before monitoring"],
                processing_time_seconds=0.0
            )
        
        try:
            logger.info("Starting drift detection analysis")
            
            alerts = []
            drift_scores = {}
            recommendations = []
            
            # 1. Data drift detection
            data_drift_detected, data_drift_scores = self._detect_data_drift(current_data)
            drift_scores.update(data_drift_scores)
            
            # 2. Feature drift detection
            feature_drift_detected, feature_drift_scores = self._detect_feature_drift(current_data)
            drift_scores.update(feature_drift_scores)
            
            # 3. Concept drift detection
            concept_drift_detected = False
            if target_column and predictions is not None:
                if target_column in current_data.columns:
                    concept_drift_detected, concept_metrics, concept_info = self.concept_detector.detect_concept_drift(
                        current_data[target_column], predictions, prediction_probabilities
                    )
                    drift_scores['concept_drift'] = concept_metrics.get('accuracy', 0.0)
                    
                    if concept_drift_detected:
                        alert = self.alert_manager.create_alert(
                            DriftType.CONCEPT_DRIFT,
                            AlertSeverity.HIGH,
                            None,
                            1.0 - concept_metrics.get('accuracy', 0.0),
                            self.config.performance_degradation_threshold,
                            concept_info
                        )
                        if alert:
                            alerts.append(alert)
                        recommendations.append("Model retraining recommended due to concept drift")
            
            # 4. Data quality assessment
            quality_metrics_obj = self.quality_monitor.assess_data_quality(current_data)
            quality_metrics = {
                'missing_data_percentage': quality_metrics_obj.missing_data_percentage,
                'duplicate_percentage': quality_metrics_obj.duplicate_percentage,
                'outlier_percentage': quality_metrics_obj.outlier_percentage,
                'overall_quality_score': quality_metrics_obj.overall_quality_score
            }
            
            # Check quality thresholds and create alerts
            if quality_metrics_obj.missing_data_percentage > self.config.missing_data_threshold * 100:
                alert = self.alert_manager.create_alert(
                    DriftType.DATA_DRIFT,
                    AlertSeverity.MEDIUM,
                    "missing_data",
                    quality_metrics_obj.missing_data_percentage / 100,
                    self.config.missing_data_threshold,
                    f"High missing data percentage: {quality_metrics_obj.missing_data_percentage:.2f}%"
                )
                if alert:
                    alerts.append(alert)
                recommendations.append("Investigate data collection pipeline for missing data issues")
            
            if quality_metrics_obj.duplicate_percentage > self.config.duplicate_threshold * 100:
                alert = self.alert_manager.create_alert(
                    DriftType.DATA_DRIFT,
                    AlertSeverity.LOW,
                    "duplicates",
                    quality_metrics_obj.duplicate_percentage / 100,
                    self.config.duplicate_threshold,
                    f"High duplicate percentage: {quality_metrics_obj.duplicate_percentage:.2f}%"
                )
                if alert:
                    alerts.append(alert)
                recommendations.append("Review data deduplication processes")
            
            # 5. Generate recommendations
            if data_drift_detected or feature_drift_detected:
                recommendations.append("Consider retraining models with recent data")
                recommendations.append("Review feature engineering pipeline")
            
            if quality_metrics_obj.overall_quality_score < 0.7:
                recommendations.append("Improve data quality before model inference")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Log monitoring results
            audit_logger.log_data_access(
                user_id="system",
                resource="drift_monitor",
                action="drift_detection",
                success=True,
                details={
                    "data_drift_detected": data_drift_detected,
                    "concept_drift_detected": concept_drift_detected,
                    "feature_drift_detected": feature_drift_detected,
                    "alerts_generated": len(alerts),
                    "processing_time_seconds": processing_time
                }
            )
            
            logger.info(f"Drift detection completed: {len(alerts)} alerts generated")
            
            return DriftDetectionResult(
                timestamp=datetime.now(),
                data_drift_detected=data_drift_detected,
                concept_drift_detected=concept_drift_detected,
                feature_drift_detected=feature_drift_detected,
                alerts=alerts,
                drift_scores=drift_scores,
                quality_metrics=quality_metrics,
                recommendations=recommendations,
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Drift detection failed: {str(e)}"
            logger.error(error_message)
            
            return DriftDetectionResult(
                timestamp=datetime.now(),
                data_drift_detected=False,
                concept_drift_detected=False,
                feature_drift_detected=False,
                alerts=[],
                drift_scores={},
                quality_metrics={},
                recommendations=[f"Error in drift detection: {str(e)}"],
                processing_time_seconds=processing_time
            )
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate data for drift monitoring."""
        try:
            if data.empty:
                logger.error("Data is empty")
                return False
            
            if self.reference_data is not None:
                # Check if columns match reference data
                missing_columns = set(self.reference_data.columns) - set(data.columns)
                if missing_columns:
                    logger.warning(f"Missing columns compared to reference: {missing_columns}")
            
            return True
            
        except Exception as e:
            logger.error(f"Drift monitoring validation failed: {e}")
            return False
    
    def _detect_data_drift(self, current_data: pd.DataFrame) -> Tuple[bool, Dict[str, float]]:
        """Detect data drift across all features."""
        drift_detected = False
        drift_scores = {}
        
        common_columns = set(self.reference_data.columns) & set(current_data.columns)
        
        for column in common_columns:
            ref_series = self.reference_data[column]
            curr_series = current_data[column]
            
            if pd.api.types.is_numeric_dtype(ref_series):
                # Numerical drift detection
                column_drift, drift_score, test_info = self.statistical_detector.detect_numerical_drift(
                    ref_series, curr_series
                )
            else:
                # Categorical drift detection
                column_drift, drift_score, test_info = self.statistical_detector.detect_categorical_drift(
                    ref_series, curr_series
                )
            
            drift_scores[f"{column}_drift"] = drift_score
            
            if column_drift:
                drift_detected = True
                
                # Create alert
                severity = AlertSeverity.HIGH if drift_score > 0.5 else AlertSeverity.MEDIUM
                alert = self.alert_manager.create_alert(
                    DriftType.DATA_DRIFT,
                    severity,
                    column,
                    drift_score,
                    self.config.ks_test_threshold if pd.api.types.is_numeric_dtype(ref_series) else self.config.chi2_threshold,
                    f"Data drift detected in {column}: {test_info}",
                    {"test_info": test_info}
                )
            
            # Also calculate PSI
            psi_drift, psi_score = self.statistical_detector.calculate_psi(ref_series, curr_series)
            drift_scores[f"{column}_psi"] = psi_score
            
            if psi_drift and not column_drift:  # Only alert if not already alerted
                alert = self.alert_manager.create_alert(
                    DriftType.DATA_DRIFT,
                    AlertSeverity.MEDIUM,
                    column,
                    psi_score,
                    self.config.psi_threshold,
                    f"Population stability drift detected in {column}: PSI={psi_score:.4f}"
                )
        
        return drift_detected, drift_scores
    
    def _detect_feature_drift(self, current_data: pd.DataFrame) -> Tuple[bool, Dict[str, float]]:
        """Detect feature-level drift patterns."""
        drift_detected = False
        drift_scores = {}
        
        try:
            # Check correlation drift
            common_numeric_columns = []
            for col in self.reference_data.columns:
                if (col in current_data.columns and 
                    pd.api.types.is_numeric_dtype(self.reference_data[col]) and
                    pd.api.types.is_numeric_dtype(current_data[col])):
                    common_numeric_columns.append(col)
            
            if len(common_numeric_columns) > 1:
                ref_corr = self.reference_data[common_numeric_columns].corr()
                curr_corr = current_data[common_numeric_columns].corr()
                
                # Calculate correlation difference
                corr_diff = np.abs(ref_corr - curr_corr).mean().mean()
                drift_scores['correlation_drift'] = corr_diff
                
                if corr_diff > self.config.correlation_drift_threshold:
                    drift_detected = True
                    
                    alert = self.alert_manager.create_alert(
                        DriftType.FEATURE_DRIFT,
                        AlertSeverity.MEDIUM,
                        "correlation_structure",
                        corr_diff,
                        self.config.correlation_drift_threshold,
                        f"Feature correlation drift detected: average difference={corr_diff:.4f}"
                    )
            
        except Exception as e:
            logger.warning(f"Feature drift detection failed: {e}")
        
        return drift_detected, drift_scores
    
    def _calculate_reference_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate reference statistics for drift detection."""
        stats = {}
        
        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                stats[column] = {
                    'type': 'numeric',
                    'mean': data[column].mean(),
                    'std': data[column].std(),
                    'min': data[column].min(),
                    'max': data[column].max(),
                    'quantiles': data[column].quantile([0.25, 0.5, 0.75]).to_dict()
                }
            else:
                stats[column] = {
                    'type': 'categorical',
                    'value_counts': data[column].value_counts().to_dict(),
                    'unique_count': data[column].nunique()
                }
        
        return stats
    
    def _encode_for_monitoring(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for monitoring."""
        from sklearn.preprocessing import LabelEncoder
        
        encoded_data = data.copy()
        
        for column in data.columns:
            if data[column].dtype == 'object' or data[column].dtype.name == 'category':
                try:
                    le = LabelEncoder()
                    encoded_data[column] = le.fit_transform(data[column].astype(str))
                except:
                    # If encoding fails, drop the column
                    encoded_data = encoded_data.drop(columns=[column])
        
        return encoded_data.fillna(0)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring system summary."""
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            'monitoring_active': self.monitoring_active,
            'reference_data_samples': len(self.reference_data) if self.reference_data is not None else 0,
            'reference_features': len(self.reference_data.columns) if self.reference_data is not None else 0,
            'active_alerts': len(active_alerts),
            'alert_breakdown': {
                severity.value: len([a for a in active_alerts if a.severity == severity])
                for severity in AlertSeverity
            },
            'config': {
                'ks_test_threshold': self.config.ks_test_threshold,
                'psi_threshold': self.config.psi_threshold,
                'concept_drift_threshold': self.config.concept_drift_threshold,
                'monitoring_enabled': self.config.monitoring_enabled
            }
        }
    
    def export_monitoring_report(self, file_path: str) -> bool:
        """Export monitoring report to file."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': self.get_monitoring_summary(),
                'active_alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'drift_type': alert.drift_type.value,
                        'severity': alert.severity.value,
                        'feature_name': alert.feature_name,
                        'drift_score': alert.drift_score,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in self.alert_manager.get_active_alerts()
                ],
                'reference_statistics': self.reference_statistics
            }
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Monitoring report exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export monitoring report: {e}")
            return False


# Factory functions and utilities
def create_drift_monitor(config: Optional[DriftDetectionConfig] = None) -> DriftMonitor:
    """Create a drift monitor instance."""
    return DriftMonitor(config)


def monitor_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame,
                      target_column: Optional[str] = None,
                      config: Optional[DriftDetectionConfig] = None) -> DriftDetectionResult:
    """Convenience function to monitor data drift."""
    monitor = create_drift_monitor(config)
    monitor.set_reference_data(reference_data, target_column)
    return monitor.process(current_data, target_column)


def get_default_drift_config() -> DriftDetectionConfig:
    """Get default drift detection configuration."""
    return DriftDetectionConfig()


def get_sensitive_drift_config() -> DriftDetectionConfig:
    """Get sensitive drift detection configuration."""
    return DriftDetectionConfig(
        ks_test_threshold=0.01,  # More sensitive
        psi_threshold=0.05,
        chi2_threshold=0.01,
        performance_degradation_threshold=0.05,
        missing_data_threshold=0.05,
        enable_alerts=True
    )