"""
Feature engineering pipeline for credit risk modeling.
Includes behavioral, financial, temporal, and relational feature extraction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings

from ..core.interfaces import DataProcessor
from ..core.config import get_config
from ..core.logging import get_logger, get_audit_logger


logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering pipeline."""

    # Behavioral features
    enable_behavioral_features: bool = True
    spending_pattern_window: int = 12  # months
    payment_timing_threshold: int = 30  # days

    # Financial features
    enable_financial_features: bool = True
    income_percentile_bins: int = 10
    loan_amount_percentile_bins: int = 10

    # Temporal features
    enable_temporal_features: bool = True
    seasonal_periods: List[int] = field(default_factory=lambda: [3, 6, 12])

    # Outlier detection
    enable_outlier_detection: bool = True
    outlier_contamination: float = 0.1
    outlier_method: str = "isolation_forest"  # "isolation_forest", "lof"

    # Class imbalance handling
    enable_class_balancing: bool = True
    balancing_strategy: str = "smote"  # "smote", "undersample", "smoteenn"
    target_balance_ratio: float = 0.3  # Target minority class ratio


@dataclass
class FeatureImportance:
    """Feature importance information."""

    feature_name: str
    importance_score: float
    feature_type: str  # "behavioral", "financial", "temporal", "relational"
    description: str


@dataclass
class FeatureEngineeringResult:
    """Result of feature engineering process."""

    success: bool
    features: Optional[pd.DataFrame]
    target: Optional[pd.Series]
    feature_names: List[str]
    feature_importance: List[FeatureImportance]
    outliers_detected: int
    class_distribution: Dict[str, int]
    processing_time_seconds: float
    message: str


class BehavioralFeatureExtractor:
    """Extracts behavioral features from customer data."""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral features."""
        behavioral_features = data.copy()

        # Credit utilization patterns
        if all(
            col in data.columns for col in ["num_open_credit_accounts", "credit_score"]
        ):
            behavioral_features["credit_utilization_ratio"] = (
                self._calculate_credit_utilization(data)
            )
            behavioral_features["credit_account_density"] = data[
                "num_open_credit_accounts"
            ] / (data["age"] - 18 + 1)

        # Payment behavior patterns
        if "num_delinquent_accounts" in data.columns:
            behavioral_features["delinquency_rate"] = data[
                "num_delinquent_accounts"
            ] / (data["num_open_credit_accounts"] + 1)
            behavioral_features["payment_reliability_score"] = (
                self._calculate_payment_reliability(data)
            )

        # Spending patterns (derived from loan purpose and amount)
        if all(
            col in data.columns
            for col in ["loan_purpose", "loan_amount_inr", "annual_income_inr"]
        ):
            behavioral_features["spending_category_risk"] = (
                self._calculate_spending_risk(data)
            )
            behavioral_features["loan_to_income_ratio"] = (
                data["loan_amount_inr"] / data["annual_income_inr"]
            )

        # Risk appetite indicators
        if all(
            col in data.columns
            for col in ["loan_amount_inr", "loan_term_months", "annual_income_inr"]
        ):
            behavioral_features["risk_appetite_score"] = self._calculate_risk_appetite(
                data
            )
            behavioral_features["monthly_payment_burden"] = (
                self._calculate_monthly_payment_burden(data)
            )

        # Credit history depth
        if "age" in data.columns:
            behavioral_features["potential_credit_history_years"] = np.maximum(
                data["age"] - 18, 0
            )
            behavioral_features["credit_maturity_score"] = (
                self._calculate_credit_maturity(data)
            )

        logger.info(
            f"Extracted {len(behavioral_features.columns) - len(data.columns)} behavioral features"
        )
        return behavioral_features

    def _calculate_credit_utilization(self, data: pd.DataFrame) -> pd.Series:
        """Calculate credit utilization ratio."""
        # Estimate credit utilization based on credit score and accounts
        base_utilization = (
            850 - data["credit_score"]
        ) / 550  # Inverse relationship with credit score
        account_factor = (
            np.log1p(data["num_open_credit_accounts"]) / 3
        )  # More accounts = potentially higher utilization
        return np.clip(base_utilization * account_factor, 0, 1)

    def _calculate_payment_reliability(self, data: pd.DataFrame) -> pd.Series:
        """Calculate payment reliability score."""
        # Higher score = more reliable
        base_score = 1 - (
            data["num_delinquent_accounts"] / (data["num_open_credit_accounts"] + 1)
        )
        credit_score_factor = data["credit_score"] / 850
        past_default_penalty = 1 - data.get("past_default", 0) * 0.3
        return base_score * credit_score_factor * past_default_penalty

    def _calculate_spending_risk(self, data: pd.DataFrame) -> pd.Series:
        """Calculate spending category risk score."""
        risk_mapping = {
            "Personal Loan": 0.7,
            "Credit Card": 0.8,
            "Business Loan": 0.6,
            "Education Loan": 0.3,
            "Home Loan": 0.2,
        }
        return data["loan_purpose"].map(risk_mapping).fillna(0.5)

    def _calculate_risk_appetite(self, data: pd.DataFrame) -> pd.Series:
        """Calculate risk appetite score."""
        loan_ratio = data["loan_amount_inr"] / data["annual_income_inr"]
        term_factor = data["loan_term_months"] / 240  # Normalize by 20 years
        return np.clip(loan_ratio * term_factor, 0, 2)

    def _calculate_monthly_payment_burden(self, data: pd.DataFrame) -> pd.Series:
        """Calculate estimated monthly payment burden."""
        # Simplified calculation assuming 8% annual interest rate
        monthly_income = data["annual_income_inr"] / 12
        estimated_monthly_payment = (data["loan_amount_inr"] * 0.08 / 12) / (
            1 - (1 + 0.08 / 12) ** (-data["loan_term_months"])
        )
        return estimated_monthly_payment / monthly_income

    def _calculate_credit_maturity(self, data: pd.DataFrame) -> pd.Series:
        """Calculate credit maturity score."""
        potential_years = np.maximum(data["age"] - 18, 0)
        credit_score_factor = data["credit_score"] / 850
        account_diversity = np.log1p(data["num_open_credit_accounts"]) / 3
        return potential_years * credit_score_factor * account_diversity / 10


class FinancialFeatureExtractor:
    """Extracts financial features from customer data."""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract financial features."""
        financial_features = data.copy()

        # Income-based features
        if "annual_income_inr" in data.columns:
            financial_features["income_percentile"] = self._calculate_percentile_rank(
                data["annual_income_inr"], self.config.income_percentile_bins
            )
            financial_features["income_log"] = np.log1p(data["annual_income_inr"])
            financial_features["income_stability_proxy"] = (
                self._calculate_income_stability(data)
            )

        # Loan amount features
        if "loan_amount_inr" in data.columns:
            financial_features["loan_amount_percentile"] = (
                self._calculate_percentile_rank(
                    data["loan_amount_inr"], self.config.loan_amount_percentile_bins
                )
            )
            financial_features["loan_amount_log"] = np.log1p(data["loan_amount_inr"])

        # Debt-to-income analysis
        if "debt_to_income_ratio" in data.columns:
            financial_features["dti_risk_category"] = self._categorize_dti_risk(
                data["debt_to_income_ratio"]
            )
            financial_features["dti_squared"] = data["debt_to_income_ratio"] ** 2

        # Credit score features
        if "credit_score" in data.columns:
            financial_features["credit_score_normalized"] = data["credit_score"] / 850
            financial_features["credit_score_category"] = self._categorize_credit_score(
                data["credit_score"]
            )
            financial_features["credit_score_squared"] = (
                data["credit_score"] / 850
            ) ** 2

        # Financial ratios and derived metrics
        if all(
            col in data.columns
            for col in ["loan_amount_inr", "annual_income_inr", "loan_term_months"]
        ):
            financial_features["loan_affordability_ratio"] = (
                self._calculate_affordability_ratio(data)
            )
            financial_features["financial_leverage"] = (
                self._calculate_financial_leverage(data)
            )

        # Employment type risk
        if "employment_type" in data.columns:
            financial_features["employment_stability_score"] = (
                self._calculate_employment_stability(data)
            )

        logger.info(
            f"Extracted {len(financial_features.columns) - len(data.columns)} financial features"
        )
        return financial_features

    def _calculate_percentile_rank(self, series: pd.Series, bins: int) -> pd.Series:
        """Calculate percentile rank and bin."""
        return pd.qcut(series, q=bins, labels=False, duplicates="drop").fillna(0)

    def _calculate_income_stability(self, data: pd.DataFrame) -> pd.Series:
        """Calculate income stability proxy based on employment type."""
        stability_mapping = {
            "Government": 0.9,
            "Salaried": 0.7,
            "Self-Employed": 0.4,
            "Unemployed": 0.1,
        }
        return data["employment_type"].map(stability_mapping).fillna(0.5)

    def _categorize_dti_risk(self, dti_series: pd.Series) -> pd.Series:
        """Categorize debt-to-income ratio risk."""
        return pd.cut(
            dti_series,
            bins=[0, 0.2, 0.4, 0.6, 1.0, float("inf")],
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
            include_lowest=True,
        )

    def _categorize_credit_score(self, credit_score: pd.Series) -> pd.Series:
        """Categorize credit score."""
        return pd.cut(
            credit_score,
            bins=[0, 580, 670, 740, 800, 850],
            labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
            include_lowest=True,
        )

    def _calculate_affordability_ratio(self, data: pd.DataFrame) -> pd.Series:
        """Calculate loan affordability ratio."""
        monthly_income = data["annual_income_inr"] / 12
        # Estimate monthly payment (simplified)
        estimated_payment = data["loan_amount_inr"] / data["loan_term_months"]
        return estimated_payment / monthly_income

    def _calculate_financial_leverage(self, data: pd.DataFrame) -> pd.Series:
        """Calculate financial leverage indicator."""
        return data["loan_amount_inr"] / (
            data["annual_income_inr"] + data["loan_amount_inr"]
        )

    def _calculate_employment_stability(self, data: pd.DataFrame) -> pd.Series:
        """Calculate employment stability score."""
        stability_scores = {
            "Government": 0.95,
            "Salaried": 0.75,
            "Self-Employed": 0.45,
            "Unemployed": 0.05,
        }
        return data["employment_type"].map(stability_scores).fillna(0.5)


class TemporalFeatureExtractor:
    """Extracts temporal features from time-series data."""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features."""
        temporal_features = data.copy()

        # Age-based temporal features
        if "age" in data.columns:
            temporal_features["age_group"] = self._categorize_age(data["age"])
            temporal_features["age_squared"] = data["age"] ** 2
            temporal_features["age_normalized"] = (data["age"] - 18) / (
                100 - 18
            )  # Normalize to 0-1

        # Loan term temporal features
        if "loan_term_months" in data.columns:
            temporal_features["loan_term_years"] = data["loan_term_months"] / 12
            temporal_features["loan_term_category"] = self._categorize_loan_term(
                data["loan_term_months"]
            )
            temporal_features["is_long_term_loan"] = (
                data["loan_term_months"] > 60
            ).astype(int)

        # Life stage features
        if all(col in data.columns for col in ["age", "marital_status"]):
            temporal_features["life_stage"] = self._determine_life_stage(data)
            temporal_features["family_responsibility_score"] = (
                self._calculate_family_responsibility(data)
            )

        # Career stage features
        if all(col in data.columns for col in ["age", "employment_type"]):
            temporal_features["career_stage"] = self._determine_career_stage(data)
            temporal_features["earning_potential_score"] = (
                self._calculate_earning_potential(data)
            )

        logger.info(
            f"Extracted {len(temporal_features.columns) - len(data.columns)} temporal features"
        )
        return temporal_features

    def _categorize_age(self, age_series: pd.Series) -> pd.Series:
        """Categorize age into groups."""
        return pd.cut(
            age_series,
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=[
                "Young",
                "Early Career",
                "Mid Career",
                "Senior Career",
                "Pre-Retirement",
                "Senior",
            ],
            include_lowest=True,
        )

    def _categorize_loan_term(self, term_series: pd.Series) -> pd.Series:
        """Categorize loan term."""
        return pd.cut(
            term_series,
            bins=[0, 12, 36, 60, 120, 480],
            labels=["Short", "Medium", "Long", "Very Long", "Ultra Long"],
            include_lowest=True,
        )

    def _determine_life_stage(self, data: pd.DataFrame) -> pd.Series:
        """Determine life stage based on age and marital status."""

        def classify_life_stage(row):
            age, marital = row["age"], row["marital_status"]
            if age < 25:
                return "Young Single" if marital == "Single" else "Young Married"
            elif age < 35:
                return f"Early Adult {marital}"
            elif age < 50:
                return f"Mid Life {marital}"
            else:
                return f"Mature {marital}"

        return data[["age", "marital_status"]].apply(classify_life_stage, axis=1)

    def _calculate_family_responsibility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate family responsibility score."""
        base_score = 0.3  # Base responsibility

        # Marital status factor
        marital_factor = (
            data["marital_status"]
            .map({"Single": 0.0, "Married": 0.4, "Divorced": 0.2, "Widowed": 0.3})
            .fillna(0.0)
        )

        # Age factor (peak responsibility in middle age)
        age_factor = np.where(
            data["age"] < 35,
            (data["age"] - 18) / 17 * 0.3,
            np.where(data["age"] < 55, 0.3, 0.3 - (data["age"] - 55) / 45 * 0.2),
        )

        return base_score + marital_factor + age_factor

    def _determine_career_stage(self, data: pd.DataFrame) -> pd.Series:
        """Determine career stage."""

        def classify_career_stage(row):
            age, employment = row["age"], row["employment_type"]
            if employment == "Unemployed":
                return "Unemployed"
            elif age < 25:
                return "Entry Level"
            elif age < 35:
                return "Early Career"
            elif age < 50:
                return "Mid Career"
            elif age < 65:
                return "Senior Career"
            else:
                return "Retirement Age"

        return data[["age", "employment_type"]].apply(classify_career_stage, axis=1)

    def _calculate_earning_potential(self, data: pd.DataFrame) -> pd.Series:
        """Calculate earning potential score."""
        # Base potential by employment type
        employment_potential = (
            data["employment_type"]
            .map(
                {
                    "Government": 0.7,
                    "Salaried": 0.8,
                    "Self-Employed": 0.6,
                    "Unemployed": 0.1,
                }
            )
            .fillna(0.5)
        )

        # Age factor (peak earning years)
        age_factor = np.where(
            data["age"] < 25,
            0.3,
            np.where(
                data["age"] < 45,
                0.3 + (data["age"] - 25) / 20 * 0.7,
                np.where(data["age"] < 65, 1.0 - (data["age"] - 45) / 20 * 0.3, 0.4),
            ),
        )

        return employment_potential * age_factor


class OutlierDetector:
    """Detects outliers in the dataset."""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.outlier_detectors = {
            "isolation_forest": IsolationForest(
                contamination=config.outlier_contamination, random_state=42
            ),
            "lof": LocalOutlierFactor(contamination=config.outlier_contamination),
        }

    def detect_outliers(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Detect outliers in the dataset."""
        if not self.config.enable_outlier_detection:
            return data, np.zeros(len(data), dtype=bool)

        # Select numeric columns for outlier detection
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        numeric_data = data[numeric_columns].fillna(data[numeric_columns].median())

        # Detect outliers
        detector = self.outlier_detectors[self.config.outlier_method]

        if self.config.outlier_method == "isolation_forest":
            outlier_labels = detector.fit_predict(numeric_data)
            outlier_mask = outlier_labels == -1
        else:  # LOF
            outlier_labels = detector.fit_predict(numeric_data)
            outlier_mask = outlier_labels == -1

        # Log outlier detection results
        outlier_count = outlier_mask.sum()
        logger.info(
            f"Detected {outlier_count} outliers using {self.config.outlier_method}"
        )

        # Return data without outliers
        clean_data = data[~outlier_mask].copy()

        return clean_data, outlier_mask


class ClassBalancer:
    """Handles class imbalance in the dataset."""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.balancers = {
            "smote": SMOTE(random_state=42),
            "undersample": RandomUnderSampler(random_state=42),
            "smoteenn": SMOTEENN(random_state=42),
        }

    def balance_classes(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance classes in the dataset."""
        if not self.config.enable_class_balancing:
            return X, y

        # Check current class distribution
        original_distribution = y.value_counts().to_dict()
        logger.info(f"Original class distribution: {original_distribution}")

        # Apply balancing strategy
        balancer = self.balancers[self.config.balancing_strategy]

        # Convert to numpy for resampling
        X_resampled, y_resampled = balancer.fit_resample(X, y)

        # Convert back to pandas
        X_balanced = pd.DataFrame(X_resampled, columns=X.columns)
        y_balanced = pd.Series(y_resampled, name=y.name)

        # Log new distribution
        new_distribution = y_balanced.value_counts().to_dict()
        logger.info(f"Balanced class distribution: {new_distribution}")

        return X_balanced, y_balanced


class FeatureEngineeringPipeline(DataProcessor):
    """Main feature engineering pipeline."""

    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        self.config = config or FeatureEngineeringConfig()
        self.behavioral_extractor = BehavioralFeatureExtractor(self.config)
        self.financial_extractor = FinancialFeatureExtractor(self.config)
        self.temporal_extractor = TemporalFeatureExtractor(self.config)
        self.outlier_detector = OutlierDetector(self.config)
        self.class_balancer = ClassBalancer(self.config)

        # Feature importance tracking
        self.feature_importance_scores = {}

    def process(
        self, data: pd.DataFrame, target_column: str = "default"
    ) -> FeatureEngineeringResult:
        """Process feature engineering pipeline."""
        start_time = datetime.now()

        try:
            logger.info("Starting feature engineering pipeline")

            # Separate features and target
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")

            X = data.drop(columns=[target_column])
            y = data[target_column]

            original_shape = X.shape
            logger.info(f"Input data shape: {original_shape}")

            # Extract behavioral features
            if self.config.enable_behavioral_features:
                X = self.behavioral_extractor.extract_features(X)

            # Extract financial features
            if self.config.enable_financial_features:
                X = self.financial_extractor.extract_features(X)

            # Extract temporal features
            if self.config.enable_temporal_features:
                X = self.temporal_extractor.extract_features(X)

            logger.info(f"After feature extraction: {X.shape}")

            # Handle missing values
            X = self._handle_missing_values(X)

            # Encode categorical variables
            X = self._encode_categorical_variables(X)

            # Detect and remove outliers
            X_clean, outlier_mask = self.outlier_detector.detect_outliers(X)
            y_clean = y[~outlier_mask] if len(outlier_mask) > 0 else y

            # Balance classes
            X_balanced, y_balanced = self.class_balancer.balance_classes(
                X_clean, y_clean
            )

            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(
                X_balanced, y_balanced
            )

            # Get final class distribution
            class_distribution = y_balanced.value_counts().to_dict()

            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Feature engineering completed. Final shape: {X_balanced.shape}"
            )

            # Log feature engineering
            audit_logger.log_data_access(
                user_id="system",
                resource="feature_engineering_pipeline",
                action="feature_extraction",
                success=True,
                details={
                    "original_shape": original_shape,
                    "final_shape": X_balanced.shape,
                    "outliers_removed": (
                        outlier_mask.sum() if len(outlier_mask) > 0 else 0
                    ),
                    "processing_time_seconds": processing_time,
                },
            )

            return FeatureEngineeringResult(
                success=True,
                features=X_balanced,
                target=y_balanced,
                feature_names=list(X_balanced.columns),
                feature_importance=feature_importance,
                outliers_detected=outlier_mask.sum() if len(outlier_mask) > 0 else 0,
                class_distribution=class_distribution,
                processing_time_seconds=processing_time,
                message="Feature engineering completed successfully",
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Feature engineering failed: {str(e)}"

            logger.error(error_message)

            return FeatureEngineeringResult(
                success=False,
                features=None,
                target=None,
                feature_names=[],
                feature_importance=[],
                outliers_detected=0,
                class_distribution={},
                processing_time_seconds=processing_time,
                message=error_message,
            )

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate processed features."""
        try:
            # Check for infinite values
            if np.isinf(data.select_dtypes(include=[np.number])).any().any():
                logger.warning("Infinite values detected in features")
                return False

            # Check for excessive missing values
            missing_percentage = data.isnull().sum().sum() / (
                len(data) * len(data.columns)
            )
            if missing_percentage > 0.5:
                logger.warning(
                    f"High missing value percentage: {missing_percentage:.2%}"
                )
                return False

            # Check feature variance
            numeric_features = data.select_dtypes(include=[np.number])
            zero_variance_features = numeric_features.columns[
                numeric_features.var() == 0
            ]
            if len(zero_variance_features) > 0:
                logger.warning(
                    f"Zero variance features detected: {list(zero_variance_features)}"
                )

            return True

        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            return False

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Separate numeric and categorical columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=["object", "category"]).columns

        # Handle numeric missing values with median imputation
        if len(numeric_columns) > 0:
            numeric_imputer = SimpleImputer(strategy="median")
            data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

        # Handle categorical missing values with mode imputation
        if len(categorical_columns) > 0:
            categorical_imputer = SimpleImputer(strategy="most_frequent")
            data[categorical_columns] = categorical_imputer.fit_transform(
                data[categorical_columns]
            )

        logger.info("Missing values handled")
        return data

    def _encode_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        categorical_columns = data.select_dtypes(include=["object", "category"]).columns

        if len(categorical_columns) == 0:
            return data

        # Use one-hot encoding for categorical variables with few categories
        # Use label encoding for high cardinality categorical variables

        encoded_data = data.copy()

        for column in categorical_columns:
            unique_values = data[column].nunique()

            if unique_values <= 10:  # One-hot encode
                dummies = pd.get_dummies(data[column], prefix=column, drop_first=True)
                encoded_data = pd.concat(
                    [encoded_data.drop(columns=[column]), dummies], axis=1
                )
            else:  # Label encode
                label_encoder = LabelEncoder()
                encoded_data[column] = label_encoder.fit_transform(
                    data[column].astype(str)
                )

        logger.info(f"Encoded {len(categorical_columns)} categorical variables")
        return encoded_data

    def _calculate_feature_importance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> List[FeatureImportance]:
        """Calculate feature importance using correlation and mutual information."""
        feature_importance = []

        try:
            from sklearn.feature_selection import mutual_info_classif
            from scipy.stats import pearsonr

            # Calculate mutual information
            mi_scores = mutual_info_classif(X, y, random_state=42)

            # Calculate correlation for numeric features
            numeric_features = X.select_dtypes(include=[np.number]).columns

            for i, feature in enumerate(X.columns):
                # Determine feature type based on name patterns
                feature_type = self._classify_feature_type(feature)

                # Use mutual information as primary importance score
                importance_score = mi_scores[i]

                # Add correlation information for numeric features
                if feature in numeric_features:
                    try:
                        corr, _ = pearsonr(X[feature], y)
                        importance_score = max(importance_score, abs(corr))
                    except:
                        pass

                feature_importance.append(
                    FeatureImportance(
                        feature_name=feature,
                        importance_score=importance_score,
                        feature_type=feature_type,
                        description=self._get_feature_description(feature),
                    )
                )

            # Sort by importance score
            feature_importance.sort(key=lambda x: x.importance_score, reverse=True)

        except ImportError:
            logger.warning(
                "scikit-learn not available for feature importance calculation"
            )
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")

        return feature_importance

    def _classify_feature_type(self, feature_name: str) -> str:
        """Classify feature type based on name patterns."""
        feature_name_lower = feature_name.lower()

        if any(
            keyword in feature_name_lower
            for keyword in ["behavior", "payment", "spending", "credit_util"]
        ):
            return "behavioral"
        elif any(
            keyword in feature_name_lower
            for keyword in ["income", "loan", "debt", "financial", "dti"]
        ):
            return "financial"
        elif any(
            keyword in feature_name_lower
            for keyword in ["age", "term", "time", "stage", "years"]
        ):
            return "temporal"
        elif any(
            keyword in feature_name_lower
            for keyword in ["account", "relation", "network"]
        ):
            return "relational"
        else:
            return "other"

    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description for feature."""
        descriptions = {
            "credit_utilization_ratio": "Estimated credit utilization based on credit score and accounts",
            "payment_reliability_score": "Payment reliability based on delinquencies and credit history",
            "spending_category_risk": "Risk score based on loan purpose category",
            "risk_appetite_score": "Risk appetite based on loan amount and term relative to income",
            "monthly_payment_burden": "Estimated monthly payment as percentage of income",
            "income_percentile": "Income percentile ranking",
            "loan_affordability_ratio": "Loan affordability based on estimated monthly payments",
            "employment_stability_score": "Employment stability score based on job type",
            "life_stage": "Life stage classification based on age and marital status",
            "career_stage": "Career stage based on age and employment type",
        }

        return descriptions.get(feature_name, f"Engineered feature: {feature_name}")


# Factory functions and utilities
def create_feature_engineering_pipeline(
    config: Optional[FeatureEngineeringConfig] = None,
) -> FeatureEngineeringPipeline:
    """Create a feature engineering pipeline instance."""
    return FeatureEngineeringPipeline(config)


def engineer_banking_features(
    data: pd.DataFrame,
    target_column: str = "default",
    config: Optional[FeatureEngineeringConfig] = None,
) -> FeatureEngineeringResult:
    """Convenience function to engineer banking features."""
    pipeline = create_feature_engineering_pipeline(config)
    return pipeline.process(data, target_column)


def get_default_config() -> FeatureEngineeringConfig:
    """Get default feature engineering configuration."""
    return FeatureEngineeringConfig()


def get_minimal_config() -> FeatureEngineeringConfig:
    """Get minimal feature engineering configuration for testing."""
    return FeatureEngineeringConfig(
        enable_behavioral_features=True,
        enable_financial_features=True,
        enable_temporal_features=False,
        enable_outlier_detection=False,
        enable_class_balancing=False,
    )
