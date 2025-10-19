"""
Synthetic data generation using CTGAN for privacy-preserving data augmentation.
Includes quality validation and evaluation metrics for synthetic data.
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier

# Statistical and ML imports
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..core.config import get_config
from ..core.interfaces import DataProcessor
from ..core.logging import get_audit_logger, get_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""

    # CTGAN parameters
    epochs: int = 300
    batch_size: int = 500
    generator_dim: Tuple[int, ...] = (256, 256)
    discriminator_dim: Tuple[int, ...] = (256, 256)
    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    discriminator_steps: int = 1
    log_frequency: bool = True
    verbose: bool = False

    # Data generation parameters
    num_samples: Optional[int] = None  # If None, generates same number as original
    sample_multiplier: float = 1.0  # Multiplier for original data size

    # Quality validation parameters
    enable_quality_validation: bool = True
    statistical_test_threshold: float = 0.05
    utility_test_enabled: bool = True
    privacy_test_enabled: bool = True

    # Evaluation metrics
    evaluate_column_shapes: bool = True
    evaluate_column_pair_trends: bool = True
    evaluate_ml_efficacy: bool = True


@dataclass
class SyntheticDataQuality:
    """Quality metrics for synthetic data."""

    # Statistical similarity
    kolmogorov_smirnov_scores: Dict[str, float] = field(default_factory=dict)
    wasserstein_distances: Dict[str, float] = field(default_factory=dict)
    correlation_similarity: float = 0.0

    # Machine learning efficacy
    ml_efficacy_score: float = 0.0
    feature_importance_similarity: float = 0.0

    # Privacy metrics
    privacy_score: float = 0.0
    nearest_neighbor_distance: float = 0.0

    # Overall quality
    overall_quality_score: float = 0.0


@dataclass
class SyntheticDataResult:
    """Result of synthetic data generation."""

    success: bool
    synthetic_data: Optional[pd.DataFrame]
    quality_metrics: Optional[SyntheticDataQuality]
    generation_time_seconds: float
    validation_passed: bool
    message: str
    model_path: Optional[str] = None


class CTGANWrapper:
    """Wrapper for CTGAN with enhanced functionality."""

    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.model = None
        self.discrete_columns = []
        self.label_encoders = {}
        self.column_info = {}

        # Try to import CTGAN
        try:
            from ctgan import CTGAN

            self.CTGAN = CTGAN
            self.ctgan_available = True
        except ImportError:
            logger.warning(
                "CTGAN not available. Using fallback synthetic data generation."
            )
            self.ctgan_available = False

    def fit(self, data: pd.DataFrame) -> bool:
        """Fit CTGAN model on the data."""
        try:
            if not self.ctgan_available:
                return self._fit_fallback(data)

            # Prepare data for CTGAN
            processed_data, discrete_columns = self._preprocess_data(data)

            # Initialize CTGAN model
            self.model = self.CTGAN(
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                generator_dim=self.config.generator_dim,
                discriminator_dim=self.config.discriminator_dim,
                generator_lr=self.config.generator_lr,
                discriminator_lr=self.config.discriminator_lr,
                discriminator_steps=self.config.discriminator_steps,
                log_frequency=self.config.log_frequency,
                verbose=self.config.verbose,
            )

            # Fit the model
            logger.info("Training CTGAN model...")
            self.model.fit(processed_data, discrete_columns)

            logger.info("CTGAN model training completed")
            return True

        except Exception as e:
            logger.error(f"CTGAN training failed: {e}")
            return False

    def generate(self, num_samples: int) -> Optional[pd.DataFrame]:
        """Generate synthetic data."""
        try:
            if not self.ctgan_available or self.model is None:
                return self._generate_fallback(num_samples)

            # Generate synthetic data
            logger.info(f"Generating {num_samples} synthetic samples...")
            synthetic_data = self.model.sample(num_samples)

            # Post-process the data
            synthetic_data = self._postprocess_data(synthetic_data)

            logger.info("Synthetic data generation completed")
            return synthetic_data

        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
            return None

    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Preprocess data for CTGAN."""
        processed_data = data.copy()
        discrete_columns = []

        # Identify and encode categorical columns
        for column in data.columns:
            if data[column].dtype == "object" or data[column].dtype.name == "category":
                # Label encode categorical columns
                le = LabelEncoder()
                processed_data[column] = le.fit_transform(data[column].astype(str))
                self.label_encoders[column] = le
                discrete_columns.append(column)

                # Store column info
                self.column_info[column] = {
                    "type": "categorical",
                    "categories": le.classes_.tolist(),
                }
            else:
                # Store numeric column info
                self.column_info[column] = {
                    "type": "numeric",
                    "min": float(data[column].min()),
                    "max": float(data[column].max()),
                    "mean": float(data[column].mean()),
                    "std": float(data[column].std()),
                }

        self.discrete_columns = discrete_columns
        return processed_data, discrete_columns

    def _postprocess_data(self, synthetic_data: pd.DataFrame) -> pd.DataFrame:
        """Post-process synthetic data."""
        processed_data = synthetic_data.copy()

        # Decode categorical columns
        for column in self.discrete_columns:
            if column in self.label_encoders:
                # Ensure values are within valid range
                max_label = len(self.label_encoders[column].classes_) - 1
                processed_data[column] = np.clip(
                    processed_data[column].round(), 0, max_label
                )

                # Inverse transform
                processed_data[column] = self.label_encoders[column].inverse_transform(
                    processed_data[column].astype(int)
                )

        # Clip numeric columns to reasonable ranges
        for column in processed_data.columns:
            if column not in self.discrete_columns and column in self.column_info:
                col_info = self.column_info[column]
                if col_info["type"] == "numeric":
                    # Use 3-sigma rule for clipping
                    lower_bound = col_info["mean"] - 3 * col_info["std"]
                    upper_bound = col_info["mean"] + 3 * col_info["std"]

                    # But respect original min/max if they're more restrictive
                    lower_bound = max(lower_bound, col_info["min"])
                    upper_bound = min(upper_bound, col_info["max"])

                    processed_data[column] = np.clip(
                        processed_data[column], lower_bound, upper_bound
                    )

        return processed_data

    def _fit_fallback(self, data: pd.DataFrame) -> bool:
        """Fallback method when CTGAN is not available."""
        logger.info("Using fallback synthetic data generation (statistical sampling)")

        # Store data statistics for fallback generation
        self.fallback_stats = {}

        for column in data.columns:
            if data[column].dtype == "object" or data[column].dtype.name == "category":
                # Store categorical distribution
                value_counts = data[column].value_counts(normalize=True)
                self.fallback_stats[column] = {
                    "type": "categorical",
                    "distribution": value_counts.to_dict(),
                }
            else:
                # Store numeric distribution parameters
                self.fallback_stats[column] = {
                    "type": "numeric",
                    "mean": float(data[column].mean()),
                    "std": float(data[column].std()),
                    "min": float(data[column].min()),
                    "max": float(data[column].max()),
                }

        return True

    def _generate_fallback(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic data using statistical sampling."""
        synthetic_data = {}

        for column, stats in self.fallback_stats.items():
            if stats["type"] == "categorical":
                # Sample from categorical distribution
                categories = list(stats["distribution"].keys())
                probabilities = list(stats["distribution"].values())
                synthetic_data[column] = np.random.choice(
                    categories, size=num_samples, p=probabilities
                )
            else:
                # Sample from normal distribution with clipping
                samples = np.random.normal(
                    stats["mean"], stats["std"], size=num_samples
                )
                synthetic_data[column] = np.clip(samples, stats["min"], stats["max"])

        return pd.DataFrame(synthetic_data)


class SyntheticDataEvaluator:
    """Evaluates quality of synthetic data."""

    def __init__(self, config: SyntheticDataConfig):
        self.config = config

    def evaluate_quality(
        self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> SyntheticDataQuality:
        """Evaluate synthetic data quality."""
        quality = SyntheticDataQuality()

        try:
            # Statistical similarity tests
            if self.config.evaluate_column_shapes:
                quality.kolmogorov_smirnov_scores = self._evaluate_column_distributions(
                    original_data, synthetic_data
                )
                quality.wasserstein_distances = self._evaluate_wasserstein_distances(
                    original_data, synthetic_data
                )

            # Correlation similarity
            quality.correlation_similarity = self._evaluate_correlation_similarity(
                original_data, synthetic_data
            )

            # Machine learning efficacy
            if self.config.evaluate_ml_efficacy:
                quality.ml_efficacy_score, quality.feature_importance_similarity = (
                    self._evaluate_ml_efficacy(original_data, synthetic_data)
                )

            # Privacy evaluation
            if self.config.privacy_test_enabled:
                quality.privacy_score, quality.nearest_neighbor_distance = (
                    self._evaluate_privacy(original_data, synthetic_data)
                )

            # Calculate overall quality score
            quality.overall_quality_score = self._calculate_overall_quality(quality)

            logger.info(
                f"Synthetic data quality evaluation completed. Overall score: {quality.overall_quality_score:.3f}"
            )

        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")

        return quality

    def _evaluate_column_distributions(
        self, original: pd.DataFrame, synthetic: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate column-wise distribution similarity using KS test."""
        ks_scores = {}

        for column in original.columns:
            if column in synthetic.columns:
                if pd.api.types.is_numeric_dtype(original[column]):
                    try:
                        # Kolmogorov-Smirnov test
                        ks_stat, p_value = stats.ks_2samp(
                            original[column].dropna(), synthetic[column].dropna()
                        )
                        ks_scores[column] = 1 - ks_stat  # Higher is better
                    except:
                        ks_scores[column] = 0.0
                else:
                    # For categorical data, compare distributions
                    orig_dist = original[column].value_counts(normalize=True)
                    synth_dist = synthetic[column].value_counts(normalize=True)

                    # Calculate overlap
                    common_categories = set(orig_dist.index) & set(synth_dist.index)
                    if common_categories:
                        overlap_score = sum(
                            min(orig_dist.get(cat, 0), synth_dist.get(cat, 0))
                            for cat in common_categories
                        )
                        ks_scores[column] = overlap_score
                    else:
                        ks_scores[column] = 0.0

        return ks_scores

    def _evaluate_wasserstein_distances(
        self, original: pd.DataFrame, synthetic: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate Wasserstein distances for numeric columns."""
        wasserstein_scores = {}

        numeric_columns = original.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if column in synthetic.columns:
                try:
                    # Calculate Wasserstein distance
                    distance = wasserstein_distance(
                        original[column].dropna(), synthetic[column].dropna()
                    )

                    # Normalize by data range
                    data_range = original[column].max() - original[column].min()
                    if data_range > 0:
                        normalized_distance = distance / data_range
                        wasserstein_scores[column] = max(0, 1 - normalized_distance)
                    else:
                        wasserstein_scores[column] = 1.0
                except:
                    wasserstein_scores[column] = 0.0

        return wasserstein_scores

    def _evaluate_correlation_similarity(
        self, original: pd.DataFrame, synthetic: pd.DataFrame
    ) -> float:
        """Evaluate correlation matrix similarity."""
        try:
            # Get numeric columns
            numeric_columns = original.select_dtypes(include=[np.number]).columns
            common_numeric = [
                col for col in numeric_columns if col in synthetic.columns
            ]

            if len(common_numeric) < 2:
                return 1.0

            # Calculate correlation matrices
            orig_corr = original[common_numeric].corr()
            synth_corr = synthetic[common_numeric].corr()

            # Calculate similarity (using correlation of correlations)
            orig_corr_flat = orig_corr.values[
                np.triu_indices_from(orig_corr.values, k=1)
            ]
            synth_corr_flat = synth_corr.values[
                np.triu_indices_from(synth_corr.values, k=1)
            ]

            if len(orig_corr_flat) > 0:
                correlation_similarity = np.corrcoef(orig_corr_flat, synth_corr_flat)[
                    0, 1
                ]
                return (
                    max(0, correlation_similarity)
                    if not np.isnan(correlation_similarity)
                    else 0.0
                )
            else:
                return 1.0

        except Exception as e:
            logger.warning(f"Correlation similarity evaluation failed: {e}")
            return 0.0

    def _evaluate_ml_efficacy(
        self, original: pd.DataFrame, synthetic: pd.DataFrame
    ) -> Tuple[float, float]:
        """Evaluate machine learning efficacy."""
        try:
            # Assume last column is target (or find a suitable target)
            target_column = None
            for col in ["default", "target", "label", "y"]:
                if col in original.columns:
                    target_column = col
                    break

            if target_column is None:
                # Use last column as target
                target_column = original.columns[-1]

            # Prepare data
            X_orig = original.drop(columns=[target_column])
            y_orig = original[target_column]

            X_synth = synthetic.drop(columns=[target_column], errors="ignore")
            y_synth = synthetic.get(target_column)

            if y_synth is None or len(X_synth.columns) == 0:
                return 0.0, 0.0

            # Align columns
            common_features = [col for col in X_orig.columns if col in X_synth.columns]
            if len(common_features) == 0:
                return 0.0, 0.0

            X_orig = X_orig[common_features]
            X_synth = X_synth[common_features]

            # Handle categorical variables
            X_orig_encoded = self._encode_for_ml(X_orig)
            X_synth_encoded = self._encode_for_ml(X_synth)

            # Train models
            rf_orig = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_synth = RandomForestClassifier(n_estimators=50, random_state=42)

            # Split original data for testing
            X_train, X_test, y_train, y_test = train_test_split(
                X_orig_encoded, y_orig, test_size=0.2, random_state=42
            )

            # Train on original data
            rf_orig.fit(X_train, y_train)
            orig_score = rf_orig.score(X_test, y_test)

            # Train on synthetic data, test on original
            rf_synth.fit(X_synth_encoded, y_synth)
            synth_score = rf_synth.score(X_test, y_test)

            # ML efficacy is how well synthetic-trained model performs on real test data
            ml_efficacy = synth_score / orig_score if orig_score > 0 else 0.0

            # Feature importance similarity
            if hasattr(rf_orig, "feature_importances_") and hasattr(
                rf_synth, "feature_importances_"
            ):
                importance_corr = np.corrcoef(
                    rf_orig.feature_importances_, rf_synth.feature_importances_
                )[0, 1]
                feature_importance_similarity = (
                    max(0, importance_corr) if not np.isnan(importance_corr) else 0.0
                )
            else:
                feature_importance_similarity = 0.0

            return min(1.0, ml_efficacy), feature_importance_similarity

        except Exception as e:
            logger.warning(f"ML efficacy evaluation failed: {e}")
            return 0.0, 0.0

    def _evaluate_privacy(
        self, original: pd.DataFrame, synthetic: pd.DataFrame
    ) -> Tuple[float, float]:
        """Evaluate privacy preservation."""
        try:
            # Simple privacy evaluation: nearest neighbor distance
            # Higher distance means better privacy

            # Sample subset for efficiency
            sample_size = min(1000, len(original), len(synthetic))
            orig_sample = original.sample(n=sample_size, random_state=42)
            synth_sample = synthetic.sample(n=sample_size, random_state=42)

            # Encode categorical variables
            orig_encoded = self._encode_for_ml(orig_sample)
            synth_encoded = self._encode_for_ml(synth_sample)

            # Align columns
            common_cols = [
                col for col in orig_encoded.columns if col in synth_encoded.columns
            ]
            if len(common_cols) == 0:
                return 0.5, 0.0

            orig_encoded = orig_encoded[common_cols]
            synth_encoded = synth_encoded[common_cols]

            # Calculate minimum distances from synthetic to original points
            from sklearn.metrics.pairwise import euclidean_distances

            distances = euclidean_distances(synth_encoded, orig_encoded)
            min_distances = np.min(distances, axis=1)
            avg_min_distance = np.mean(min_distances)

            # Normalize by data scale
            data_scale = np.std(orig_encoded.values)
            normalized_distance = avg_min_distance / data_scale if data_scale > 0 else 0

            # Privacy score: higher distance = better privacy
            privacy_score = min(1.0, normalized_distance / 2.0)  # Normalize to 0-1

            return privacy_score, avg_min_distance

        except Exception as e:
            logger.warning(f"Privacy evaluation failed: {e}")
            return 0.5, 0.0

    def _encode_for_ml(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode data for machine learning."""
        encoded_data = data.copy()

        for column in data.columns:
            if data[column].dtype == "object" or data[column].dtype.name == "category":
                # Simple label encoding
                le = LabelEncoder()
                encoded_data[column] = le.fit_transform(data[column].astype(str))

        return encoded_data.fillna(0)

    def _calculate_overall_quality(self, quality: SyntheticDataQuality) -> float:
        """Calculate overall quality score."""
        scores = []

        # Statistical similarity (40% weight)
        if quality.kolmogorov_smirnov_scores:
            avg_ks = np.mean(list(quality.kolmogorov_smirnov_scores.values()))
            scores.append(avg_ks * 0.2)

        if quality.wasserstein_distances:
            avg_wasserstein = np.mean(list(quality.wasserstein_distances.values()))
            scores.append(avg_wasserstein * 0.2)

        # Correlation similarity (20% weight)
        scores.append(quality.correlation_similarity * 0.2)

        # ML efficacy (30% weight)
        scores.append(quality.ml_efficacy_score * 0.2)
        scores.append(quality.feature_importance_similarity * 0.1)

        # Privacy (10% weight)
        scores.append(quality.privacy_score * 0.1)

        return sum(scores) if scores else 0.0


class SyntheticDataGenerator(DataProcessor):
    """Main synthetic data generation pipeline."""

    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        self.config = config or SyntheticDataConfig()
        self.ctgan_wrapper = CTGANWrapper(self.config)
        self.evaluator = SyntheticDataEvaluator(self.config)
        self.model_trained = False

    def process(
        self, data: pd.DataFrame, target_column: Optional[str] = None
    ) -> SyntheticDataResult:
        """Process synthetic data generation."""
        start_time = datetime.now()

        try:
            logger.info("Starting synthetic data generation")
            logger.info(f"Input data shape: {data.shape}")

            # Prepare data
            if target_column and target_column in data.columns:
                # Ensure target column is last for consistency
                columns = [col for col in data.columns if col != target_column] + [
                    target_column
                ]
                data = data[columns]

            # Train CTGAN model
            logger.info("Training synthetic data model...")
            training_success = self.ctgan_wrapper.fit(data)

            if not training_success:
                raise Exception("Model training failed")

            self.model_trained = True

            # Determine number of samples to generate
            if self.config.num_samples is not None:
                num_samples = self.config.num_samples
            else:
                num_samples = int(len(data) * self.config.sample_multiplier)

            # Generate synthetic data
            logger.info(f"Generating {num_samples} synthetic samples...")
            synthetic_data = self.ctgan_wrapper.generate(num_samples)

            if synthetic_data is None:
                raise Exception("Synthetic data generation failed")

            # Validate synthetic data quality
            quality_metrics = None
            validation_passed = True

            if self.config.enable_quality_validation:
                logger.info("Evaluating synthetic data quality...")
                quality_metrics = self.evaluator.evaluate_quality(data, synthetic_data)

                # Check if quality meets thresholds
                validation_passed = self._validate_quality_thresholds(quality_metrics)

            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Synthetic data generation completed in {processing_time:.2f} seconds"
            )

            # Log synthetic data generation
            audit_logger.log_data_access(
                user_id="system",
                resource="synthetic_data_generator",
                action="synthetic_data_generation",
                success=True,
                details={
                    "original_samples": len(data),
                    "synthetic_samples": len(synthetic_data),
                    "quality_score": (
                        quality_metrics.overall_quality_score
                        if quality_metrics
                        else None
                    ),
                    "processing_time_seconds": processing_time,
                },
            )

            return SyntheticDataResult(
                success=True,
                synthetic_data=synthetic_data,
                quality_metrics=quality_metrics,
                generation_time_seconds=processing_time,
                validation_passed=validation_passed,
                message="Synthetic data generation completed successfully",
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Synthetic data generation failed: {str(e)}"

            logger.error(error_message)

            return SyntheticDataResult(
                success=False,
                synthetic_data=None,
                quality_metrics=None,
                generation_time_seconds=processing_time,
                validation_passed=False,
                message=error_message,
            )

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate synthetic data."""
        try:
            # Basic validation checks
            if data.empty:
                logger.error("Synthetic data is empty")
                return False

            # Check for reasonable data types
            if data.isnull().all().any():
                logger.warning("Some columns in synthetic data are entirely null")

            # Check for infinite values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                if np.isinf(data[numeric_cols]).any().any():
                    logger.error("Infinite values found in synthetic data")
                    return False

            return True

        except Exception as e:
            logger.error(f"Synthetic data validation failed: {e}")
            return False

    def generate_additional_samples(self, num_samples: int) -> Optional[pd.DataFrame]:
        """Generate additional synthetic samples using trained model."""
        if not self.model_trained:
            logger.error("Model not trained. Call process() first.")
            return None

        try:
            return self.ctgan_wrapper.generate(num_samples)
        except Exception as e:
            logger.error(f"Additional sample generation failed: {e}")
            return None

    def save_model(self, model_path: str) -> bool:
        """Save trained model."""
        if not self.model_trained:
            logger.error("No trained model to save")
            return False

        try:
            # Create directory if it doesn't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)

            if self.ctgan_wrapper.ctgan_available and self.ctgan_wrapper.model:
                # Save CTGAN model
                import pickle

                with open(model_path, "wb") as f:
                    pickle.dump(
                        {
                            "model": self.ctgan_wrapper.model,
                            "discrete_columns": self.ctgan_wrapper.discrete_columns,
                            "label_encoders": self.ctgan_wrapper.label_encoders,
                            "column_info": self.ctgan_wrapper.column_info,
                            "config": self.config,
                        },
                        f,
                    )
            else:
                # Save fallback model
                import pickle

                with open(model_path, "wb") as f:
                    pickle.dump(
                        {
                            "fallback_stats": self.ctgan_wrapper.fallback_stats,
                            "config": self.config,
                        },
                        f,
                    )

            logger.info(f"Model saved to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            return False

    def load_model(self, model_path: str) -> bool:
        """Load trained model."""
        try:
            import pickle

            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            if "model" in model_data:
                # Load CTGAN model
                self.ctgan_wrapper.model = model_data["model"]
                self.ctgan_wrapper.discrete_columns = model_data["discrete_columns"]
                self.ctgan_wrapper.label_encoders = model_data["label_encoders"]
                self.ctgan_wrapper.column_info = model_data["column_info"]
            else:
                # Load fallback model
                self.ctgan_wrapper.fallback_stats = model_data["fallback_stats"]

            self.config = model_data["config"]
            self.model_trained = True

            logger.info(f"Model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False

    def _validate_quality_thresholds(
        self, quality_metrics: SyntheticDataQuality
    ) -> bool:
        """Validate if quality metrics meet minimum thresholds."""
        # Define minimum quality thresholds
        min_overall_quality = 0.5
        min_ml_efficacy = 0.3
        min_correlation_similarity = 0.3

        if quality_metrics.overall_quality_score < min_overall_quality:
            logger.warning(
                f"Overall quality score {quality_metrics.overall_quality_score:.3f} below threshold {min_overall_quality}"
            )
            return False

        if quality_metrics.ml_efficacy_score < min_ml_efficacy:
            logger.warning(
                f"ML efficacy score {quality_metrics.ml_efficacy_score:.3f} below threshold {min_ml_efficacy}"
            )
            return False

        if quality_metrics.correlation_similarity < min_correlation_similarity:
            logger.warning(
                f"Correlation similarity {quality_metrics.correlation_similarity:.3f} below threshold {min_correlation_similarity}"
            )
            return False

        return True

    def generate_quality_report(
        self, quality_metrics: SyntheticDataQuality
    ) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        return {
            "overall_quality_score": quality_metrics.overall_quality_score,
            "statistical_similarity": {
                "kolmogorov_smirnov_scores": quality_metrics.kolmogorov_smirnov_scores,
                "wasserstein_distances": quality_metrics.wasserstein_distances,
                "average_ks_score": (
                    np.mean(list(quality_metrics.kolmogorov_smirnov_scores.values()))
                    if quality_metrics.kolmogorov_smirnov_scores
                    else 0.0
                ),
                "average_wasserstein_score": (
                    np.mean(list(quality_metrics.wasserstein_distances.values()))
                    if quality_metrics.wasserstein_distances
                    else 0.0
                ),
            },
            "correlation_analysis": {
                "correlation_similarity": quality_metrics.correlation_similarity
            },
            "machine_learning_efficacy": {
                "ml_efficacy_score": quality_metrics.ml_efficacy_score,
                "feature_importance_similarity": quality_metrics.feature_importance_similarity,
            },
            "privacy_metrics": {
                "privacy_score": quality_metrics.privacy_score,
                "nearest_neighbor_distance": quality_metrics.nearest_neighbor_distance,
            },
            "quality_assessment": {
                "excellent": quality_metrics.overall_quality_score >= 0.8,
                "good": 0.6 <= quality_metrics.overall_quality_score < 0.8,
                "acceptable": 0.4 <= quality_metrics.overall_quality_score < 0.6,
                "poor": quality_metrics.overall_quality_score < 0.4,
            },
        }


# Factory functions and utilities
def create_synthetic_data_generator(
    config: Optional[SyntheticDataConfig] = None,
) -> SyntheticDataGenerator:
    """Create a synthetic data generator instance."""
    return SyntheticDataGenerator(config)


def generate_synthetic_banking_data(
    data: pd.DataFrame,
    target_column: str = "default",
    config: Optional[SyntheticDataConfig] = None,
) -> SyntheticDataResult:
    """Convenience function to generate synthetic banking data."""
    generator = create_synthetic_data_generator(config)
    return generator.process(data, target_column)


def get_default_synthetic_config() -> SyntheticDataConfig:
    """Get default synthetic data configuration."""
    return SyntheticDataConfig()


def get_fast_synthetic_config() -> SyntheticDataConfig:
    """Get fast synthetic data configuration for testing."""
    return SyntheticDataConfig(
        epochs=50,  # Reduced for faster training
        batch_size=500,
        sample_multiplier=0.5,  # Generate fewer samples
        enable_quality_validation=True,
        evaluate_ml_efficacy=False,  # Skip ML evaluation for speed
    )


def get_high_quality_config() -> SyntheticDataConfig:
    """Get high-quality synthetic data configuration."""
    return SyntheticDataConfig(
        epochs=500,  # More epochs for better quality
        batch_size=500,
        generator_dim=(512, 512),  # Larger networks
        discriminator_dim=(512, 512),
        sample_multiplier=1.0,
        enable_quality_validation=True,
        evaluate_ml_efficacy=True,
        evaluate_column_shapes=True,
        evaluate_column_pair_trends=True,
    )
